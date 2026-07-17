"""OpenAI-compatible inference client for the DR-RLM inference driver.

Drives the SAME SkyRL env / turn-loop / prompt / tool-facade as RL training, but routes each
turn's generation to an OpenAI-compatible chat-completions API (OpenAI, OpenRouter, a local
vLLM OpenAI server, ...) instead of the trained policy. It fixes three things in
``OpenRouterInferenceClient.generate`` that break general API inference:

  1. PARITY (the important one). The parent decodes ``prompt_token_ids`` into ONE user
     message, collapsing the multi-turn conversation the agent_loop built. We RECONSTRUCT the
     message list (system / user / assistant turns) from the chat-template markers, so the API
     model sees the same logical conversation the policy was trained on. The content is
     identical to training; only the template *wrapper* differs — which is unavoidable for a
     *different* model. For true byte-identity, serve the trained policy via vLLM
     (token-in-token-out) — this client is for running a different model (A2) through the
     unified env.
  2. reasoning models (gpt-5-class) 400 on ``max_tokens`` -> send ``max_completion_tokens``.
  3. configurable ``base_url`` + ``api_key``; drop the hardcoded ``reasoning:{effort:"none"}``.

The ``InferenceEngineInput`` / ``InferenceEngineOutput`` contract and the ``{responses,
response_ids, stop_reasons}`` return shape are exactly what ``SkyRLGymGenerator.agent_loop``
expects (skyrl_gym_generator.py:376-382).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import aiohttp

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineOutput,
)
from examples.train.rlm.openrouter_client import OpenRouterInferenceClient

# ChatML role-turn markers (Qwen3 / most OSS chat templates). A COMPLETE turn is
# ``<|im_start|>role\n...content...<|im_end|>``. The trailing open ``<|im_start|>assistant``
# (the generation prompt) has no closing ``<|im_end|>`` and is intentionally NOT captured —
# the API generates that next assistant turn.
_CHATML_TURN = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.DOTALL)


@dataclass
class OpenAICompatInferenceClient(OpenRouterInferenceClient):
    """OpenAI-compatible chat-completions client with multi-turn message reconstruction."""

    base_url: str = "https://api.openai.com/v1"
    use_max_completion_tokens: bool = True
    """Reasoning models 400 on ``max_tokens`` -> send ``max_completion_tokens``. Set False for
    endpoints (older vLLM OpenAI servers) that only accept ``max_tokens``."""

    def __post_init__(self):
        # Unlike the parent, do NOT require OPENROUTER_API_KEY — api_key comes from from_endpoint.
        pass

    @classmethod
    def from_endpoint(cls, model, tokenizer, base_url, api_key, use_max_completion_tokens=True):
        return cls(
            proxy_url=base_url,
            server_urls=[base_url],
            data_parallel_size=1,
            model_name=model,
            tokenizer=tokenizer,
            api_key=api_key or "",
            base_url=base_url.rstrip("/"),
            use_max_completion_tokens=use_max_completion_tokens,
        )

    def _messages_from_token_ids(self, ids: List[int]) -> List[Dict[str, str]]:
        """Reconstruct the multi-turn message list from tokenized chat history (NO flatten),
        so the API model sees system/user/assistant turns rather than one blob."""
        templated = self.tokenizer.decode(ids, skip_special_tokens=False)
        msgs = [{"role": r, "content": c.strip()} for r, c in _CHATML_TURN.findall(templated)]
        if msgs:
            return msgs
        # Unknown template -> degrade gracefully to a single user message with the plain text.
        return [{"role": "user", "content": self.tokenizer.decode(ids, skip_special_tokens=True)}]

    async def generate(self, input_batch: InferenceEngineInput, model=None) -> InferenceEngineOutput:
        # NOTE: agent_loop calls generate(engine_input, model=self.policy_model_name)
        # (skyrl_gym_generator.py:379). We accept + ignore `model` here — the answering model is
        # this client's own model_name (set via from_endpoint). The parent's generate lacks this
        # param (it only works on the frozen lm_callback path, which calls it without model).
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        sp: Dict[str, Any] = input_batch.get("sampling_params") or {}

        if prompts is not None:
            message_lists: List[List[Dict[str, str]]] = list(prompts)
        elif prompt_token_ids is not None:
            message_lists = [self._messages_from_token_ids(ids) for ids in prompt_token_ids]
        else:
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided.")

        body_template: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": sp.get("temperature", 0.7),
            "top_p": sp.get("top_p", 1.0),
        }
        max_new = sp.get("max_generate_length") or sp.get("max_tokens")
        if max_new:
            key = "max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"
            body_template[key] = max_new
        if sp.get("additional_kwargs"):
            body_template.update(sp["additional_kwargs"])

        session = await self._get_session()
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async def _post_one(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            body = dict(body_template)
            body["messages"] = messages
            last_exc = None
            # 6 attempts, capped exponential backoff (1,2,4,8,16s ≈ 30s total): provider blips
            # come in short bursts (observed gemini 404/5xx flurries) — 3 attempts/3s was thin
            # enough to drop a whole item (HB item 8). The cap keeps a genuinely dead provider
            # from stalling the run; the failure still surfaces after ~30s.
            for attempt in range(6):
                try:
                    async with session.post(
                        url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    if attempt < 5:
                        await asyncio.sleep(min(2 ** attempt, 16))
            raise RuntimeError(f"chat-completions call failed after 6 attempts: {last_exc}") from last_exc

        data_list = await asyncio.gather(*(_post_one(m) for m in message_lists))

        responses: List[str] = []
        response_ids: List[List[int]] = []
        stop_reasons: List[str] = []
        for data in data_list:
            api_usage = data.get("usage", {}) or {}
            self.usage["prompt_tokens"] += api_usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += api_usage.get("completion_tokens", 0)
            self.usage["requests"] += 1
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "") or ""
            responses.append(text)
            response_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
            stop_reasons.append(choice.get("finish_reason") or "stop")

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
            rollout_expert_indices=None,
        )
