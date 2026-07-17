"""DR-Tulu flat-baseline ReAct env for SkyRL — faithful to the original Open-Instruct RL.

The flat arm of the controlled comparison trains DR-Tulu *as its authors trained it*
(see docs/DRTULU_RL_FAITHFULNESS_AUDIT.md for the file:line-verified audit), but inside
the same SkyRL trainer as the recursive arm. This env is the SkyRL counterpart of the
original rollout loop (`open_instruct/tool_utils/tool_vllm.py::ToolUseLLM`) and reward
(`open_instruct/search_rewards/longform_rubric_rewards.py`), with the deviations
documented in the audit (local judge, frozen bm25s corpus, chat-turn encoding,
static rubrics).

Byte-parity contract
--------------------
Observation/prompt formats are byte-identical to what OUR SFT data and eval harness
produced (dr_agent client.py parser-mode + LocalBm25sSearchTool), which is itself the
original agent library:

  * tool call     : ``<call_tool name="snippet_search" ...>query</call_tool>``
                    (parser v20250824; ``</call>`` tolerated as closer)
  * tool output   : ``<tool_output><snippet id={call_id}-{i}>\\nTitle: ...\\nURL: ...\\n
                    Snippet: ...\\n</snippet>\\n...</tool_output>`` — call_id =
                    uuid4()[:8], id UNQUOTED, no newline inside the outer wrap
                    (mcp_tools.py:602-608 + tool_parsers.py:428-430)
  * budget error  : ``<tool_output>exceed allowed tool call requests</tool_output>``
                    then the model may keep generating (client.py:758-777)
  * unknown tool  : episode ends (client.py:750-754 breaks when no registered tool
                    matches — the prompt advertises google_search/browse_webpage but
                    only snippet_search is wired, same as the eval baseline)
  * system prompt : yaml ``system_prompt`` verbatim; ``additional_instructions[qtype]``
                    appended to the USER turn as ``question + "\\n\\n" + instr``
                    (auto_search_sft.py:104-184)

Reward (original composite, weights longform_rubric_rewards.REWARD_WEIGHTS)
---------------------------------------------------------------------------
  R = scale × (0.5·rubric + 0.2·citation + 0.2·format + 0.1·num_search_turns)

  * rubric   : judge.score_report_sync on the ``<answer>`` inner text (cites kept,
               tags stripped) — verbatim DR-Tulu prompt/scale/aggregation
  * citation : judge.score_in_context_citations_sync on the FULL single-stream
               transcript (actions + tool outputs), snippets = {id: rendered block}
               — exactly the original's extracted_citations map
  * format   : 0.5·has(<answer>) + 0.3·has(well-formed <cite>) + 0.2·has(≥1 call)
  * search   : min(valid_calls_before_answer / 3, 1); 0 when no <answer> (the
               original computes it on extracted_context, which is None then)
  * no <answer> ⇒ rubric = citation = 0 (format term still credited — verbatim)
  * scale    : original multiplies by 10 (verification_reward); provably a no-op under
               std-normalized GRPO advantages, so we default to 1.0 for cross-arm
               dashboard comparability. Set DR_TULU_REWARD_SCALE=10 for the original.

Config is via env vars (same names the run scripts already export); the optional
``environment.skyrl_gym.dr_tulu`` config section overrides them.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

from .corpus_search import get_backend
from .judge import (
    JudgeConfig,
    score_in_context_citations_sync,
    score_report_sync,
)

# ---------------------------------------------------------------------------
# verbatim regexes (tool_parsers.py v20250824 + format_utils.py)
# ---------------------------------------------------------------------------

_CALL_TOOL_RE = re.compile(r"<call_tool\s+([^>]*?)>(.*?)</call_tool>", re.DOTALL)
_CALL_TOOL_SHORT_RE = re.compile(r"<call_tool\s+([^>]*?)>(.*?)</call>", re.DOTALL)
_ATTR_RE = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_ANSWER_INNER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# verbatim from format_utils.compute_format_reward
_CITE_FORMAT_RE = re.compile(r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>", re.DOTALL)


def _parse_call_blocks(text: str) -> List[Tuple[Dict[str, str], str]]:
    """All ``<call_tool ...>content</call_tool>`` blocks as (attrs, content).
    Mirrors extract_search_tool_calls: primary closer first, ``</call>`` fallback."""
    matches = _CALL_TOOL_RE.findall(text)
    if not matches:
        matches = _CALL_TOOL_SHORT_RE.findall(text)
    out = []
    for attr_str, content in matches:
        attrs = {m.group(1): (m.group(2) or m.group(3) or m.group(4) or "") for m in _ATTR_RE.finditer(attr_str)}
        out.append((attrs, content))
    return out


def compute_format_reward(response: str) -> float:
    """Verbatim port of format_utils.compute_format_reward (mcp v20250824 path)."""
    answer_r = 1.0 if _ANSWER_RE.search(response) else 0.0
    cite_r = 1.0 if _CITE_FORMAT_RE.search(response) else 0.0
    queries = [c for _a, c in _parse_call_blocks(response) if c.strip()]
    query_r = 1.0 if queries else 0.0
    return 0.5 * answer_r + 0.3 * cite_r + 0.2 * query_r


def compute_search_turns_reward(context: Optional[str], upper_bound: int = 3) -> float:
    """Verbatim port of search_utils.score_num_in_context_search_turns. ``context`` is
    the transcript BEFORE ``<answer>`` (None when there is no answer → 0.0)."""
    if not context:
        return 0.0
    n = len([c for _a, c in _parse_call_blocks(context) if c.strip()])
    return min(float(n) / upper_bound, 1.0)


# ---------------------------------------------------------------------------
# observation rendering (byte-parity with LocalBm25sSearchTool + parser wrap)
# ---------------------------------------------------------------------------

def _wrap(formatted: str) -> str:
    """UnifiedToolCallParserV20250824.format_result — no added newlines."""
    return f"<tool_output>{formatted}</tool_output>"


def _doc_body(contents: str, url: str) -> Tuple[str, str]:
    """(title, stringify body) — ports local_bm25s.py:144-152 + Document.stringify."""
    title = (contents.split(". ", 1)[0] or url or "Document").strip()[:120] or "Document"
    lines = [f"Title: {title or 'No title available'}"]
    if url:
        lines.append(f"URL: {url}")
    lines.append(f"Snippet: {contents or 'No content available'}")
    return title, "\n".join(lines)


def render_search_output(hits: List[Dict[str, Any]], call_id: str) -> Tuple[str, Dict[str, str]]:
    """MCPSearchTool._format_output: ``<snippet id={call_id}-{i}>\\n{body}\\n</snippet>``
    joined by ``\\n``. Returns (wrapped text, {snippet_id: body}) — the body is what the
    original citation reward stores as the snippet text for that id."""
    blocks, snippets = [], {}
    for i, h in enumerate(hits):
        contents = h.get("snippet") or ""
        _title, body = _doc_body(contents, h.get("url") or "")
        sid = f"{call_id}-{i}"
        snippets[sid] = body
        blocks.append(f"<snippet id={sid}>\n{body}\n</snippet>")
    return _wrap("\n".join(blocks)), snippets


# ---------------------------------------------------------------------------
# system prompt (yaml, module-cached)
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_FILE = "/gpfs/home5/lgehringer/Dr-RLM/dr-tulu/agent/dr_agent/shared_prompts/unified_tool_calling_v20250907.yaml"
_PROMPT_CACHE: Dict[str, dict] = {}


def _load_prompt(path: str) -> dict:
    if path not in _PROMPT_CACHE:
        import yaml

        with open(path) as f:
            _PROMPT_CACHE[path] = yaml.safe_load(f)
    return _PROMPT_CACHE[path]


# ---------------------------------------------------------------------------
# the env
# ---------------------------------------------------------------------------

class DrTuluEnv(BaseTextEnv):
    """Flat DR-Tulu ReAct episode over the frozen bm25s corpus, composite reward."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        spec = extras["reward_spec"]
        self.query: str = str(spec.get("query") or "")
        self.rubrics: List[dict] = list(spec.get("rubrics") or [])
        self.question_type: str = str(extras.get("question_type") or "long_form")
        self.max_turns = int(extras.get("max_turns", 14))

        cfg = {k: v for k, v in dict(env_config or {}).items()}
        _env = os.environ.get
        self.prompt_file = cfg.get("prompt_file", _env("DR_TULU_PROMPT_FILE", _DEFAULT_PROMPT_FILE))
        self.search_k = int(cfg.get("search_k", _env("DR_TULU_SEARCH_K", "10")))
        self.max_tool_calls = int(cfg.get("max_tool_calls", _env("DR_TULU_MAX_TOOL_CALLS", "12")))
        self.reward_scale = float(cfg.get("reward_scale", _env("DR_TULU_REWARD_SCALE", "1.0")))
        self.backend_payload = {
            "search_backend": cfg.get("search_backend", _env("SEARCH_BACKEND", "bm25s")),
            "search_corpus_path": cfg.get("search_corpus_path", _env("SEARCH_CORPUS_PATH")),
            "search_index_path": cfg.get("search_index_path", _env("SEARCH_INDEX_PATH")),
            "search_endpoint": cfg.get("search_endpoint", _env("SEARCH_ENDPOINT")),
        }
        self.judge_cfg = JudgeConfig.from_env_payload(
            {
                "judge_model": cfg.get("judge_model", _env("JUDGE_MODEL", "Qwen/Qwen3.5-4B")),
                "judge_base_url": cfg.get("judge_base_url", _env("JUDGE_BASE_URL", "http://localhost:8100/v1")),
                "judge_api_key_env": "JUDGE_API_KEY",
                "judge_max_concurrency": cfg.get("judge_max_concurrency", _env("DR_TULU_JUDGE_CONCURRENCY", "5")),
            }
        )

        # episode state
        self.tool_calls_used = 0
        self.snippets: Dict[str, str] = {}  # id -> rendered body (the citations map)
        self.transcript: List[str] = []  # single-stream reconstruction (actions + obs)
        self._components: Dict[str, float] = {}

    # -- prompt composition (auto_search_sft.py:104-184) -----------------------
    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        prompts = _load_prompt(self.prompt_file)
        system = prompts["system_prompt"]
        instr = (prompts.get("additional_instructions") or {}).get(self.question_type)
        out = [{"role": "system", "content": system}]
        for msg in prompt:
            if msg.get("role") == "system":
                continue  # our system message is the harness's, not the data's
            m = dict(msg)
            if m.get("role") == "user" and instr:
                m["content"] = m["content"] + "\n\n" + instr
                instr = None  # first user turn only
            out.append(m)
        return out, {}

    # -- one ReAct step ---------------------------------------------------------
    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.transcript.append(action)

        calls = _parse_call_blocks(action)
        name = calls[0][0].get("name", "") if calls else None
        out_of_turns = self.turns >= self.max_turns

        # no tool call, or a tool the harness doesn't provide -> episode over
        # (client.py breaks in both cases; <answer> needs no special-casing)
        if not calls or name != "snippet_search":
            return self._final_step()

        if self.tool_calls_used >= self.max_tool_calls:
            obs = _wrap("exceed allowed tool call requests")
        else:
            obs = self._execute_search(calls[0][1], calls[0][0])
            self.tool_calls_used += 1

        self.transcript.append(obs)
        if out_of_turns:
            return self._final_step()
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": obs}], reward=0.0, done=False, metadata={}
        )

    def _execute_search(self, content: str, attrs: Dict[str, str]) -> str:
        query = content.strip()
        if not query:
            return _wrap("No valid query found in tool call.")
        k = self.search_k
        if "num_results" in attrs:  # per-call override, mirrors LocalBm25sSearchTool
            try:
                k = int(attrs["num_results"])
            except (ValueError, TypeError):
                pass
        backend = get_backend(self.backend_payload)
        hits = (backend.search(query, k) if backend else None) or []
        if not hits:
            return _wrap("No results found for the query.")
        call_id = str(uuid.uuid4())[:8]
        obs, new_snippets = render_search_output(hits, call_id)
        self.snippets.update(new_snippets)
        return obs

    # -- terminal reward (longform_rubric_rewards composite) --------------------
    def _final_step(self) -> BaseTextEnvStepOutput:
        full = "".join(self.transcript)
        m = _ANSWER_INNER_RE.search(full)
        answer = m.group(1).strip() if m else None
        context = full[: m.start()] if m else None  # extract_answer_context_citations

        fmt = compute_format_reward(full)
        srch = compute_search_turns_reward(context)
        if answer:
            rubric, per_criterion = score_report_sync(answer, self.query, self.rubrics, self.judge_cfg)
            cite = score_in_context_citations_sync(self.query, full, self.snippets, self.judge_cfg)
        else:
            rubric, cite, per_criterion = 0.0, 0.0, {}

        R = self.reward_scale * (0.5 * rubric + 0.2 * cite + 0.2 * fmt + 0.1 * srch)
        self._components = {
            "reward": R, "rubric": rubric, "citation": cite, "format": fmt,
            "num_search_turns": srch, "finalized": 1.0 if answer else 0.0,
            "tool_calls": float(self.tool_calls_used), "snippets": float(len(self.snippets)),
        }
        return BaseTextEnvStepOutput(
            observations=[], reward=R, done=True,
            metadata={"components": dict(self._components), "per_criterion": per_criterion},
        )

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._components)
