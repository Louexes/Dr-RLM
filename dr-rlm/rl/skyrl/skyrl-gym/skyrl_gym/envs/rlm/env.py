import json
import os
import re

from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from .repl import PersistentREPL, REPLResult, _iter_tool_entries


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_RLM_SYSTEM_PROMPT = """\
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you submit a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
4. An `answer` dict, initialized to `{"content": "", "ready": False}`, that you use to submit your final answer (see below).
{custom_tools_section}
When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier:
```repl
# your code here
```

Use variables as buffers to build up your final answer. Make sure to explicitly look through the context in the REPL before answering your query.

IMPORTANT — submitting your final answer:
When (and ONLY when) you are done with the task, submit your final answer from inside a ```repl``` block by writing it into the `answer` dict:
```repl
answer["content"] = "your final answer here"
answer["ready"] = True
```
`answer["content"]` holds the final answer (a string, number, or anything `str()`-able). The run terminates as soon as `answer["ready"]` is set to True, and `answer["content"]` is returned. Do NOT set `answer["ready"] = True` until you have actually completed the task; you may update `answer["content"]` across multiple steps before flipping `ready`.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this". Output to the REPL environment as much as possible.\
"""


# ---------------------------------------------------------------------------
# Per-turn user prompt injection
# ---------------------------------------------------------------------------

_USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    'to answer the original prompt: "{root_prompt}".\n\n'
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)


def _build_user_prompt(root_prompt: Optional[str], iteration: int) -> Dict[str, str]:
    """Build the per-turn user message injected before every model call."""
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your prompt / context yet. "
            "Your next action should be to look through and figure out how to answer the prompt, "
            "so don't just provide a final answer yet.\n\n"
        )
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = safeguard + body
    else:
        prefix = "The history before is your previous interactions with the REPL environment. "
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = prefix + body
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Parsing helpers (from rlm/rlm/utils/parsing.py)
# ---------------------------------------------------------------------------

# Matches: ```repl\n<code>\n```
_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)


def _find_code_block(text: str) -> Optional[str]:
    """Return the LAST ```repl ... ``` code block in the response, or None."""
    matches = _REPL_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def _format_execution_result(result: REPLResult) -> str:
    """Format a REPLResult as a string for display in the conversation (from rlm/rlm/utils/parsing.py)."""
    parts = []
    if result.stdout:
        parts.append(f"\n{result.stdout}")
    if result.stderr:
        parts.append(f"\n{result.stderr}")
    important_vars = {
        k: ""
        for k, v in result.locals.items()
        if not k.startswith("_")
        and k not in ("__builtins__", "__name__", "__doc__")
        and isinstance(v, (str, int, float, bool, list, dict, tuple))
    }
    if important_vars:
        parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    return "\n\n".join(parts) if parts else "No output"


def _format_context_metadata(context_payload) -> str:
    """Build the model-facing 'your context is a ... with ... total characters' line."""
    if isinstance(context_payload, str):
        ctx_type, lengths = "str", [len(context_payload)]
    elif isinstance(context_payload, dict):
        ctx_type = "dict"
        lengths = []
        for chunk in context_payload.values():
            if isinstance(chunk, str):
                lengths.append(len(chunk))
            else:
                try:
                    lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    lengths.append(len(repr(chunk)))
    elif isinstance(context_payload, list):
        ctx_type, lengths = "list", [len(str(c)) for c in context_payload]
    else:
        ctx_type, lengths = type(context_payload).__name__, [len(repr(context_payload))]
    return (
        f"Your context is a {ctx_type} with {sum(lengths)} total characters, "
        f"and is broken up into chunks of char lengths: {lengths}."
    )


def _format_tools_for_prompt(custom_tools: Optional[Dict[str, Any]]) -> Optional[str]:
    """Format custom tools for inclusion in the system prompt."""
    lines = []
    for name, value, description in _iter_tool_entries(custom_tools):
        if callable(value):
            lines.append(f"- `{name}`: {description}" if description else f"- `{name}`: A custom function")
        else:
            lines.append(
                f"- `{name}`: {description}" if description else f"- `{name}`: A custom {type(value).__name__} value"
            )
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------


class BaseRLMEnv(BaseTextEnv):
    """Base class for Recursive Language Model (RLM) environments.

    Provides REPL plumbing, parent/child rollout wiring, the multi-turn loop,
    and answer-dict final-answer detection. Task-specific behavior — reward, system
    prompt, REPL tools — is supplied by subclasses via three override hooks:

      • ``_get_reward(final_answer)``  — score the final answer (default: 0.0)
      • ``_get_system_prompt()``       — return the system prompt template
      • ``_get_repl_tools()``          — return task-specific REPL helpers
                                         (called after ``self._context`` is set,
                                         so closures can capture it)

    See ``examples/train/rlm/multi_paper_env/evidence_rlm_env.py`` for a worked example.

    All per-rollout knobs come through ``extras``:
      • ``repl_timeout`` — REPL execution timeout in seconds (default 180)
      • ``max_turns`` — turn budget (default 10)
      • ``lm_callback`` / ``subcall_fn`` — LM query callbacks injected by the generator
      • ``depth`` — rollout depth in a parent/child tree (default 0)

    Ephemeral user-prompt mechanism
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each turn the model should see a per-turn instruction prompt (e.g.
    "Think step-by-step …") as the last user message, but previous turns'
    prompts must NOT accumulate in the chat history.

    This is achieved by mutating the generator's ``chat_history`` list
    through a stashed reference (``self._chat_history_ref``):

      • ``init()`` appends ``turn0_prompt`` to the returned list and stores
        the reference.
      • ``step()`` pops the stale prompt off the tail (it is still there
        because ``step`` runs *before* the generator appends new messages),
        then includes the next prompt as the last element of observations
        so the generator appends it at the tail for the next turn.

    NOTE: this relies on the generator never copying/rebinding
    ``chat_history`` after ``init()`` — it must remain the same list
    object.  The upstream ``SkyRLGymGenerator`` satisfies this.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        self.max_turns = extras.get("max_turns", 10)
        # Opt-in robustness flags (env-var gated; default OFF => behavior byte-identical).
        self._salvage_on_max_turns = os.environ.get("DR_RLM_SALVAGE_ON_MAX_TURNS", "0") == "1"
        self._finalize_nudge_turns = int(os.environ.get("DR_RLM_FINALIZE_NUDGE_TURNS", "0") or 0)

        self.lm_callback = extras.get("lm_callback", None)
        self.subcall_fn = extras.get("subcall_fn", None)

        self.repl: Optional[PersistentREPL] = None
        self._context: Any = None
        self._tools: Dict[str, Any] = {}
        self._final_answer: Optional[str] = None
        self._final_obj: Optional[Dict[str, Any]] = None  # raw `answer` dict (structured-final)
        self._reward: float = 0.0
        self._turn_index = 0

        # Per-node TRAJECTORY record (one entry per model turn): the model's response, the
        # repl code it ran, the stdout it saw, and whether it submitted. Exposed via
        # get_metrics() so the driver can dump the whole recursion tree for parsing/viz —
        # a durable record of how each design choice changes agent behavior.
        self._trajectory: List[Dict[str, Any]] = []

        # Shared reference to the generator's chat_history list (set in init).
        self._chat_history_ref: Optional[ConversationType] = None

    # ------------------------------------------------------------------
    # Override hooks — subclasses customize task-specific behavior here
    # ------------------------------------------------------------------

    def _get_reward(self, final_answer: str) -> float:
        """Score the final answer. **Subclasses should override.**

        Only called when the rollout produced a final answer; turn-limit
        timeouts and other no-answer terminations score 0 without invoking
        this method.
        """
        return 1.0 if final_answer else 0.0

    def _get_system_prompt(self) -> str:
        """Return the system prompt template. Override for custom prompts.

        The returned string may contain ``{custom_tools_section}`` which the
        base will replace with auto-rendered descriptions of LM-query tools
        (when ``lm_callback`` is set) and the tools from ``_get_repl_tools()``.
        """
        return DEFAULT_RLM_SYSTEM_PROMPT

    def _get_repl_tools(self) -> Dict[str, Any]:
        """Return task-specific REPL helpers as ``{name: callable_or_value}``.

        Called after ``self._context`` is populated, so subclasses can return
        closures that capture per-rollout context. Default: empty.
        """
        return {}

    def _get_context_metadata_text(self, context_payload: Any) -> str:
        """Model-facing description of what `context` holds (the 2nd init message).

        Default: the generic "Your context is a {type} with N total characters" line.
        Override when that framing is wrong for the task — e.g. DR-RLM, where `context`
        is just the research question and all evidence comes from search().
        """
        return _format_context_metadata(context_payload)

    def _get_user_prompt(self, iteration: int) -> Dict[str, str]:
        """The ephemeral per-turn user message (appended each turn, popped next step()).

        Default: the generic "use the REPL environment (which contains the context)"
        scaffold. Override to keep the per-turn framing consistent with the task.
        """
        return _build_user_prompt(self._root_prompt, iteration)

    def _finalize_answer(self, raw_content: str, final_obj: Optional[Dict[str, Any]]) -> str:
        """Turn the model's submitted answer into the string used for reward / metrics.

        Base behavior is identity (return the raw content), so envs that don't use the
        structured ``answer`` dict are unaffected. Subclasses (e.g. ``DrRlmEnv``) override
        this to render structured citations into the report and stash the normalized
        object. ``final_obj`` is the captured ``answer`` dict (``{content, citations}``)
        snapshotted when the model flipped ``answer["ready"]`` True."""
        return raw_content

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        root_prompt = "\n".join(msg["content"] for msg in prompt if msg.get("content"))
        context_payload = self.extras.get("extra_info", {}).get("context_text") or root_prompt
        if isinstance(context_payload, str):
            try:
                decoded = json.loads(context_payload)
                if isinstance(decoded, dict):
                    context_payload = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        self._root_prompt = root_prompt
        self._context = context_payload

        self._tools = self._get_repl_tools() or {}

        self.repl = PersistentREPL(
            timeout=self.extras.get("repl_timeout", 180.0),
            custom_tools=self._tools,
            lm_callback=self.lm_callback,
            subcall_fn=self.subcall_fn,
            # per-node fan-out cap (0 = unlimited); also accept it nested under the dr_rlm
            # config block, which propagates to children by reference.
            max_children=int(
                self.extras.get("max_children_per_node")
                or (self.extras.get("dr_rlm") or {}).get("max_children_per_node", 0)
                or 0
            ),
            # Structured child returns promise "every entry is a dict" — make the REPL keep
            # that promise for budget-exhausted/error entries too (same dr_rlm-block peek
            # as max_children above).
            child_uniform_dict=(
                str((self.extras.get("dr_rlm") or {}).get("child_return_mode", "")) == "structured"
            ),
        )
        self.repl.add_context(context_payload, context_index=0)

        metadata_text = self._get_context_metadata_text(context_payload)
        system_content = self._build_system_prompt()

        self._turn_index = 0
        turn0_prompt = self._get_user_prompt(iteration=0)

        init_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": metadata_text},
            turn0_prompt,
        ]
        # Stash reference — the generator will use this same list object as
        # chat_history throughout the episode, so we can pop the ephemeral
        # prompt off it in step().
        self._chat_history_ref = init_messages
        return init_messages, {}

    def _build_system_prompt(self) -> str:
        template = self._get_system_prompt()

        custom_tools_section = ""
        if self.lm_callback is not None:
            custom_tools_section += (
                "\n5. LM query tools available in the REPL:\n"
                "- `llm_query(prompt)` — make a direct LLM call, returns str\n"
                "- `llm_query_batched(prompts)` — batch LLM calls, returns list[str]\n"
                "- `rlm_query(prompt)` — recursive LM call that spawns a child agent with its own REPL, returns str\n"
                "- `rlm_query_batched(prompts)` — batch recursive calls in parallel, returns list[str]"
            )
        if self._tools:
            tools_formatted = _format_tools_for_prompt(self._tools)
            if tools_formatted:
                section_num = 6 if self.lm_callback is not None else 5
                custom_tools_section += (
                    f"\n{section_num}. Custom tools and data available in the REPL:\n{tools_formatted}"
                )

        return template.replace("{custom_tools_section}", custom_tools_section)

    def _record_turn(self, action, code, result, submitted: bool) -> None:
        """Append one turn to this node's trajectory (truncated, JSON-safe)."""
        def _clip(s, n):
            s = "" if s is None else str(s)
            return s if len(s) <= n else s[:n] + f"… [+{len(s) - n} chars]"
        self._trajectory.append({
            "turn": self._turn_index,
            "response": _clip(action, 24_000),                          # model's full message (reasoning + repl block)
            "code": _clip(code, 12_000) if code is not None else None,  # the ```repl``` block it ran
            "stdout": _clip(getattr(result, "stdout", None), 12_000) if result is not None else None,
            "stderr": _clip(getattr(result, "stderr", None), 4_000) if result is not None else None,
            "submitted": submitted,                                     # did this turn flip answer["ready"]?
        })

    def _try_salvage(self) -> None:
        """On max_turns with nothing submitted, capture a written-but-unsubmitted
        ``answer["content"]`` (a finished report the model never flagged ready=True) so it is
        not discarded. Opt-in via DR_RLM_SALVAGE_ON_MAX_TURNS=1; default off => no-op."""
        if self._final_answer is not None or not getattr(self, "_salvage_on_max_turns", False):
            return
        try:
            ans = self.repl.locals.get("answer") if self.repl is not None else None
            content = ans.get("content") if isinstance(ans, dict) else None
            if isinstance(content, str) and content.strip():
                self._final_obj = dict(ans)
                self._final_answer = self._finalize_answer(content, self._final_obj)
                self._reward = self._get_reward(self._final_answer)
        except Exception:
            pass

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._turn_index += 1

        # Pop the previous turn's ephemeral user prompt from chat_history.
        # step() runs before the generator appends the new assistant+obs
        # messages, so the stale prompt is still the last element.
        self._chat_history_ref.pop()

        done = self.turns >= self.max_turns
        code = _find_code_block(action)

        # Branch 1: model didn't produce a repl block.
        if code is None:
            self._record_turn(action, None, None, submitted=False)
            if done:
                self._try_salvage()
                return self._make_step_output([], done=True)
            obs_text = "[No ```repl``` code block found. Wrap your code in ```repl\\n...\\n``` blocks.]"
            return self._make_step_output([{"role": "user", "content": obs_text}], done=done)

        # Branch 2: execute the repl block.
        result = self.repl.execute(code)

        # Final-answer detection: the model flipped answer["ready"]=True inside the exec'd
        # cell, which the REPL surfaces on result.final_answer (+ result.final_obj).
        final_answer = result.final_answer
        if final_answer is not None:
            # Submission gate hook: a subclass may REJECT this submission once (returning a
            # user-message string) so the model can repair it and resubmit — e.g. DR-RLM's
            # empty-citations bounce. Never bounces on the last turn (no room to resubmit).
            bounce = self._submission_bounce(result.final_obj) if not done else None
            if bounce is not None:
                try:  # reset the ready flag so the model resubmits cleanly
                    ans = self.repl.locals.get("answer")
                    if isinstance(ans, dict):
                        ans["ready"] = False
                except Exception:
                    pass
                self._record_turn(action, code, result, submitted=False)
                return self._make_step_output([{"role": "user", "content": bounce}], done=False)
            self._record_turn(action, code, result, submitted=True)
            self._final_obj = result.final_obj
            # subclasses may render structured citations into the graded report here
            self._final_answer = self._finalize_answer(final_answer, result.final_obj)
            self._reward = self._get_reward(self._final_answer)
            return self._make_step_output([], done=True)

        # Hit max_turns without an answer: salvage a written-but-unsubmitted report, else reward 0.
        if done:
            self._record_turn(action, code, result, submitted=False)
            self._try_salvage()
            return self._make_step_output([], done=True)

        # Otherwise emit the REPL output and continue.
        self._record_turn(action, code, result, submitted=False)
        result_str = _format_execution_result(result)
        _MAX_RESULT_LEN = 20_000
        if len(result_str) > _MAX_RESULT_LEN:
            result_str = result_str[:_MAX_RESULT_LEN] + f"... + [{len(result_str) - _MAX_RESULT_LEN} chars...]"
        obs_text = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}"
        return self._make_step_output([{"role": "user", "content": obs_text}], done=False)

    def _submission_bounce(self, final_obj) -> Optional[str]:
        """Optional one-time submission gate, consulted when the model flips
        ``answer["ready"] = True`` (and it is not the last turn). Return a user-message
        string to REJECT the submission — the episode continues, the message is shown, and
        the model may repair and resubmit — or None to accept. Default: never bounce, so
        base behavior is byte-identical."""
        return None

    def _make_step_output(self, observations: List[Dict[str, str]], done: bool) -> BaseTextEnvStepOutput:
        """Build a step output.

        When not done, appends the next turn's ephemeral user prompt to
        observations so the generator places it at the tail of chat_history.
        It will be popped at the start of the next step().
        """
        if not done:
            next_prompt = self._get_user_prompt(self._turn_index)
            observations = observations + [next_prompt]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=self._reward if done else 0.0,
            done=done,
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns_used": self.turns,
            "final_value_set": self._final_answer is not None,
            "final_answer": self._final_answer,
            "answer_obj": self._final_obj,
            "reward": self._reward,
            "trajectory": self._trajectory,   # per-turn record for parsing/visualization
        }

    def close(self):
        if self.repl is not None:
            self.repl.cleanup()
            self.repl = None
