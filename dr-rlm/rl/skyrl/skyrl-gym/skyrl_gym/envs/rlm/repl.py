import io
import os
import resource
import shutil
import signal
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Final-answer buffer — the model submits via answer["ready"] = True
# ---------------------------------------------------------------------------


class _AnswerDict(dict):
    """REPL-visible dict where ``answer["ready"] = True`` signals completion.

    Behaves exactly like ``dict`` for the model, but fires ``on_ready`` the first
    time ``ready`` flips truthy, passing itself so the env can snapshot both
    ``content`` and the optional ``citations`` list at that instant — event-driven
    capture, mirroring upstream RLM's ``local_repl._AnswerDict``. The next
    ``execute`` then surfaces it on ``REPLResult.final_answer`` / ``final_obj``.
    """

    def __init__(self, on_ready: Optional[Callable[["_AnswerDict"], None]] = None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Safe builtins — blocks eval/exec/compile/input/globals/locals
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    # Blocked to prevent sandbox escape. To expose specific libraries to the RLM,
    # pre-bind them into self.globals in setup() (e.g. self.globals["math"] = math)
    # so the model can use them without needing __import__ or open.
    "__import__": None,
    "open": None,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked (None = raises NameError on access)
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}

# Names that are always restored after every execution so model overwrites don't persist.
RESERVED_TOOL_NAMES: frozenset = frozenset(
    {
        "answer",
        "SHOW_VARS",
        "context",
        "llm_query",
        "llm_query_batched",
        "rlm_query",
        "rlm_query_batched",
    }
)


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: Dict[str, Any]  # snapshot of self.locals after execution
    final_answer: Optional[str]  # set when the model flips answer["ready"]=True
    final_obj: Optional[Dict[str, Any]] = None  # the captured `answer` dict when ready (content + citations)


def _can_use_sigalrm() -> bool:
    """SIGALRM is only usable from the main thread on Unix."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


_MEM_GUARD_INSTALLED = False


def _install_memory_guard() -> None:
    """Cap anonymous heap so a runaway model-written cell raises MemoryError instead of
    OOM-killing the host process.

    Cells ``exec()`` IN THIS PROCESS with only a wall-clock timeout, and the thread-fallback
    timeout can only ABANDON a runaway thread — abandoned code keeps allocating as a leaked
    daemon thread. Observed 2026-07-05: the RL trainer driver ballooned 5 GB -> 535 GB during
    rollout generation and Ray OOM-killed the whole training job (24457033) at step 6.

    RLIMIT_DATA (Linux >= 4.7) counts brk + private anonymous mmap — i.e. Python/numpy heap —
    but NOT file-backed maps (Ray plasma /dev/shm objects, mmapped bm25s indexes), so the cap
    bounds exactly the thing model code can blow up. Legit driver heap is ~5 GB; default cap
    64 GB. On breach, malloc fails inside the offending cell -> MemoryError -> caught by
    execute() and returned to the model as a traceback; training continues.
    Tune/disable via RLM_REPL_MEM_CAP_GB (<=0 disables). Process-wide, installed once.
    """
    global _MEM_GUARD_INSTALLED
    if _MEM_GUARD_INSTALLED:
        return
    _MEM_GUARD_INSTALLED = True
    try:
        cap_gb = float(os.environ.get("RLM_REPL_MEM_CAP_GB", "64"))
        if cap_gb <= 0:
            return
        cap = int(cap_gb * 1024**3)
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        new_soft = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
        if soft == resource.RLIM_INFINITY or soft > new_soft:
            resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard))
    except Exception:
        pass  # never let the guard break REPL construction (e.g. non-Linux)


class PersistentREPL:
    """
    A persistent Python REPL that maintains state (variables) across
    multiple execute() calls. Used by RLMEnv to give the model a stateful
    programming environment it can interact with across turns.

    Globals hold builtins and scaffold functions (SHOW_VARS, and the
    llm_query/rlm_query family when enabled); locals hold user-created
    variables, the context payload, and the reserved ``answer`` submission dict.
    After every execution scaffold names are restored so the model cannot
    permanently overwrite them.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        custom_tools: Optional[Dict[str, Any]] = None,
        lm_callback: Optional[Callable[[List[str]], List[str]]] = None,
        subcall_fn: Optional[Callable[[str], str]] = None,
        max_children: int = 0,
        child_uniform_dict: bool = False,
    ):
        self.timeout = timeout
        self.custom_tools: Dict[str, Any] = custom_tools or {}
        self.lm_callback = lm_callback
        self.subcall_fn = subcall_fn
        # Per-NODE fan-out cap: total child sub-agents this node may spawn across all its
        # rlm_query / rlm_query_batched calls (0 = unlimited). Bounds tree WIDTH (the tool-call
        # budget bounds Serper calls, not the number of sub-agents). Counted across turns.
        self.max_children = int(max_children or 0)
        self._children_spawned = 0
        # When the child->parent contract promises dicts (DR-RLM structured mode), EVERY
        # rlm_query(_batched) entry must honor it — including budget-exhausted and error
        # messages, which are otherwise raw strings. A single string in `subs` makes the
        # model's `subs[i]["content"]` raise (observed: TypeError -> dead turns -> uncited
        # report). With this flag, non-dict entries wrap as {"content": msg, "citations": []}.
        self.child_uniform_dict = bool(child_uniform_dict)
        # Cell-clock bookkeeping (see execute/_pause_timeout): set defensively here so the
        # query wrappers can run even outside execute() (e.g. direct calls in tests).
        self._exec_start = time.monotonic()
        self._wait_secs = 0.0
        self._wait_active_since = None
        self._alarm_armed = False
        self.temp_dir = tempfile.mkdtemp(prefix="skyrl_repl_")
        _install_memory_guard()
        self._validate_custom_tools()
        self.setup()

    _CHILD_BUDGET_MSG = (
        "[sub-agent budget exhausted: this node has already spawned its maximum number of "
        "child sub-agents; do NOT delegate further — synthesize your answer from the results "
        "you already have.]"
    )

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self):
        self.globals: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: Dict[str, Any] = {}
        self._last_final_answer: Optional[str] = None
        self._last_final_obj: Optional[Dict[str, Any]] = None
        # `answer` is the reserved, mutable submission buffer the model builds across turns
        # (upstream RLM's dict-based final-answer format). As an _AnswerDict it captures the
        # submission the instant the model flips `ready` True — the single detection point.
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

        self._exec_combined: Optional[Dict[str, Any]] = None  # live combined dict during exec
        self.globals["SHOW_VARS"] = self._show_vars

        if self.lm_callback is not None:
            self.globals["llm_query"] = self._llm_query
            self.globals["llm_query_batched"] = self._llm_query_batched
            self.globals["rlm_query"] = self._rlm_query
            self.globals["rlm_query_batched"] = self._rlm_query_batched

        for name, value, _ in _iter_tool_entries(self.custom_tools):
            if callable(value):
                self.globals[name] = value
            else:
                self.locals[name] = value

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def add_context(self, context_payload, context_index: int = 0):
        """Bind the context payload directly into the REPL namespace.

        Assigning to self.locals avoids depending on `open`/`__import__` inside
        the sandbox (both are blocked) and skips an unnecessary temp-file
        round-trip.
        """
        var_name = f"context_{context_index}"
        self.locals[var_name] = context_payload
        if context_index == 0:
            self.locals["context"] = context_payload

    # ------------------------------------------------------------------
    # Scaffold functions injected into the REPL namespace
    # ------------------------------------------------------------------

    def _capture_answer(self, answer_dict) -> None:
        """Snapshot the submission the instant ``answer["ready"]`` flips True.

        Captures ``content`` (``str``-ified, so list/dict answers round-trip the same way the
        old ``FINAL_VAR(var)`` path did via ``str(var)``) and the optional ``citations`` list
        (consumed only by the DR-RLM ``ledger_support`` ablation; default provenance is
        scraped from inline ``<cite>`` tags downstream)."""
        self._last_final_obj = {
            "content": str(answer_dict.get("content", "")),
            "citations": list(answer_dict.get("citations") or []),
        }
        self._last_final_answer = self._last_final_obj["content"]

    def _show_vars(self) -> str:
        """Show all user-created variables in the REPL."""
        lookup = self._exec_combined if self._exec_combined is not None else self.locals
        available = {
            k: type(v).__name__
            for k, v in lookup.items()
            if not k.startswith("_") and k not in self.globals and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    # ------------------------------------------------------------------
    # LM query functions (registered only when lm_callback is provided)
    # ------------------------------------------------------------------

    def _llm_query(self, prompt: str, model: Optional[str] = None) -> str:
        """Make a direct LLM call. Returns the response as a string."""
        try:
            with self._pause_timeout():
                return self.lm_callback([prompt])[0]
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: List[str], model: Optional[str] = None) -> List[str]:
        """Make batched LLM calls. Returns list of responses in the same order as prompts."""
        try:
            with self._pause_timeout():
                return self.lm_callback(prompts)
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    @staticmethod
    def _parse_child_result(result: Any) -> Any:
        """Normalize a child's result for the parent's REPL.

        A child may return a structured object directly (e.g. DR-RLM hands back the
        ``{"content", "citations"}`` answer dict, so the parent can merge citations
        programmatically) — in that case return it unchanged. Otherwise the child
        returned its final answer as a string (via tokenizer.decode); if that string
        is a valid Python literal (e.g. a list) parse it back to a native object.
        """
        import ast

        if not isinstance(result, str):
            return result
        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result

    def _rlm_query(self, prompt: str, model: Optional[str] = None, context: Any = None) -> Any:
        """Spawn a child RLM agent with its own REPL for deeper reasoning on a subtask.

        Falls back to a plain llm_query if no subcall_fn is configured.

        Args:
            context: If provided, overrides the child's REPL ``context`` variable
                (e.g. a single paper string instead of the parent's full dict).
        """
        if self.subcall_fn is not None:
            if self.max_children and self._children_spawned >= self.max_children:
                return self._wrap_child(self._CHILD_BUDGET_MSG)
            self._children_spawned += 1
            try:
                with self._pause_timeout():
                    result = self.subcall_fn(prompt, context=context)
                return self._wrap_child(self._parse_child_result(result))
            except Exception as e:
                return self._wrap_child(f"Error: RLM query failed - {e}")
        return self._llm_query(prompt, model)

    def _rlm_query_batched(
        self, prompts: List[str], model: Optional[str] = None, context_list: Optional[List[Any]] = None
    ) -> List[Any]:
        """Spawn child RLM agents for multiple prompts in parallel.

        Results are returned in the same order as input prompts.
        Falls back to llm_query_batched if no subcall_fn is configured.

        Args:
            context_list: If provided, must be the same length as *prompts*.
                Each element overrides the child's REPL ``context`` variable
                (e.g. a single paper string instead of the parent's full dict).
        """
        if self.subcall_fn is not None:
            contexts = context_list or [None] * len(prompts)
            # Per-node fan-out budget: spawn only up to the remaining allowance; prompts beyond it
            # get the budget-exhausted message so the returned list keeps its length and order.
            if self.max_children:
                n_spawn = max(0, min(len(prompts), self.max_children - self._children_spawned))
            else:
                n_spawn = len(prompts)
            self._children_spawned += n_spawn
            capped = [self._wrap_child(self._CHILD_BUDGET_MSG)] * (len(prompts) - n_spawn)
            spawn_prompts, spawn_ctx = prompts[:n_spawn], contexts[:n_spawn]
            if not spawn_prompts:
                return capped

            results: List[Any] = [""] * len(spawn_prompts)

            def _run(index: int, prompt: str, context: Any) -> None:
                try:
                    results[index] = self._wrap_child(
                        self._parse_child_result(self.subcall_fn(prompt, context=context))
                    )
                except Exception as e:
                    results[index] = self._wrap_child(f"Error: RLM query failed - {e}")

            with self._pause_timeout():
                if len(spawn_prompts) == 1:
                    _run(0, spawn_prompts[0], spawn_ctx[0])
                else:
                    with ThreadPoolExecutor(max_workers=min(2, len(spawn_prompts))) as executor:
                        futures = [executor.submit(_run, i, p, c) for i, (p, c) in enumerate(zip(spawn_prompts, spawn_ctx))]
                        for f in as_completed(futures):
                            f.result()

            return results + capped

        return self._llm_query_batched(prompts, model)

    def _wrap_child(self, entry: Any) -> Any:
        """Honor the uniform child->parent contract when enabled: every rlm_query(_batched)
        entry the model sees is a {"content", "citations"} dict — including budget-exhausted
        and error messages — so `subs[i]["content"]` can never raise on a stray string."""
        if self.child_uniform_dict and not isinstance(entry, dict):
            return {"content": str(entry), "citations": []}
        return entry

    def _restore_scaffold(self):
        """Restore reserved names after execution so model overwrites don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                # `answer` is the submission buffer the model accumulates across turns.
                # Normally it stays an _AnswerDict, so flipping `ready` already fired the
                # capture via the callback. If the model rebound it to a plain dict, the
                # callback never ran: capture here when `ready` is set, then re-wrap so the
                # next cell can signal. Mirrors upstream RLM's local_repl._restore_scaffold.
                current = self.locals.get("answer")
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_obj is None:
                            self._last_final_obj = {
                                "content": str(current.get("content", "")),
                                "citations": list(current.get("citations") or []),
                            }
                            self._last_final_answer = self._last_final_obj["content"]
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "llm_query" and self.lm_callback is not None:
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched" and self.lm_callback is not None:
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query" and self.lm_callback is not None:
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched" and self.lm_callback is not None:
                self.globals["rlm_query_batched"] = self._rlm_query_batched

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, code: str) -> REPLResult:
        # SIGALRM is preferred because the OS delivers the signal directly into
        # the executing frame, cleanly interrupting even infinite loops or
        # blocking I/O.  The thread-based fallback can only *abandon* a stuck
        # thread (Python has no way to kill a thread), so runaway code keeps
        # burning CPU as a leaked daemon thread.
        #
        # However, SIGALRM is only available on Unix *and* only from the main
        # thread (Python raises ValueError if you call signal.signal() from a
        # non-main thread).  In practice this means Ray workers — which run env
        # logic on spawned threads — must use the thread fallback.
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error_str: Optional[str] = None

        # Time spent BLOCKED on sub-agents / sub-LLM calls must not count against the cell
        # clock: a deep-research child legitimately runs for minutes, and killing the
        # spawning cell at `timeout` orphans the children's results (`subs` never assigned;
        # the parent then writes an UNCITED report from stdout fragments — observed on 5/16
        # DRB items). The query wrappers accumulate their wait into `_wait_secs` via
        # `_pause_timeout()`; both timeout paths below charge only (elapsed - _wait_secs).
        self._exec_start = time.monotonic()
        self._wait_secs = 0.0
        self._wait_active_since = None   # set while a wait is IN PROGRESS (thread path)
        self._alarm_armed = False

        if _can_use_sigalrm():
            error_str = self._execute_with_sigalrm(code, stdout_buf, stderr_buf)
        else:
            error_str = self._execute_with_thread_timeout(code, stdout_buf, stderr_buf)

        final_answer = self._last_final_answer
        final_obj = self._last_final_obj
        self._last_final_answer = None
        self._last_final_obj = None

        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue() + (error_str or ""),
            locals=self.locals.copy(),
            final_answer=final_answer,
            final_obj=final_obj,
        )

    def _execute_with_sigalrm(self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> Optional[str]:
        def _raise_timeout(*_):
            raise TimeoutError("Code execution timed out")

        old_alarm = None
        try:
            old_alarm = signal.signal(signal.SIGALRM, _raise_timeout)
            self._alarm_armed = True
            signal.alarm(int(self.timeout))
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                combined = {**self.globals, **self.locals}
                self._exec_combined = combined
                exec(code, combined, combined)
                self._exec_combined = None
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value
                self._restore_scaffold()
        except TimeoutError:
            self._exec_combined = None
            return f"Timeout after {int(self.timeout)} seconds\n"
        except Exception:
            self._exec_combined = None
            return traceback.format_exc()
        finally:
            self._alarm_armed = False
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        return None

    def _execute_with_thread_timeout(
        self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO
    ) -> Optional[str]:
        """Fallback when SIGALRM is unavailable (e.g. non-main thread in Ray workers)."""
        result: dict = {"error": None}

        def _run():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    combined = {**self.globals, **self.locals}
                    self._exec_combined = combined
                    exec(code, combined, combined)
                    self._exec_combined = None
                    for key, value in combined.items():
                        if key not in self.globals and not key.startswith("_"):
                            self.locals[key] = value
                    self._restore_scaffold()
            except Exception:
                self._exec_combined = None
                result["error"] = traceback.format_exc()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Sliced join: the deadline charges only non-wait time. CRUCIAL — subtract BOTH the
        # completed wait (`_wait_secs`) AND any IN-PROGRESS wait (`_wait_active_since`).
        # Without the in-progress term, a single child-wait longer than `timeout` is killed
        # mid-wait (`_wait_secs` isn't updated until the wait's `finally`) — which orphaned
        # `subs` and produced uncited reports on slower domains (sqav2/HealthBench: 3/4 root
        # timeouts, all DURING rlm_query_batched).
        while t.is_alive():
            t.join(timeout=1.0)
            if not t.is_alive():
                break
            now = time.monotonic()
            in_progress = (now - self._wait_active_since) if self._wait_active_since else 0.0
            if (now - self._exec_start - self._wait_secs - in_progress) > self.timeout:
                return f"Timeout after {int(self.timeout)} seconds\n"
        return result["error"]

    @contextmanager
    def _pause_timeout(self):
        """Exempt a blocking sub-agent / sub-LLM wait from the cell-execution clock.

        SIGALRM path: disarm the alarm for the wait, then re-arm with the remaining (non-wait)
        budget. Thread path: mark the wait as in-progress (`_wait_active_since`) so the sliced
        join discounts it WHILE it is happening, then fold it into `_wait_secs` on exit. Nested
        waits (a batched call already paused, an inner one re-entering) keep the OUTER start so
        the whole span counts once."""
        use_alarm = self._alarm_armed and _can_use_sigalrm()
        if use_alarm:
            signal.alarm(0)
        t0 = time.monotonic()
        outer = self._wait_active_since is None
        if outer:
            self._wait_active_since = t0
        try:
            yield
        finally:
            if outer:
                self._wait_secs += time.monotonic() - self._wait_active_since
                self._wait_active_since = None
            if use_alarm:
                spent = time.monotonic() - self._exec_start - self._wait_secs
                signal.alarm(max(5, int(self.timeout - spent)))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_custom_tools(self):
        conflicts = set(self.custom_tools.keys()) & RESERVED_TOOL_NAMES
        if conflicts:
            raise ValueError(
                f"Custom tools cannot override reserved REPL names: {sorted(conflicts)}. "
                f"Reserved: {sorted(RESERVED_TOOL_NAMES)}"
            )


def _iter_tool_entries(custom_tools: Optional[Dict[str, Any]]):
    """Yield (name, value, description) for each custom tool.

    Supports two declaration formats:
    1. Plain:        {"name": callable_or_value}
    2. With desc:    {"name": {"tool": callable_or_value, "description": "..."}}
    """
    if not custom_tools:
        return
    for name, entry in custom_tools.items():
        if isinstance(entry, dict) and "tool" in entry:
            desc = entry.get("description")
            yield name, entry["tool"], desc if isinstance(desc, str) else None
        else:
            yield name, entry, None
