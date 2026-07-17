"""Run-safety guards for DR-RLM generation, shared by the env-side tool facade
(``tools/provider.make_tools``) and the inference driver (``agent/infer_driver``).

Restores the protection the unified ``--driver new`` path had dropped relative to the legacy
``generate.py``: detect a HARD (non-transient) search failure — Serper out-of-credits / auth /
quota — so the run ABORTS immediately instead of silently emitting ungrounded reports and
burning more API calls. The facade used to swallow every search exception to ``[]``, so the
credit signal (the 2026-05-31 "Serper out of credits" incident) never reached the driver loop.

Now: ``make_tools.search`` calls ``note_exception`` on failure (latching a run-global flag and
short-circuiting further Serper calls for the rest of the tree), and the driver checks
``hard_error_latched()`` after each item and exits. Process-global is correct — each driver
run is a fresh process.
"""

from __future__ import annotations

import threading

# Lowercased substrings that mark a NON-transient search failure (dead/empty key, auth, quota,
# payment). Mirrors legacy generate.py's hard-marker list, plus rate-limit phrasings. Erring
# toward a (cheap, safe) false abort over burning credits is intentional here.
HARD_MARKERS = (
    "not enough credits", "insufficient credit", "out of credits",
    "status 400", "status 401", "status 402", "status 403", "status 429",
    "http 401", "http 402", "http 403", "http 429",
    "error 402", "error 429",
    "unauthorized", "forbidden", "invalid api key", "api key",
    "quota", "payment required", "payment", "too many requests", "rate limit",
    # Jina (page-reader) balance exhaustion: its message is "InsufficientBalanceError:
    # Account balance not enough ... please recharge (uid: ...)" — matched none of the
    # above, so a Jina-out run limped through every item producing collapsed reports
    # instead of aborting at item 1. Latch on its specific wording.
    "insufficientbalanceerror", "balance not enough", "please recharge", "account balance",
)

_STATE = {"hit": False, "msg": None}
_LOCK = threading.Lock()


def note_exception(exc) -> bool:
    """If ``exc`` looks like a HARD search failure, latch the run-global flag and return True.
    Transient errors return False (the caller still returns ``[]`` for that one call)."""
    msg = str(exc).lower()
    if any(k in msg for k in HARD_MARKERS):
        with _LOCK:
            if not _STATE["hit"]:
                _STATE["hit"] = True
                _STATE["msg"] = str(exc)[:300]
        return True
    return False


def hard_error_latched() -> bool:
    with _LOCK:
        return _STATE["hit"]


def get_hard_error() -> dict:
    """Snapshot ``{hit, msg}`` for the driver's abort message."""
    with _LOCK:
        return dict(_STATE)


def reset() -> None:
    """Clear the flag (tests / explicit re-use within one process)."""
    with _LOCK:
        _STATE["hit"] = False
        _STATE["msg"] = None
