"""Tests for pre-submit citation verification (answer_format.verify_citations).

Targets the biggest measured harness gap: citations are made reliably but loosely
(DRB-FACT valid_rate 0.345 on hard items; sqav2 citation alignment ~0). Verification
batch-judges each (claim, ledger-snippet) pair via the LM and drops explicit
UNSUPPORTED verdicts before rendering. Invariants under test:

* only explicit UNSUPPORTED drops a citation (parse noise keeps it — judge noise
  must not eat valid citations);
* pairs with no claim text or no ledger snippet are kept untouched (and not judged);
* one BATCHED lm_callback call, never one per citation;
* any callback failure returns the object unchanged (never breaks a rollout);
* the env honors the `citation_verification` flag (off => no LM call).
"""

import threading

import examples.train.dr_rlm  # noqa: F401
from examples.train.dr_rlm.answer_format import verify_citations
from examples.train.dr_rlm.dr_rlm_env import DrRlmEnv


def _ledger(**snips):
    return {"snippets": {k: {"node_rid": "n", "text": v, "url": ""} for k, v in snips.items()},
            "_lock": threading.Lock()}


_OBJ = {
    "content": "Paris is the capital. The moon is cheese.",
    "citations": [
        {"id": "a-0", "claim": "Paris is the capital."},
        {"id": "a-1", "claim": "The moon is cheese."},
        {"id": "a-2", "claim": ""},          # no claim -> not judged, kept
        {"id": "missing", "claim": "x y z"}, # no snippet -> not judged, kept
    ],
}
_LED = _ledger(**{"a-0": "Paris has been France's capital for centuries.",
                  "a-1": "The moon is made of rock and dust.",
                  "a-2": "irrelevant"})


def test_drops_only_explicit_unsupported():
    calls = []

    def lm(prompts):
        calls.append(list(prompts))
        out = []
        for p in prompts:
            out.append("UNSUPPORTED" if "cheese" in p else "SUPPORTED")
        return out

    obj, stats = verify_citations(_OBJ, _LED, lm)
    kept_ids = [c["id"] for c in obj["citations"]]
    assert kept_ids == ["a-0", "a-2", "missing"]      # cheese-claim dropped, unjudgeable kept
    assert stats == {"checked": 2, "dropped": 1}
    assert len(calls) == 1 and len(calls[0]) == 2     # ONE batched call, only judgeable pairs


def test_parse_noise_keeps_citation():
    obj, stats = verify_citations(_OBJ, _LED, lambda ps: ["hmm not sure"] * len(ps))
    assert len(obj["citations"]) == 4 and stats["dropped"] == 0


def test_callback_failure_is_harmless():
    def boom(ps):
        raise RuntimeError("judge down")
    obj, stats = verify_citations(_OBJ, _LED, boom)
    assert obj is _OBJ and stats == {"checked": 0, "dropped": 0}


def test_no_ledger_or_no_citations_is_noop():
    assert verify_citations({"content": "x", "citations": []}, _LED, lambda ps: [])[0]["citations"] == []
    obj, _ = verify_citations(_OBJ, None, lambda ps: [])
    assert obj is _OBJ


def test_env_flag_gates_verification():
    def lm(prompts):
        raise AssertionError("lm_callback must not be called when flag is off")

    e = DrRlmEnv(extras={"depth": 0, "lm_callback": lm, "dr_rlm_ledger": _LED,
                         "dr_rlm": {"max_recursion_depth": 0, "citation_verification": False}})
    e._finalize_answer("text", {"content": "Paris is the capital.",
                                "citations": [{"id": "a-0", "claim": "Paris is the capital."}]})
    assert e._citation_verify_stats is None

    hits = []
    e2 = DrRlmEnv(extras={"depth": 0, "lm_callback": lambda ps: (hits.append(len(ps)) or ["SUPPORTED"] * len(ps)),
                          "dr_rlm_ledger": _LED,
                          "dr_rlm": {"max_recursion_depth": 0, "citation_verification": True}})
    e2._finalize_answer("text", {"content": "Paris is the capital.",
                                 "citations": [{"id": "a-0", "claim": "Paris is the capital."}]})
    assert hits == [1] and e2._citation_verify_stats == {"checked": 1, "dropped": 0}
