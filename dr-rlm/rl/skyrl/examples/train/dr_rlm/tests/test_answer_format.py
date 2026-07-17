"""Tests for the structured-answer contract (answer_format.py): normalization with the
evidence-ledger backstop, and claim-aware inline rendering (the DRB-FACT-friendly path).

All pure/sync — no judge, no HTTP, no GPU.
"""

import re

from examples.train.dr_rlm.answer_format import normalize_answer, render_report, cited_ids_of
from examples.train.dr_rlm.judge import all_cited_ids

# The EXACT regex dr-tulu's DeepResearchBench submission formatter uses to extract citations
# (deep_research_bench_eval/drb_formatter.py: parse_and_format_citations). The "fact" FACT
# verifies is group 2 (text INSIDE the tag). Locking it here so our render can't drift out of
# DRB-FACT compatibility (verified live against drb_formatter.data_format on 2026-05-31).
_DRB_CITE = re.compile(r'<cite id="([^"]+)">([^<]+)</cite>')


LEDGER = {
    "snippets": {
        "child-0": {"node_rid": "child", "text": "X is true per source", "url": "u0"},
        "root-1": {"node_rid": "root", "text": "Y holds", "url": "u1"},
    }
}


# ---- normalize_answer -----------------------------------------------------

def test_normalize_keeps_only_ledgered_dict_cites():
    obj = {"content": "The finding holds.", "citations": [
        {"id": "child-0", "claim": "The finding holds."},
        {"id": "missing-9", "claim": "bogus"},  # not in ledger -> dropped (backstop)
    ]}
    out = normalize_answer(obj, "", "root", LEDGER)
    assert out["content"] == "The finding holds."
    assert out["citations"] == [{"id": "child-0", "claim": "The finding holds."}]
    assert cited_ids_of(out) == ["child-0"]


def test_normalize_accepts_bare_string_ids():
    out = normalize_answer({"content": "c", "citations": ["root-1", "missing"]}, "", "root", LEDGER)
    assert out["citations"] == [{"id": "root-1", "claim": ""}]


def test_normalize_legacy_scrapes_inline_cites_when_no_obj():
    # final_obj is None (no structured citations): recover id+claim from prose <cite> tags
    content = 'A claim <cite id="child-0">the finding</cite>. Uncited bit.'
    out = normalize_answer(None, content, "root", LEDGER)
    assert {"id": "child-0", "claim": "the finding"} in out["citations"]
    assert cited_ids_of(out) == ["child-0"]


def test_normalize_falls_back_to_scrape_when_citations_empty():
    # structured obj but the model forgot the list, yet inlined a cite -> still recovered
    obj = {"content": 'It holds <cite id="root-1">Y holds</cite>.', "citations": []}
    out = normalize_answer(obj, "", "root", LEDGER)
    assert cited_ids_of(out) == ["root-1"]


def test_normalize_no_ledger_accepts_ids():
    out = normalize_answer({"content": "c", "citations": [{"id": "anything-3", "claim": "z"}]},
                           "", "root", None)  # no ledger -> backstop disabled
    assert cited_ids_of(out) == ["anything-3"]


def test_normalize_dedups_id_claim_pairs_but_keeps_distinct_claims():
    obj = {"content": "c", "citations": [
        {"id": "child-0", "claim": "a"}, {"id": "child-0", "claim": "a"},  # exact dup
        {"id": "child-0", "claim": "b"},                                   # same id, new claim
    ]}
    out = normalize_answer(obj, "", "root", LEDGER)
    assert len(out["citations"]) == 2
    assert cited_ids_of(out) == ["child-0"]  # ids collapsed


def test_normalize_tolerates_junk_without_crashing():
    obj = {"content": "c", "citations": [None, 7, {"no_id": 1}, {"id": "child-0"}]}
    out = normalize_answer(obj, "", "root", LEDGER)
    assert out["citations"] == [{"id": "child-0", "claim": ""}]


# ---- render_report --------------------------------------------------------

def test_render_inline_attaches_citation_to_its_claim():
    obj = {"content": "The finding holds. Other text.",
           "citations": [{"id": "child-0", "claim": "The finding holds."}]}
    rep = render_report(obj, LEDGER, mode="inline")
    assert '<cite id="child-0">The finding holds.</cite>' in rep
    assert "child-0" in all_cited_ids(rep)
    assert "References:" not in rep  # claim located -> no fallback


def test_render_does_not_duplicate_already_inline_cite():
    obj = {"content": 'It holds <cite id="child-0">the finding</cite>.',
           "citations": [{"id": "child-0", "claim": "the finding"}]}
    rep = render_report(obj, LEDGER, mode="inline")
    assert rep.count("child-0") == 1


def test_render_unlocatable_claim_falls_back_to_references():
    obj = {"content": "Totally different prose.",
           "citations": [{"id": "child-0", "claim": "claim not present in content"}]}
    rep = render_report(obj, LEDGER, mode="inline")
    assert "References:" in rep
    assert "child-0" in all_cited_ids(rep)  # id still present + parseable


def test_render_appended_mode_lists_everything():
    obj = {"content": "Prose with no inline cites.",
           "citations": [{"id": "child-0", "claim": "The finding holds."}]}
    rep = render_report(obj, LEDGER, mode="appended")
    assert "References:" in rep
    assert "child-0" in all_cited_ids(rep)


def test_render_no_citations_is_identity():
    assert render_report({"content": "just text", "citations": []}, LEDGER) == "just text"


# ---- the orchestrator-merge / FACT-recovery scenario ----------------------

def test_merged_child_cite_becomes_visible_in_graded_report():
    """Root paraphrased the child uncited in prose, but merged the child's structured
    citation. After normalize+render the child's evidence id is present in the graded
    report (FACT can see it) AND in cited_ids (credit), with the bogus id stripped."""
    obj = {"content": "Synthesis: the finding holds across sources.",
           "citations": [
               {"id": "child-0", "claim": "the finding holds across sources"},
               {"id": "fabricated-2", "claim": "never retrieved"},  # backstop drops this
           ]}
    out = normalize_answer(obj, "", "root", LEDGER)
    rep = render_report(out, LEDGER, mode="inline")
    assert "child-0" in all_cited_ids(rep)      # FACT-visible (claim matched -> inlined)
    assert cited_ids_of(out) == ["child-0"]     # credit channel, fabricated id dropped
    assert "fabricated-2" not in rep


# ---- DRB-FACT format contract (self-contained: the extractor regex, not the external repo) --

def test_render_inline_matches_drb_fact_extractor():
    # inline path: DRB extracts (id, claim) — the claim is the "fact" it support-checks
    obj = {"content": "Solar costs dropped sharply over the decade.",
           "citations": [{"id": "n1-0", "claim": "Solar costs dropped sharply over the decade"}]}
    rep = render_report(normalize_answer(obj, "", "root", LEDGER_DRB), LEDGER_DRB, mode="inline")
    assert ("n1-0", "Solar costs dropped sharply over the decade") in _DRB_CITE.findall(rep)


def test_render_fallback_puts_claim_in_tag_not_source_quote():
    # unlocatable claim -> References fallback, but the tag's inner text must be the CLAIM
    # (the fact DRB verifies), NOT the ledger snippet (which would make support circular)
    obj = {"content": "Unrelated prose.",
           "citations": [{"id": "n1-0", "claim": "wind generation rose"}]}
    rep = render_report(normalize_answer(obj, "", "root", LEDGER_DRB), LEDGER_DRB, mode="inline")
    pairs = dict(_DRB_CITE.findall(rep))
    assert pairs.get("n1-0") == "wind generation rose"  # claim in tag, not the snippet text


LEDGER_DRB = {"snippets": {"n1-0": {"node_rid": "n1", "text": "IRENA: PV LCOE -89%", "url": "http://x"}}}


def test_normalize_quotes_unquoted_inline_cites():
    # the A2 failure reproduced: model emits UNQUOTED <cite id=X>; normalize must rewrite to
    # id="X" so the DRB FACT extractor (which requires double quotes) can parse it.
    out = normalize_answer({"content": 'Solar fell <cite id=n1-0>sharply</cite>.', "ready": True},
                           "", "root", LEDGER_DRB)
    assert '<cite id="n1-0">sharply</cite>' in out["content"]
    assert "<cite id=n1-0>" not in out["content"]
    assert cited_ids_of(out) == ["n1-0"]
    assert ("n1-0", "sharply") in _DRB_CITE.findall(render_report(out, LEDGER_DRB, mode="inline"))


def test_normalize_quotes_singlequoted_inline_cites():
    out = normalize_answer({"content": "x <cite id='n1-0'>c</cite>.", "ready": True}, "", "r", LEDGER_DRB)
    assert '<cite id="n1-0">c</cite>' in out["content"]
