"""Offline checks for the flat DR-Tulu env (dr_tulu_env.py).

Backend + judge are monkeypatched — these verify the byte-parity formats and the
verbatim reward-term formulas against the values pinned in
docs/DRTULU_RL_FAITHFULNESS_AUDIT.md, not model behavior.
"""

import pytest

from examples.train.dr_rlm import dr_tulu_env as E


class _FakeBackend:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, k):
        self.queries.append((query, k))
        return self.hits


HITS = [
    {"docid": "d1", "snippet": "Alpha result. More text here.", "url": "http://a", "score": 1.0},
    {"docid": "d2", "snippet": "Beta result. Details.", "url": "", "score": 0.5},
]

EXTRAS = {
    "reward_spec": {"query": "what is alpha?", "rubrics": [{"description": "mentions alpha", "title": "a", "weight": 1.0}]},
    "question_type": "long_form",
    "max_turns": 14,
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    prompt_file = tmp_path / "p.yaml"
    prompt_file.write_text(
        'system_prompt: "SYS"\nadditional_instructions:\n  long_form: "LONGFORM INSTR"\n  short_form: "SHORT"\n'
    )
    monkeypatch.setenv("DR_TULU_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("DR_TULU_MAX_TOOL_CALLS", "2")
    e = E.DrTuluEnv(env_config={}, extras=dict(EXTRAS))
    fake = _FakeBackend(HITS)
    monkeypatch.setattr(E, "get_backend", lambda payload: fake)
    e._fake = fake
    return e


CALL = '<call_tool name="snippet_search">alpha</call_tool>'


def test_prompt_composition(env):
    out, _ = env.init([{"role": "user", "content": "Q?"}])
    assert out[0] == {"role": "system", "content": "SYS"}
    # additional_instructions go in the USER turn, "\n\n"-separated (auto_search_sft.py:104-184)
    assert out[1] == {"role": "user", "content": "Q?\n\nLONGFORM INSTR"}


def test_tool_output_byte_format(env, monkeypatch):
    monkeypatch.setattr(E.uuid, "uuid4", lambda: type("U", (), {"__str__": lambda s: "a1b2c3d4-xxxx"})())
    out = env.step(CALL)
    obs = out["observations"][0]["content"]
    assert obs == (
        "<tool_output><snippet id=a1b2c3d4-0>\n"
        "Title: Alpha result\nURL: http://a\nSnippet: Alpha result. More text here.\n</snippet>\n"
        "<snippet id=a1b2c3d4-1>\nTitle: Beta result\nSnippet: Beta result. Details.\n</snippet></tool_output>"
    )
    assert not out["done"]
    assert env.snippets["a1b2c3d4-0"].startswith("Title: Alpha result")
    assert env._fake.queries == [("alpha", 10)]


def test_unknown_tool_ends_episode(env, monkeypatch):
    # client.py breaks when no registered tool matches -> done + scored
    monkeypatch.setattr(E, "score_report_sync", lambda *a: (0.0, {}))
    monkeypatch.setattr(E, "score_in_context_citations_sync", lambda *a, **k: 0.0)
    out = env.step('<call_tool name="google_search">x</call_tool>')
    assert out["done"]


def test_budget_message_and_continue(env):
    env.step(CALL)
    env.step(CALL)  # budget (2) used up
    out = env.step(CALL)
    assert out["observations"][0]["content"] == "<tool_output>exceed allowed tool call requests</tool_output>"
    assert not out["done"]  # model may keep generating (client.py:758-777)


def test_composite_reward_no_answer(env, monkeypatch):
    monkeypatch.setattr(E, "score_report_sync", lambda *a: (pytest.fail("judge must not run"), {}))
    env.step(CALL)
    out = env.step("I ran out of ideas.")  # no call, no answer -> done
    # rubric=cite=0; format = 0.2 (has >=1 call in transcript); search = 0 (no <answer> => context None)
    assert out["done"]
    assert out["reward"] == pytest.approx(0.2 * 0.2)


def test_composite_reward_full(env, monkeypatch):
    monkeypatch.setattr(E, "score_report_sync", lambda report, q, rubrics, cfg: (0.8, {"a": 0.8}))
    monkeypatch.setattr(E, "score_in_context_citations_sync", lambda q, full, snips, cfg: 0.5)
    env.step(CALL)
    env.step(CALL)
    out = env.step('<answer>Alpha is <cite id="a1b2c3d4-0">a thing.</cite></answer>')
    # format: answer(0.5) + cite(0.3) + query(0.2) = 1.0 ; search: min(2/3,1)
    expected = 0.5 * 0.8 + 0.2 * 0.5 + 0.2 * 1.0 + 0.1 * (2 / 3)
    assert out["done"]
    assert out["reward"] == pytest.approx(expected)
    assert out["metadata"]["components"]["finalized"] == 1.0


def test_rubric_judge_gets_answer_inner_text(env, monkeypatch):
    seen = {}

    def fake_rubric(report, q, rubrics, cfg):
        seen["report"] = report
        return 1.0, {}

    monkeypatch.setattr(E, "score_report_sync", fake_rubric)
    monkeypatch.setattr(E, "score_in_context_citations_sync", lambda *a, **k: 0.0)
    env.step('<answer> The answer with <cite id="x-0">cites kept</cite>. </answer>')
    # <answer> tags stripped, .strip()ed, <cite> KEPT (format_utils.py:53-64)
    assert seen["report"] == 'The answer with <cite id="x-0">cites kept</cite>.'


def test_format_reward_verbatim():
    assert E.compute_format_reward("nothing") == 0.0
    assert E.compute_format_reward("<answer>x</answer>") == pytest.approx(0.5)
    assert E.compute_format_reward('<cite id=s1>claim</cite>') == pytest.approx(0.3)
    assert E.compute_format_reward('<call_tool name="snippet_search">q</call_tool>') == pytest.approx(0.2)


def test_search_turns_verbatim():
    ctx = '<call_tool name="snippet_search">a</call_tool>' * 5
    assert E.compute_search_turns_reward(ctx) == 1.0  # min(5/3,1)
    assert E.compute_search_turns_reward(None) == 0.0
    assert E.compute_search_turns_reward('<call_tool name="s">  </call_tool>') == 0.0  # empty query invalid


def test_empty_and_no_result_errors(env, monkeypatch):
    out = env.step('<call_tool name="snippet_search">   </call_tool>')
    assert out["observations"][0]["content"] == "<tool_output>No valid query found in tool call.</tool_output>"
    env._fake.hits = []
    monkeypatch.setattr(E, "get_backend", lambda payload: env._fake)
    out = env.step(CALL)
    assert out["observations"][0]["content"] == "<tool_output>No results found for the query.</tool_output>"


def test_max_turns_scores(env, monkeypatch):
    monkeypatch.setattr(E, "score_report_sync", lambda *a: (0.0, {}))
    monkeypatch.setattr(E, "score_in_context_citations_sync", lambda *a, **k: 0.0)
    env.max_turns = 2
    out = env.step(CALL)
    assert not out["done"]
    out = env.step(CALL)
    assert out["done"]  # turn budget exhausted -> scored termination, not a dropped rollout
