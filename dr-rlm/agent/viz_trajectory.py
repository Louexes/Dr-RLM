#!/usr/bin/env python
"""Render a captured DR-RLM trajectory JSONL into a single self-contained HTML file.

The trajectory JSONL is produced by `generate.py --log-dir ...` (rlm's RLMLogger). Its shape:

  line 0 : {"type": "metadata", root_model, max_depth, max_iterations, ...}
  line k : {"type": "iteration", iteration, response, code_blocks[], final_answer, iteration_time}
             code_blocks[i] = {code, result: {stdout, stderr, locals, execution_time, rlm_calls[], final_answer}}
             rlm_calls[j]   = {root_model, prompt, response, usage_summary, execution_time,
                               metadata?: {run_metadata, iterations[]}}   <-- a SUB-AGENT's own full trajectory

So one root file holds the ENTIRE recursion tree: this renderer walks `rlm_calls[].metadata.iterations`
recursively, nesting each sub-agent under the turn that spawned it.

Usage:
    python viz_trajectory.py TRAJ.jsonl [-o OUT.html] [--row deep_research_bench.jsonl]

`--row` is the matching generate.py output row file; if given, the problem text + the run's
aggregate counters (tokens, tool calls, recursion summary) are shown in the header.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_trajectory(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (metadata, iterations) from a trajectory JSONL."""
    metadata: Dict[str, Any] = {}
    iterations: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                metadata = obj
            elif obj.get("type") == "iteration":
                iterations.append(obj)
    return metadata, iterations


def find_row(row_path: Optional[str], example_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not row_path or not os.path.exists(row_path):
        return None
    rows = [json.loads(l) for l in open(row_path) if l.strip()]
    if example_id is not None:
        for r in rows:
            if str(r.get("example_id")) == str(example_id):
                return r
    return rows[0] if rows else None


def infer_example_id(traj_path: str) -> Optional[str]:
    # filename: traj_<bench>_<id>_<ts>_<uuid>.jsonl  -> pull the <id> (token before the date)
    base = os.path.basename(traj_path)
    parts = base.replace(".jsonl", "").split("_")
    for i, p in enumerate(parts):
        if i + 1 < len(parts) and parts[i + 1].count("-") == 2 and parts[i + 1][:4].isdigit():
            return p  # token right before the YYYY-MM-DD stamp
    return None


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def pre(text: Any, cls: str) -> str:
    return f'<pre class="{cls}">{esc(text)}</pre>'


def badge(label: str, value: Any, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(label)}: {esc(value)}</span>'


def _stdout_flags(stdout: str) -> List[str]:
    flags = []
    if stdout:
        if "TOOL BUDGET EXHAUSTED" in stdout:
            flags.append('<span class="badge warn">budget exhausted</span>')
        n_snip = stdout.count("<snippet id=")
        n_web = stdout.count("<webpage id=")
        if n_snip:
            flags.append(badge("snippets", n_snip, "io"))
        if n_web:
            flags.append(badge("webpages", n_web, "io"))
    return flags


def render_codeblock(cb: Dict[str, Any], depth: int) -> str:
    res = cb.get("result", {}) or {}
    code = cb.get("code", "")
    stdout = res.get("stdout", "") or ""
    stderr = res.get("stderr", "") or ""
    locals_ = res.get("locals", {}) or {}
    rlm_calls = res.get("rlm_calls", []) or []
    local_keys = [k for k in locals_.keys() if not k.startswith("_") and k not in ("answer",)]

    parts = ['<div class="cb">']
    parts.append('<div class="cb-label">▸ REPL code</div>')
    parts.append(pre(code, "code"))

    out_flags = _stdout_flags(stdout)
    flagstr = (" " + " ".join(out_flags)) if out_flags else ""
    if stdout.strip():
        parts.append(f'<div class="cb-label">stdout{flagstr} <span class="dim">({len(stdout)} chars)</span></div>')
        parts.append(pre(stdout, "stdout"))
    if stderr.strip():
        parts.append('<div class="cb-label err-label">stderr</div>')
        parts.append(pre(stderr, "stderr"))

    if local_keys:
        parts.append(
            '<details class="locals"><summary>locals after turn '
            f'({len(local_keys)} vars)</summary>'
        )
        rows = "".join(
            f'<tr><td class="k">{esc(k)}</td><td class="t">{esc(type_of(locals_[k]))}</td>'
            f'<td class="v">{esc(preview(locals_[k]))}</td></tr>'
            for k in local_keys
        )
        parts.append(f'<table class="loctab">{rows}</table></details>')

    # Nested sub-agents spawned by this code block
    for j, call in enumerate(rlm_calls):
        parts.append(render_subagent(call, depth + 1, j))

    parts.append("</div>")
    return "".join(parts)


def type_of(v: Any) -> str:
    return type(v).__name__


def preview(v: Any, n: int = 160) -> str:
    try:
        if isinstance(v, str):
            s = v
        else:
            s = json.dumps(v, default=str)
    except Exception:
        s = repr(v)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def render_subagent(call: Dict[str, Any], depth: int, idx: int) -> str:
    """A sub-agent call: show its prompt, its own nested trajectory (if captured), and its return."""
    prompt = call.get("prompt", "")
    response = call.get("response", "")
    meta = call.get("metadata") or {}
    child_iters = meta.get("iterations", []) if isinstance(meta, dict) else []
    model = call.get("root_model", "?")

    head = (
        f'<summary class="sa-sum">🜲 sub-agent #{idx} '
        f'<span class="dim">({esc(model)}, {len(child_iters)} turns)</span></summary>'
    )
    parts = [f'<details class="subagent d{min(depth,6)}" open>', head]
    if isinstance(prompt, str) and prompt.strip():
        parts.append('<div class="cb-label">prompt to sub-agent</div>')
        parts.append(pre(prompt, "prompt"))
    # the child's own turns, rendered recursively
    for it in child_iters:
        parts.append(render_iteration(it, depth))
    # the value returned to the parent
    if response:
        parts.append('<div class="cb-label">↩ returned to parent</div>')
        parts.append(pre(response, "answer"))
    parts.append("</details>")
    return "".join(parts)


def render_iteration(it: Dict[str, Any], depth: int) -> str:
    n = it.get("iteration", "?")
    response = it.get("response", "") or ""
    code_blocks = it.get("code_blocks", []) or []
    final_answer = it.get("final_answer")
    itime = it.get("iteration_time")

    badges = [badge("turn", n)]
    if itime is not None:
        badges.append(badge("t", f"{itime:.1f}s", "dim"))
    badges.append(badge("resp", f"{len(response)} ch", "dim"))
    n_sub = sum(len(cb.get("result", {}).get("rlm_calls", []) or []) for cb in code_blocks)
    if n_sub:
        badges.append(badge("sub-agents", n_sub, "io"))
    if final_answer:
        badges.append('<span class="badge final">FINAL ANSWER</span>')

    parts = [f'<details class="iter" open><summary>{" ".join(badges)}</summary>']
    if response.strip():
        parts.append('<div class="cb-label">model response</div>')
        parts.append(pre(response, "response"))
    for cb in code_blocks:
        parts.append(render_codeblock(cb, depth))
    if final_answer:
        cites = str(final_answer).count('<cite')
        cite_badge = (badge("cite tags", cites, "io" if cites else "warn"))
        parts.append(f'<div class="cb-label final-label">✅ final answer {cite_badge}</div>')
        parts.append(pre(final_answer, "final-answer"))
    parts.append("</details>")
    return "".join(parts)


CSS = """
:root{--bg:#0f1419;--card:#1a212b;--card2:#222b38;--ink:#d6deeb;--dim:#7a8aa0;
--code:#0b0f14;--blue:#82aaff;--green:#addb67;--red:#ff6b6b;--gold:#ffd479;--purple:#c792ea;--io:#5ccfe6;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b0f14;border-bottom:1px solid #2a3340;padding:14px 20px}
header h1{margin:0 0 6px;font-size:17px;color:var(--gold)}
.problem{color:var(--ink);background:#11161d;border-left:3px solid var(--purple);padding:8px 12px;margin:8px 0;border-radius:4px;white-space:pre-wrap}
.meta{color:var(--dim);font-size:12px}
.controls{margin-top:8px}
button{background:var(--card2);color:var(--ink);border:1px solid #34404f;border-radius:5px;padding:4px 10px;cursor:pointer;font-size:12px}
button:hover{background:#2c3848}
main{padding:18px 20px;max-width:1100px;margin:0 auto}
.badge{display:inline-block;background:#283242;color:var(--ink);border-radius:10px;padding:1px 8px;font-size:11px;margin-right:5px}
.badge.dim{background:transparent;color:var(--dim)}
.badge.io{background:#123;color:var(--io)}
.badge.warn{background:#3a2418;color:var(--gold)}
.badge.final{background:#2e2410;color:var(--gold)}
details{border-radius:7px;margin:8px 0}
details.iter{background:var(--card);border:1px solid #2a3340;padding:6px 12px}
summary{cursor:pointer;padding:4px 0;font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▾ ";color:var(--dim)}
details:not([open])>summary::before{content:"▸ "}
.cb{border-left:2px solid #34404f;margin:8px 0 8px 4px;padding-left:12px}
.cb-label{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 3px}
.err-label{color:var(--red)}
.final-label{color:var(--gold)}
pre{margin:0 0 4px;padding:10px 12px;border-radius:6px;overflow:auto;max-height:420px;
white-space:pre-wrap;word-break:break-word;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
pre.code{background:var(--code);color:var(--green);border:1px solid #1d2630}
pre.stdout{background:#10161d;color:#9fb3c8;border:1px solid #1d2630}
pre.stderr{background:#1f1416;color:var(--red);border:1px solid #3a1d1d}
pre.response{background:#121a24;color:var(--blue);border:1px solid #1d2630}
pre.prompt{background:#161226;color:var(--purple);border:1px solid #2a2140}
pre.answer{background:#141a12;color:var(--green);border:1px solid #243018}
pre.final-answer{background:#1c1708;color:var(--gold);border:1px solid #3a3014;max-height:none}
.subagent{background:var(--card2);border:1px solid #34404f;padding:6px 12px;margin-left:6px}
.subagent.d2{border-left:3px solid var(--purple)} .subagent.d3{border-left:3px solid var(--io)}
.subagent.d4{border-left:3px solid var(--green)} .subagent.d5{border-left:3px solid var(--gold)}
.sa-sum{color:var(--purple)}
.locals{background:#10161d;border:1px solid #1d2630;padding:4px 10px}
.locals summary{font-weight:400;color:var(--dim);font-size:12px}
.loctab{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace;margin-top:6px}
.loctab td{border-top:1px solid #1d2630;padding:3px 6px;vertical-align:top}
.loctab .k{color:var(--blue);white-space:nowrap} .loctab .t{color:var(--dim);white-space:nowrap}
.loctab .v{color:#9fb3c8}
.dim{color:var(--dim)}
"""

JS = """
function setAll(open){document.querySelectorAll('details').forEach(d=>d.open=open);}
"""


def render_html(metadata: Dict[str, Any], iterations: List[Dict[str, Any]],
                row: Optional[Dict[str, Any]], title: str) -> str:
    # header stats
    model = metadata.get("root_model", "?")
    hdr_meta = [
        badge("model", model),
        badge("max_depth", metadata.get("max_depth", "?")),
        badge("max_iters", metadata.get("max_iterations", "?")),
        badge("turns", len(iterations)),
    ]
    problem_html = ""
    if row:
        prob = row.get("problem") or (row.get("original_data") or {}).get("problem")
        if prob:
            problem_html = f'<div class="problem">{esc(prob)}</div>'
        aod = row.get("additional_output_data", {}) or {}
        ft = row.get("full_traces", {}) or {}
        rec = aod.get("recursion", {}) or {}
        for label, val in [
            ("total_tokens", ft.get("total_tokens")),
            ("prompt_tok", ft.get("prompt_tokens")),
            ("compl_tok", ft.get("completion_tokens")),
            ("search", aod.get("n_search")),
            ("browse", aod.get("n_browse")),
            ("subagents", rec.get("n_subagents")),
            ("depth", f"{rec.get('max_depth_reached')}/{rec.get('max_depth_cap')}"),
            ("budget_blocked", aod.get("budget_blocked_calls")),
        ]:
            if val is not None:
                hdr_meta.append(badge(label, val, "dim"))

    body = "".join(render_iteration(it, depth=1) for it in iterations)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<header>
  <h1>DR-RLM trajectory — {esc(title)}</h1>
  {problem_html}
  <div class="meta">{" ".join(hdr_meta)}</div>
  <div class="controls">
    <button onclick="setAll(true)">expand all</button>
    <button onclick="setAll(false)">collapse all</button>
  </div>
</header>
<main>{body}</main>
<script>{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", help="path to a traj_*.jsonl from generate.py --log-dir")
    ap.add_argument("-o", "--output", default=None, help="output .html (default: alongside input)")
    ap.add_argument("--row", default=None, help="matching generate.py output row JSONL (for problem + counters)")
    args = ap.parse_args()

    metadata, iterations = load_trajectory(args.trajectory)
    if not iterations:
        sys.exit(f"No iterations found in {args.trajectory}")

    example_id = infer_example_id(args.trajectory)
    row = find_row(args.row, example_id)
    title = os.path.basename(args.trajectory).replace(".jsonl", "")

    out = args.output or args.trajectory.replace(".jsonl", ".html")
    if out == args.trajectory:
        out = args.trajectory + ".html"
    with open(out, "w") as f:
        f.write(render_html(metadata, iterations, row, title))
    print(f"wrote {out}  ({len(iterations)} turns, model={metadata.get('root_model')})")


if __name__ == "__main__":
    main()
