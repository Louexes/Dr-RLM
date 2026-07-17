#!/usr/bin/env python
"""Render a DR-RLM (unified driver) trajectory JSONL into a self-contained HTML file.

Input schema (produced by `infer_driver.py --save-trajectories DIR`, one file per item):
  line 0    : {type:"metadata", example_id, benchmark, problem, config:{...}, n_nodes}
  line 1..N : {type:"node", rid, parent_rid, depth, child_index, n_turns, cited_ids,
               final_answer, trajectory:[{turn, response, code, stdout, stderr, submitted}]}

The recursion tree is rebuilt from parent_rid and rendered as nested collapsible cards, each
node showing its per-turn reasoning / repl code / stdout. The run `config` is shown up top so a
trajectory is self-describing: you can see exactly which design choices produced this behavior.

Usage:
    python viz_trajectory_v2.py TRAJ.jsonl [-o OUT.html]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load(path: str):
    meta: Dict[str, Any] = {}
    nodes: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            (meta.update(o) if o.get("type") == "metadata" else nodes.append(o))
    return meta, nodes


def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _tree(nodes: List[Dict[str, Any]]):
    """parent_rid -> [children]; roots = nodes whose parent is absent from the set."""
    by_parent: Dict[Optional[str], List[Dict[str, Any]]] = {}
    rids = {n.get("rid") for n in nodes}
    for n in nodes:
        p = n.get("parent_rid")
        key = p if p in rids else None
        by_parent.setdefault(key, []).append(n)
    return by_parent


def render_turn(t: Dict[str, Any]) -> str:
    parts = [f'<div class="turn"><div class="th">turn {esc(t.get("turn"))}'
             + (' · <b>SUBMITTED</b>' if t.get("submitted") else '') + '</div>']
    if t.get("response"):
        parts.append(f'<div class="lbl">model</div><pre class="resp">{esc(t["response"])}</pre>')
    if t.get("code"):
        parts.append(f'<div class="lbl">repl code</div><pre class="code">{esc(t["code"])}</pre>')
    if t.get("stdout"):
        parts.append(f'<div class="lbl">stdout</div><pre class="out">{esc(t["stdout"])}</pre>')
    if t.get("stderr"):
        parts.append(f'<div class="lbl err">stderr</div><pre class="err">{esc(t["stderr"])}</pre>')
    parts.append('</div>')
    return "".join(parts)


def render_node(n: Dict[str, Any], by_parent) -> str:
    depth = n.get("depth", 0)
    label = "ROOT" if depth == 0 else f"depth {depth}"
    cited = ", ".join(n.get("cited_ids") or []) or "—"
    head = (f'<summary><span class="badge d{min(depth,3)}">{esc(label)}</span> '
            f'rid={esc(n.get("rid"))} · turns={esc(n.get("n_turns"))} · cited={esc(len(n.get("cited_ids") or []))}</summary>')
    turns = "".join(render_turn(t) for t in (n.get("trajectory") or []))
    fa = n.get("final_answer") or ""
    fa_html = f'<div class="lbl">final_answer ({len(fa)} chars)</div><pre class="fa">{esc(fa[:4000])}</pre>' if fa else ""
    cited_html = f'<div class="lbl">cited_ids</div><div class="cited">{esc(cited)}</div>'
    kids = "".join(render_node(c, by_parent) for c in by_parent.get(n.get("rid"), []))
    kids_html = f'<div class="kids">{kids}</div>' if kids else ""
    return f'<details open class="node"><div class="nodebody">{head}{turns}{cited_html}{fa_html}</div>{kids_html}</details>'


_CSS = """
body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
h1{font-size:18px} .cfg{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin:10px 0}
.cfg code{color:#79c0ff} .prob{background:#161b22;border-left:3px solid #1f6feb;padding:8px 12px;margin:8px 0;white-space:pre-wrap}
.node{border:1px solid #30363d;border-radius:8px;margin:8px 0;background:#161b22}
.nodebody{padding:8px 10px} .kids{margin-left:22px;border-left:2px solid #21262d;padding-left:8px}
summary{cursor:pointer;font-weight:600;margin:-8px -10px 6px;padding:8px 10px}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;color:#fff;font-size:11px}
.badge.d0{background:#1f6feb}.badge.d1{background:#238636}.badge.d2{background:#9e6a03}.badge.d3{background:#8957e5}
.turn{border-top:1px dashed #30363d;padding:6px 0} .th{color:#8b949e;font-size:11px;margin-bottom:3px}
.lbl{color:#8b949e;font-size:10px;text-transform:uppercase;margin:5px 0 1px} .lbl.err{color:#f85149}
pre{margin:2px 0;padding:7px 9px;border-radius:6px;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto}
.resp{background:#0d1117;border:1px solid #30363d} .code{background:#0b1f0b;border:1px solid #196c2e;color:#7ee787}
.out{background:#11151c;border:1px solid #30363d;color:#adbac7} .err{background:#2d1416;color:#ff7b72}
.fa{background:#0d1117;border:1px solid #1f6feb} .cited{font-family:monospace;color:#79c0ff;font-size:11px}
"""


def render(meta: Dict[str, Any], nodes: List[Dict[str, Any]]) -> str:
    by_parent = _tree(nodes)
    cfg = meta.get("config") or {}
    cfg_html = " · ".join(f"<code>{esc(k)}={esc(v)}</code>" for k, v in cfg.items())
    roots = by_parent.get(None, [])
    body = "".join(render_node(r, by_parent) for r in roots)
    return (f'<!doctype html><meta charset="utf-8"><style>{_CSS}</style><div class="wrap">'
            f'<h1>Trajectory · {esc(meta.get("benchmark"))} · id={esc(meta.get("example_id"))} '
            f'· {esc(meta.get("n_nodes"))} nodes</h1>'
            f'<div class="cfg">{cfg_html}</div>'
            f'<div class="lbl">research question</div><div class="prob">{esc(meta.get("problem"))}</div>'
            f'{body}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    meta, nodes = load(a.traj)
    out = a.out or a.traj.rsplit(".", 1)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(meta, nodes))
    print(f"wrote {out}  ({len(nodes)} nodes)")


if __name__ == "__main__":
    main()
