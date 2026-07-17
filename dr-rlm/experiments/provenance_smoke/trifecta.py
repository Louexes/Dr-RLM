#!/usr/bin/env python3
"""Trifecta readout for a provenance-credit RL run.

The open question across all RL runs: which reward assignment gives
  rising R  +  rising cites/kid  +  falling orphan%   ALL TOGETHER.

Reads a credit_metrics jsonl (one row per tree, append-order = training order),
bins early->late, and reports the three signals per bin so the trend is visible.

Judge-INDEPENDENT signals (immune to Gemini outage — the a2 contamination lesson):
  cites/kid = mean n_cited over child nodes        (behavioral: do sub-agents cite?)
  orphan%   = n_orphan_children / n_children        (behavioral: do child cites survive?)
Judge-DEPENDENT (must filter R==0 outages):
  R(alive)  = mean report_reward among trees with report_reward>0
  R0%       = fraction report_reward==0 (empties OR judge outage — read with the guard)

Usage: python trifecta.py <credit_metrics.jsonl> [--bins 6]
"""
import json, sys, statistics as st


def _children(row):
    return [n for n in (row.get("nodes") or []) if (n.get("depth") or 0) >= 1]


def _root(row):
    for n in (row.get("nodes") or []):
        if (n.get("depth") or 0) == 0:
            return n
    return None


def tree_metrics(row):
    """Return per-tree metrics or None if unusable.

    root_final (judge-INDEPENDENT): root node emitted a non-empty report. This is the
    finalization signal — the ~60% empty-root failures (DRB deficit) live here, and it's
    the cleanest read separate from R0% (which conflates empties + judge outage)."""
    rer = row.get("rer_metrics") or {}
    R = row.get("report_reward", rer.get("report_reward"))
    kids = _children(row)
    nk = len(kids) or rer.get("n_nodes")
    if not nk:
        return None
    cites_per_kid = st.mean([k.get("n_cited", 0) for k in kids]) if kids else None
    n_orph = rer.get("n_orphan_children")
    orphan_frac = (n_orph / nk) if (n_orph is not None and nk) else None
    root = _root(row)
    root_final = (root.get("ans_len", 0) > 0) if root else None
    return dict(R=R, alive=(isinstance(R, (int, float)) and R > 0),
                cpk=cites_per_kid, orph=orphan_frac, rootf=root_final)


def _agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return st.mean(vals) if vals else None


def report(path, nbins=6):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tm = [m for m in (tree_metrics(r) for r in rows) if m]
    if not tm:
        print("no usable trees yet"); return
    n = len(tm); sz = max(1, n // nbins)
    # last bin absorbs the remainder so no tiny tail bin skews the early->late arrow
    bins = [tm[i * sz:(i + 1) * sz] for i in range(nbins - 1)] + [tm[(nbins - 1) * sz:]]
    bins = [b for b in bins if b]
    def f(x): return "  -  " if x is None else f"{x:5.3f}"
    print(f"trees={n}  bins={len(bins)} (~{sz}/bin)   [want: root-final UP, R(alive) UP, cites/kid UP, orphan% DOWN]")
    print(f"{'bin':>4} {'trees':>5} {'rootF%':>6} {'R0%':>5} {'R(alive)':>8} {'cites/kid':>9} {'orphan%':>8}")
    first = last = None
    for i, b in enumerate(bins):
        r0 = sum(1 for m in b if not m["alive"]) / len(b)
        rf = _agg([1.0 if m["rootf"] else 0.0 for m in b if m["rootf"] is not None])
        Ral = _agg([m["R"] for m in b if m["alive"]])
        cpk = _agg([m["cpk"] for m in b])
        orp = _agg([m["orph"] for m in b])
        row = dict(R=Ral, cpk=cpk, orp=orp, rf=rf)
        if i == 0: first = row
        last = row
        rfs = "  -  " if rf is None else f"{100*rf:4.0f}%"
        print(f"{i:>4} {len(b):>5} {rfs:>6} {100*r0:4.0f}% {f(Ral):>8} {f(cpk):>9} {f(orp):>8}")
    def trend(a, b):
        if a is None or b is None: return "n/a"
        d = b - a
        return f"{a:.3f}->{b:.3f} ({'+' if d>=0 else ''}{d:.3f})"
    print("\nEARLY->LATE:")
    print(f"  root-final%: {trend(first['rf'], last['rf'])}   want UP (finalization bottleneck)")
    print(f"  R(alive):  {trend(first['R'],  last['R'])}   want UP")
    print(f"  cites/kid: {trend(first['cpk'], last['cpk'])}   want UP")
    print(f"  orphan%:   {trend(first['orp'], last['orp'])}   want DOWN")
    up = lambda a, b: (a is not None and b is not None and b >= a)
    dn = lambda a, b: (a is not None and b is not None and b <= a)
    trifecta = up(first['R'], last['R']) and up(first['cpk'], last['cpk']) and dn(first['orp'], last['orp'])
    print(f"\n  TRIFECTA (all three right direction): {'YES ✓' if trifecta else 'no — not all three moved together'}")


def _demo():
    # synthetic: early = orphaned + low R, late = citing + higher R -> trifecta YES
    def tree(R, ncit, norph, nk=4):
        return {"report_reward": R, "rer_metrics": {"n_orphan_children": norph, "n_nodes": nk},
                "nodes": [{"depth": 1, "n_cited": ncit} for _ in range(nk)]}
    import tempfile, os
    rows = [tree(0.4, 0, 4) for _ in range(20)] + [tree(0.7, 2, 1) for _ in range(20)]
    fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    open(p, "w").write("\n".join(json.dumps(r) for r in rows))
    m0 = tree_metrics(rows[0]); m1 = tree_metrics(rows[-1])
    assert m0["orph"] == 1.0 and m1["orph"] == 0.25, (m0, m1)
    assert m0["cpk"] == 0.0 and m1["cpk"] == 2.0
    assert not m0["alive"] or m0["R"] == 0.4
    print("[demo] tree_metrics OK; sample report:")
    report(p, nbins=2); os.remove(p)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--demo":
        _demo()
    else:
        nb = 6
        if "--bins" in sys.argv:
            nb = int(sys.argv[sys.argv.index("--bins") + 1])
        report(sys.argv[1], nb)
