# w1_harness_ab

**Question:** Does the REPL substrate (A2) beat flat DR-Tulu ReAct (A3) with the same frontier model,
and how do both relate to the trained DR-Tulu-8B reference (A1)? Across ResearchQA / DRB / sqav2.

**Setup:** `jobs/` — A1 (`a1_drtulu.job`, `a1_sqav2.job`), A2 (`a2rec_*`, `a2flat_*`), A3 (`a3_gpt5mini.job`,
`a3_sqav2.job`, `a3_sqav2_grade.job`). Agent = gpt-5-mini (pre-Gemini-switch). Subsets in `data/subsets/`
(researchqa_strat49, drb_en50, sqav2_cs100). Outputs in `drrlm/runs/{A1,A2,A2flat,A2rec,A3}/`.

**Results (so far):**
- REPL (A2) ≫ flat (A3) on DRB RACE; lift is **length-confounded** (A2 longer; also early A2 FACT≈0 was a
  cite-format prompt typo, since fixed).
- Recursion ≈ flat at inference on RQA/DRB/sqav2 overall; **sqav2 high-breadth (≥12 rubric) stratum is the
  one positive: +0.049** (recursion's target regime).
- gpt-5-mini recursion rates here: DRB 72%, RQA 57%, sqav2 69% (mean fan-out ~3.5–4.4).
- A1 (trained DR-Tulu-8B) is the external "competitive with open SOTA" anchor (e.g. RQA ~0.781).

**Status:** partial (some cells incomplete — A2-rec/DRB 32 rows, A3/RQA, sqav2 A1/A2-flat). Superseded for
the recursion question by `recursion_probes/` and for HealthBench by `healthbench_gemini/`.
