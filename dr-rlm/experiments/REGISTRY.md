# Experiment Registry

Index of every experiment. Each has `jobs/` (SLURM), `NOTES.md` (question + result), and writes to `drrlm/runs/<ARM>`.

| Experiment | Date | Question | Status | Headline result |
|---|---|---|---|---|
| [w1_harness_ab](w1_harness_ab/) | 2026-05 → 06 | Does the REPL substrate (A2) beat flat DR-Tulu ReAct (A3), same frontier model? vs trained DR-Tulu-8B (A1)? | partial | A2 ≫ A3 on DRB & HealthBench (length-confounded); recursion null at inference except sqav2 high-breadth (+0.049) |
| [healthbench_gemini](healthbench_gemini/) | 2026-06-04 | A2-rec vs A3 on HealthBench-72, gemini-2.5-flash both arms | done | A2-rec **0.228** vs A3 **0.078** (matched-51, CI [+0.089,+0.213]); **recursion fired 0%** |
| [recursion_probes](recursion_probes/) | 2026-06-05 | Is the 0% HealthBench recursion the model or the task? Which models recurse? | done | Model-family trait: gpt-5-mini 100%, gemini-2.5-flash 25%, gemini-3.1-flash-lite 0%, gemini-3.1-pro 0% (on 12 sqav2 items gpt-5 recursed on) |
| [_archive](_archive/) | — | superseded / early smokes & pre-A2-rec naming | archived | n/a |

## Arms glossary
- **A1** = DR-Tulu-8B (trained reference, GPU/vLLM).
- **A2** = frontier model in the `rlm/` REPL substrate. A2-rec = recursion on (depth=2, per-node budget=10). (A2-flat retired — A3 is the flat baseline.)
- **A3** = frontier model **flat** in DR-Tulu's ReAct harness (the flat baseline).

## Models used as the agent so far
gpt-5-mini (credits exhausted), gemini-2.5-flash, gemini-3.1-flash-lite, gemini-3.1-pro-preview. Graders: gemini-2.5-flash (HealthBench), gemini-2.5-pro (DRB RACE), gpt-4.1-mini (RQA).
