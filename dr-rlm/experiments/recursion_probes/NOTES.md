# recursion_probes

**Question:** HealthBench A2-rec recursed 0%. Is that the *model* or the *task*? And which models recurse?

**Setup:** Re-run, with different agent models, the **exact 12 sqav2 items where gpt-5-mini fanned out
widely** (`data/subsets/sqav2_gemini_recprobe12.jsonl`, gpt-5 sub-agent widths 11–26). Identical config
(depth=2, orchestrator on, per-node budget=10), generation-only. Jobs: `a2rec_sqav2_gemprobe.job`
(gemini-2.5-flash), `a2rec_sqav2_g31flite_probe.job`, `a2rec_sqav2_g31pro_probe.job`. Outputs in
`drrlm/runs/A2rec_sqa_*`.

**Result — recursion is a MODEL-FAMILY trait, not a capability tier:**
| Model | Recursion on the 12 | Note |
|---|---|---|
| gpt-5-mini | 12/12 (100%) | selection baseline; overall sqav2 69% |
| gemini-2.5-flash | 3/12 (25%) | fired 9/5/4 sub-agents when it did |
| gemini-3.1-flash-lite | 0/8 (0%) | +4 crashed on empty model output |
| gemini-3.1-pro-preview | 0/10 (0%) | heavy FLAT search instead (mean ~4, up to 8) |

Flagship Gemini recursed 0% (below 2.5-flash) → not capability-ordered. **No Gemini model is a viable
recursive SFT teacher; GPT-5-class required.** Untested lever: a Gemini-tuned orchestrator prompt
(all probes used the same prompt). Also contains `a2rec_rewire_smoke.job` (validated the single-prompt rewire).
