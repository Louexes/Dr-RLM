# data/

## subsets/ — frozen benchmark subsets (provenance)
| File | N | What |
|---|---|---|
| `researchqa_strat49.jsonl` | 49 | ResearchQA stratified subset |
| `drb_en50.jsonl` | 50 | DeepResearchBench (English) subset |
| `sqav2_cs100.jsonl` | 100 | ScholarQA-CSv2, tagged `_rubric_count` / `_breadth_stratum` (low≤10:57 / mid11:10 / high≥12:33) |
| `sqav2_gemini_recprobe12.jsonl` | 12 | the 12 sqav2 items gpt-5-mini fanned out widely on (recursion probes) |
| `healthbench_hard_repr.jsonl` | 72 | HealthBench-hard representative: theme-proportional, difficulty-weighted (mean rubric 21.1), 31/72 multi-turn; tagged `_theme`/`_rubric_count`/`_n_turns` |

Built by `../analysis/build_subsets.py`. Loaders honor `SQAV2_LOCAL_PATH` / `HEALTHBENCH_LOCAL_PATH`
(set in `agent/run.sh`) so A2 and A3 read the identical frozen set.

## Training data (not stored here; HF)
- RL: `rl-research/dr-tulu-rl-data` (PUBLIC, 4,881 = OpenScholar+SearchArena subset; ~61% ≥3-facet).
- SFT: `rl-research/dr-tulu-sft-data` (13,062 traces, mean 3,064 tok).
- Corpus: `s42chen/wii-indexes` (gated).
