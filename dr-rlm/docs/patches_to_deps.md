# Patches to vendored dependencies

The vendored deps are gitignored, so our edits to them are recorded here. Reference deps (`rlm/`,
`dr-tulu/`) stay at the repo root; the RL backbone is vendored at `dr-rlm/rl/skyrl/` and the SFT
framework at `dr-rlm/sft/llama-factory/` (dr-tulu-style layout, since 2026-06-08).
Re-apply if the deps are re-cloned.

- **2026-06-28 — `dr-tulu/agent/evaluation/research_qa_eval/compute_coverage.py`: empty report → coverage 0
  (skip judge).** In `compute_coverage()`, before the per-batch judge loop, short-circuit any
  empty/whitespace `answer` to all-"Not at all" (coverage 0.0) without calling the LLM judge. **Why:**
  the grader sent an empty `Response:` to the judge, which hallucinated "Completely" on every rubric →
  coverage **1.000 for an empty report** (caught in the flat-baseline smoke, job 24271879). The untrained
  model produces ~50% empty reports (bare `search()` w/o `print()`), so unpatched this would massively
  INFLATE the untrained baseline coverage. Verified: re-grading 2 empty smoke reports → 0.000 each,
  judge skipped (no API calls). Non-empty grading path unchanged.

## rlm/ (the RLM engine)
- `rlm/rlm/core/rlm.py` (~L819): child RLM construction now passes `orchestrator=self.orchestrator`
  (was unset → defaulted True). Needed so children don't re-append the orchestrator addendum when it's
  baked into our `custom_system_prompt` (`dr-rlm/prompts/legacy_system_prompt.txt`). Date 2026-06-05.

## dr-tulu/ (graders + loaders)
- `dr-tulu/agent/evaluation/samplers/sampler/chat_completion_sampler.py`: added `reasoning_effort` param,
  forwarded to `chat.completions.create`. Lets us disable Gemini "thinking" for grading.
- `dr-tulu/agent/scripts/evaluate.py` (healthbench grader setup): for gemini graders, `max_tokens=4096`
  + `reasoning_effort="none"` (gemini is a thinking model; 1000-tok cap truncated JSON mid-value →
  infinite `while True` retry in healthbench_eval). Date 2026-06-04.
- `dr-tulu/agent/dr_agent/dataset_utils/load_dataset.py`: `load_sqav2_data` + `load_healthbench_data`
  honor `SQAV2_LOCAL_PATH` / `HEALTHBENCH_LOCAL_PATH` env (frozen subsets for A2 & A3).
- `dr-tulu/agent/evaluation/sqa_eval/convert_to_asta_format.py`: two bug fixes (KeyError 'generated_text'
  on native tool-calling output; string-index-out-of-range on empty sections) — needed to grade A3/A2 sqav2.

## SkyRL (RL backbone) — now vendored at `dr-rlm/rl/skyrl/`
- `dr-rlm/rl/skyrl/examples/train/dr_rlm/` — our DR-RLM training package (RER reward, per-node credit,
  ladder L1–L4, judge, corpus-search REPL fn, generator un-flatten, advantage estimator). Authored code,
  not a patch.
- `dr-rlm/rl/skyrl/skyrl-gym/.../envs/rlm/` + `examples/train/rlm/` — final-answer contract unified to the
  upstream `answer["ready"]` dict (FINAL/FINAL_VAR removed; event-driven `_AnswerDict`); 60 tests pass.
  Date 2026-06-08.

## llama-factory (SFT framework) — vendored at `dr-rlm/sft/llama-factory/`
- Copied from `dr-tulu/sft/llama-factory` (the proven recursive-SFT setup). No edits yet. Date 2026-06-08.

## dr-tulu/agent/evaluation/samplers/sampler/chat_completion_sampler.py
- `ChatCompletionSampler.__call__` catch-all retry: capped the exponential backoff at 60s
  (`min(2**trial, 60)`, was uncapped `2**trial`). Every exception lands in that handler — not
  just rate limits — so transient Gemini `404 model_not_found` bursts escalated to 1024-2048s
  sleeps and stalled grading jobs for hours (observed 3x on 2026-06-11: killed cross-bench job
  23669005, two stalled HealthBench grading attempts). Cap keeps retry-forever semantics but
  recovers from bursts in minutes. Date 2026-06-11.

## dr-tulu/agent/workflows/auto_search_sft.py — STEELMAN patch (task-statement parity)
- `SearchAgent.prompt()` + `AutoReasonSearchWorkflow.__call__` now accept
  `additional_instructions`; a non-empty per-example value OVERRIDES the YAML category
  instruction (exact_answer/short_form/long_form), and the workflow threads it through to the
  agent. Rationale: every dataset loader in `dr_agent/dataset_utils/load_dataset.py` builds
  per-example `additional_instructions` (e.g. ResearchQA's "240-260 words ... one-to-three
  paragraphs"), but `workflow.generate_dataset` filters dataset fields to `__call__`'s
  signature, so they were SILENTLY DROPPED — the agent only ever saw the generic YAML category
  text ("write a short paragraph" for DRB/RQA/HB). For DR-RLM-vs-DR-Tulu head-to-heads both
  systems must receive the SAME task statement; this patch makes that possible (and restores
  the loaders' clear intent). Absent/empty instructions keep the YAML fallback byte-identical
  to upstream. Verified: override + fallback behavior, py_compile. Date 2026-06-12.
- (part 2, same file/date) `DR_TULU_FRAMING_JSON` env override in `SearchAgent.prompt()`:
  a JSON map dataset_name -> task statement with HIGHEST precedence (> per-example >
  YAML category). Exporting the same JSON that drives DR-RLM's `_TASK_FRAMING` /
  `DR_RLM_TASK_FRAMING_JSON` gives both systems byte-identical task statements in
  head-to-heads. Verified precedence chain; absent env keeps prior behavior.

## dr-tulu/agent/dr_agent/tool_interface/local_bm25s.py (NEW) + auto_search_sft.py — bm25s parity bridge
- NEW `local_bm25s.py`: `LocalBm25sSearchTool` + `LocalBm25sBrowseTool` give DR-Tulu's ReAct
  harness retrieval BYTE-IDENTICAL to DR-RLM. Both reuse the RLM's in-process `_Bm25sBackend`
  (`rl/skyrl/examples/train/dr_rlm/corpus_search.py`, loaded by file path) over the SAME frozen
  index (`data/frozen_corpus/bm25s_index`, 139,257 docs). Why: the controlled comparison
  (DR-Tulu ReAct vs DR-RLM REPL) must hold retrieval constant — DR-Tulu's stock search is web
  (Serper/MCP) or pyserini/Lucene (Java). Search subclasses `MCPSearchTool` and OVERRIDES
  `__call__` to call bm25s in-process — NO MCP server (MCPMixin.__init__ is fully lazy; any
  connection only happens inside the un-called MCP pipeline). Returns Serper-shaped `Document`s
  (Title/URL/Snippet; corpus `id` NOT surfaced — DR-Tulu attributes citations by its native URL
  scheme = a legitimate harness difference). Browse subclasses `MCPBrowseTool`, resolves
  url -> full corpus contents (concatenated passages), mirroring the RLM's `get_doc` so DR-Tulu's
  browse-trained policy isn't handicapped by an inert NoBrowseTool. `auto_search_sft.py`: added
  `search_tool_name=="bm25s"` + `browse_tool_name=="bm25s"` branches, `bm25s_index_path` /
  `bm25s_corpus_path` config fields, and a guard so `before_launch_check` SKIPS the MCP-server
  check on the bm25s path (no interactive Confirm.ask in batch). Verified in isolation (no GPU):
  loads the 139k index, search returns 8 on-topic docs, browse resolves url->text incl. chaining
  + corpus-miss, py_compile + workflow import OK. Launcher: `dr-rlm/agent/eval_drtulu_untrained.job`.
  Date 2026-06-28.

- **2026-07-01 — `dr-tulu/agent/evaluation/samplers/common.py`: cap `map_with_progress` pool at 8.**
  Changed `num_threads: int = os.cpu_count() or 10` → `min(os.cpu_count() or 10, 8)`. **Why:** HealthBench
  grading nests this ThreadPool (outer over examples × inner over each example's rubrics). On a 96-core
  node that's ~96×N threads → thousands × 8MB stacks → `RuntimeError: can't start new thread` (crashes the
  whole grade after "Loaded N examples", no score). 8×8=64 peak is thread- and Gemini-rate-safe. Only
  affects eval grading (not generation). The `--debug`/`debug=1` serial path also avoids it but is ~10×
  slower. This is what makes off-GPU Gemini HB grading actually work; see docs/EVAL_GRADING_REQUIREMENTS.md
  + [[project-grading-off-gpu]].

## rlm/rlm/core/rlm.py — `sub_system_prompt` kwarg (depth-band parity for SFT teacher gen)
2026-07-03. The library hands spawned children `custom_system_prompt=self.system_prompt`
verbatim — every node gets the ROOT's prompt. DrRlmEnv (RL + unified eval driver) serves
depth bands (orchestrator root / worker children). Added optional `sub_system_prompt`:
children receive it when set (and propagate it to grandchildren); `None` = original
behavior, byte-identical. Used by `gen_recursive_sft.py` so SFT teacher trajectories match
the env's per-band prompts. Related gotcha fixed at the caller: the library runs
`.format(custom_tools_section=...)` over the prompt (rlm/utils/prompts.py:228); the
canonical prompt's literal `{id, text, url}` braces KeyError'd every rollout. Caller now
fills the section env-style (plain .replace) and brace-escapes before handing over.
