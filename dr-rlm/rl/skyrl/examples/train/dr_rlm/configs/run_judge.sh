set -x

# =============================================================================
# DR-RLM held-constant RUBRIC / CITATION JUDGE  (local Qwen3-8B via vLLM)
# -----------------------------------------------------------------------------
# The SAME judge scores every arm of the controlled comparison (L0..L4) — see
# proposal §6.1-6.2 (held constant). judge.py (RubricJudge / score_report_sync)
# talks to an OpenAI-compatible /v1/chat/completions endpoint; this script serves
# that endpoint locally so the whole reward path runs offline (zero external API).
#
# CONTRACT with the training scripts:
#   * The endpoint URL here MUST equal generator.judge_base_url in the L* scripts.
#     Default in dr_rlm_config.py is http://localhost:8100/v1  -> we serve on :8100.
#   * generator.judge_model defaults to "hosted_vllm/Qwen/Qwen3-8B"; JudgeConfig
#     strips the "hosted_vllm/" prefix, so the model id sent on the wire is
#     "Qwen/Qwen3-8B" — which is exactly what `vllm serve Qwen/Qwen3-8B` registers.
#   * The judge API key is a dummy: the L* scripts export JUDGE_API_KEY=EMPTY and
#     judge.py reads it via judge_api_key_env=JUDGE_API_KEY (vLLM ignores it unless
#     you pass --api-key, which we do NOT here).
#
# Run this FIRST and leave it running (own GPU / own node), then launch run_dr_rlm_L*.sh.
# =============================================================================

: "${JUDGE_MODEL_PATH:=Qwen/Qwen3-8B}"   # served model id == generator.judge_model (sans prefix)
: "${JUDGE_PORT:=8100}"                    # MUST match the :PORT in generator.judge_base_url
: "${JUDGE_HOST:=0.0.0.0}"
: "${JUDGE_GPU_UTIL:=0.85}"
: "${JUDGE_TP:=1}"                         # tensor-parallel size for the judge (1 GPU is plenty for 8B)
: "${JUDGE_MAX_LEN:=16384}"               # judge prompts (question+response+criterion) fit comfortably
# Pin the judge to a specific device so it does not contend with the trainer's GPUs.
: "${JUDGE_CUDA_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$JUDGE_CUDA_DEVICES}"

# vllm serve exposes an OpenAI-compatible server at http://$HOST:$PORT/v1
# (so generator.judge_base_url=http://localhost:$JUDGE_PORT/v1).
# --reasoning-parser/--enable-* are intentionally omitted: judge.py asks for plain
# JSON {"score": x} via response_format=json_object and temperature 0, and parses the
# first {...} it finds (RubricJudge._parse_score), tolerating any leading prose.
uv run vllm serve "$JUDGE_MODEL_PATH" \
  --host "$JUDGE_HOST" \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size "$JUDGE_TP" \
  --gpu-memory-utilization "$JUDGE_GPU_UTIL" \
  --max-model-len "$JUDGE_MAX_LEN" \
  --served-model-name "$JUDGE_MODEL_PATH" \
  "$@"

# Smoke test once it is up (in another shell):
#   curl -s http://localhost:${JUDGE_PORT:-8100}/v1/chat/completions \
#     -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
#     -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Output JSON {\"score\": 2}"}],
#          "temperature":0,"response_format":{"type":"json_object"}}'
