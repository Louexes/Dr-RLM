# sft/ — DR-RLM supervised fine-tuning (recursive cold start)

Cold-start SFT distills recursive deep-research trajectories into the student (Qwen3.5-4B),
masking the `tool_output` / `subagent_output` spans so loss is on policy tokens only.
**llama-factory** is vendored at **`llama-factory/`** (mirroring dr-tulu's SFT layout); the
final committed SFT runs use **SkyRL's native SFT trainer** instead.

## Pipeline
1. **Generate** recursive SFT trajectories — `../rl/skyrl/examples/train/dr_rlm/sft/gen_recursive_sft.py`
   emits sharegpt `conversations` using the SAME depth-banded prompt the RL env serves (so SFT and RL
   agree on the system turn).
2. **Train** with llama-factory on those conversations. Template config lives in `llama-factory/train/`
   (dr-tulu shipped a `qwen3-8B-sft-final.yaml` to start from).

> **Parity requirement:** SFT and RL must share (a) the chat template / tokenizer and (b) the masked
> spans — otherwise SFT→RL drifts. Pin both to the same base; mask exactly the observation spans the
> RL loss masks.
