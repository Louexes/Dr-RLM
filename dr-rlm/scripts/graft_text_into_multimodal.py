"""Graft a text-only SkyRL checkpoint (Qwen3_5ForCausalLM, model_type=qwen3_5_text) onto the
base multimodal shell (Qwen3_5ForConditionalGeneration) so it becomes vLLM-servable.

Why: SkyRL trains/exports text-only (language_model_only=true). vLLM 0.19 registers
Qwen3_5ForConditionalGeneration but NOT Qwen3_5ForCausalLM, so the raw SFT/RL export won't
serve. The text export already uses base-identical keys (model.language_model.*), and training
only touched the text backbone, so we overwrite those tensors on the base checkpoint and keep
the base vision + mtp weights and multimodal config untouched. Reusable for the RL export too.

  python scripts/graft_text_into_multimodal.py \
      --base /scratch-shared/.../Qwen3.5-4B/snapshots/<hash> \
      --text /scratch-shared/lgehringer/ckpts/dr_rlm_sft/hf/global_step_102/policy \
      --out  /scratch-shared/lgehringer/ckpts/dr_rlm_sft/served
"""

import argparse
import glob
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base multimodal snapshot dir (config + shards)")
    ap.add_argument("--text", required=True, help="text-only trained checkpoint dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text_shards = sorted(glob.glob(os.path.join(args.text, "*.safetensors")))
    # key -> (shard_file) for the text checkpoint (lazy access, no full load)
    text_key_file = {}
    for f in text_shards:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                text_key_file[k] = f
    # tied embeddings on base => skip the redundant lm_head the text export writes out
    skip = {"lm_head.weight"}

    grafted = kept = 0
    for shard in sorted(glob.glob(os.path.join(args.base, "*.safetensors"))):
        tensors, meta = {}, {}
        with safe_open(shard, framework="pt") as bh:
            meta = bh.metadata() or {}
            for k in bh.keys():
                if k in text_key_file and k not in skip:
                    with safe_open(text_key_file[k], framework="pt") as th:
                        t = th.get_tensor(k)
                    tensors[k] = t.to(torch.bfloat16)  # base language_model tensors are bf16
                    grafted += 1
                else:
                    tensors[k] = bh.get_tensor(k)
                    kept += 1
        meta.setdefault("format", "pt")
        save_file(tensors, os.path.join(args.out, os.path.basename(shard)), metadata=meta)
        print(f"[shard] {os.path.basename(shard)}: grafted so far={grafted} kept so far={kept}")

    # copy config + index + tokenizer/preprocessor from BASE (multimodal, servable arch)
    for f in os.listdir(args.base):
        if f.endswith(".safetensors"):
            continue
        src = os.path.join(args.base, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, f))

    text_skipped = [k for k in text_key_file if k in skip or k not in _base_keys(args.base)]
    print(f"\nDONE grafted={grafted} kept={kept} | text keys skipped (tied/absent-in-base)={text_skipped}")
    print(f"served checkpoint -> {args.out}")


def _base_keys(d):
    ks = set()
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            ks |= set(h.keys())
    return ks


if __name__ == "__main__":
    main()
