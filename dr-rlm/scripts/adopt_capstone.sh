#!/bin/bash
# Adopt the FAIR final harness config as CANONICAL. Run only when (a) Louis has approved and
# (b) no SLURM jobs are queued/running that import the harness (RL or eval) — the script
# refuses otherwise. Everything it changes is listed before it acts.
#
# FAIRNESS RULING (2026-06-16): the final harness is re-claim + MEASURE only, with NO
# check_citations / verification — the citation FILTER made a bespoke LLM judge call (a
# capability DR-Tulu's toolset has no analog of) and was removed. Citation validity is left
# to RL. So this script does NOT enable CHECK_TOOL anywhere.
#
# What it does:
#   1. Backs up prompts/system_prompt.txt -> prompts/system_prompt.pre_capstone.txt
#   2. Makes prompts/variants/final_v4.txt the new canonical system_prompt.txt
#      (re-claim carry rule + generic MEASURE step; no verification)
#   3. Affirms agent/frozen_baseline_v1.job has CHECK_TOOL disabled
#   4. Neutralizes the RQA task framing; runs the dr_rlm test suite
# NOT done here (manual, later): the git tag (needs the .gitignore decision).
set -euo pipefail
DRRLM=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
PY=/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers/bin/python

if squeue -u "$USER" -h -o '%j' | grep -vE '^jina_poller$' | grep -q .; then
  echo "REFUSING: jobs other than jina_poller are queued/running:"; squeue -u "$USER"; exit 1
fi

cd "$DRRLM"
cp prompts/system_prompt.txt prompts/system_prompt.pre_capstone.txt
# The variant's header comments explain the experiment; rewrite the header for canonical use.
{ echo "# DR-RLM canonical system prompt (v4, adopted 2026-06-16) — re-claim child-citation"
  echo "# carry + generic MEASURE step. NO check_citations / verification (fairness ruling:"
  echo "# a bespoke LLM judge tool DR-Tulu lacks; citation validity is left to RL). Perf"
  echo "# reference: runs/offline_ab_reclaim (DRB 3.8 valid cites; sqav2 cite-P 0.19)."
  echo "# Prior canonical preserved at system_prompt.pre_capstone.txt. Parsed by prompts.py."
  grep -v '^#' prompts/variants/final_v4.txt | sed '/./,$!d'
} > prompts/system_prompt.txt

# Affirm the frozen baseline does NOT enable the citation tool (fair config).
if grep -q '^  export CHECK_TOOL=' agent/frozen_baseline_v1.job; then
  sed -i 's/^  export CHECK_TOOL=.*/  export CHECK_TOOL=0/' agent/frozen_baseline_v1.job
fi
if grep -q '^  export CITE_BOUNCE=' agent/frozen_baseline_v1.job; then
  sed -i 's/^  export CITE_BOUNCE=.*/  export CITE_BOUNCE=0/' agent/frozen_baseline_v1.job
fi
grep -q 'CHECK_TOOL=1' agent/frozen_baseline_v1.job && { echo "CHECK_TOOL still enabled — abort"; exit 1; } || true

# Neutralize the RQA task framing (validated at zero cost vs the REPL-specific phrasing —
# runs/offline_ab_rqa_neutral: coverage 0.6016 = 0.602 — PAIRED with the canonical prompt's
# new step-0 MEASURE principle, which this script just installed).
$PY - <<'PYEOF'
import re
p = 'agent/infer_driver.py'
s = open(p).read()
neutral = (
    '        "Answer the question completely and precisely in 240-260 words. Write one to '
    'three "\n        "paragraphs (do not enumerate the facts), and support every statement '
    'with an "\n        "in-line citation to a retrieved snippet."'
)
m = re.search(r'"researchqa": \(\n(.*?)\n    \),', s, re.DOTALL)
assert m, "researchqa framing entry not found"
s = s.replace(m.group(1), neutral)
open(p, 'w').write(s)
print("[adopt_capstone] RQA framing neutralized (system-agnostic statement)")
PYEOF

cd rl/skyrl && PYTHONPATH=$PWD:$PWD/skyrl-gym $PY -m pytest examples/train/dr_rlm/tests/ -q
echo "[adopt_capstone] DONE — canonical prompt swapped, RQA framing neutralized, frozen baseline updated, tests green."
echo "[adopt_capstone] Next: git snapshot/tag (needs .gitignore decision), then fire the baseline."
