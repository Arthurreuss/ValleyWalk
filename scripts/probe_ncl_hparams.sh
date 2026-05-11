#!/usr/bin/env bash
# scripts/probe_ncl_hparams.sh
#
# Single-config NCL smoke run — exercises the standard baseline and prints
# the gap_depth / ACC / FORG that fall out, so you can sanity-check the
# implementation in ~30 seconds without launching the full thesis sweep.
#
# Baseline configuration (the values pinned across the thesis):
#   method.ncl.fisher_samples = 1000      paper-aligned K-FAC sample budget
#   method.ncl.damping        = 0.2       large enough for sparse MNIST inputs
#                                         (paper uses 1e-10 on dense features;
#                                         rot-MNIST's A factor has near-zero
#                                         eigenvalues that need more ε)
#   method.ncl.trust_radius   = 1.0       diagnostic only — not enforced
#   training.lr               = 0.1       matched to ER baselines in §A.5
#   training.momentum         = 0.9       paper always uses ρ=0.9
#   training.batch_size       = 256       thesis default
#   dataset                   = rot_mnist with T0=0°, T1=90°, 2 tasks
#   seed                      = 1
#
# Override any of these via env vars, e.g.
#   MOMENTUM=0.0 SEED=2 bash scripts/probe_ncl_hparams.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=""
for cand in ".venv/Scripts/python.exe" ".venv/bin/python" "python"; do
    if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"; break
    fi
done
[ -n "$PYTHON" ] || { echo "ERROR: no python found"; exit 1; }
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DAMPING="${DAMPING:-0.2}"
MOMENTUM="${MOMENTUM:-0.9}"
LR="${LR:-0.1}"
FISHER_SAMPLES="${FISHER_SAMPLES:-1000}"
SEED="${SEED:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/_probe/ncl_baseline_${STAMP}"
mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"
echo "Settings:   damping=$DAMPING  momentum=$MOMENTUM  lr=$LR  fisher_samples=$FISHER_SAMPLES  seed=$SEED"
echo "---"

"$PYTHON" scripts/train.py \
    method=ncl \
    dataset=rot_mnist \
    dataset.num_tasks=2 \
    'dataset.rotations_deg=[0,90]' \
    model=mlp \
    "method.ncl.damping=$DAMPING" \
    "method.ncl.fisher_samples=$FISHER_SAMPLES" \
    "training.lr=$LR" \
    "training.momentum=$MOMENTUM" \
    "seed=$SEED" \
    eval.stability_gap.eval_freq_steps=1 \
    eval.stability_gap.window_steps=250 \
    tracking.backend=csv_only \
    "hydra.run.dir=$OUT_DIR"

echo ""
echo "--- Metrics summary ---"
"$PYTHON" - <<PY
import json, sys
from pathlib import Path
root = Path("$OUT_DIR")
js = next(root.rglob("metrics_summary.json"), None)
if js is None:
    print("metrics_summary.json not found in $OUT_DIR", file=sys.stderr); sys.exit(1)
with open(js) as f:
    m = json.load(f)
for k in ("ACC", "FORG", "stability_gap_depth", "stability_gap_area",
          "stability_gap_max_drop", "stability_gap_recovery_steps"):
    v = m.get(k)
    print(f"  {k:<32} {v}")
PY
