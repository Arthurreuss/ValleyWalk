#!/usr/bin/env bash
# scripts/probe_ncl_hparams.sh
#
# Small NCL hyperparameter sweep on rot-MNIST (T0=0°, T1=90°, MLP). Each
# config writes to its own subdir; at the end we print a side-by-side
# summary of ACC / FORG / stability-gap metrics.
#
# The four configs probe one hypothesis each:
#   baseline      paper-like reference (α=1.0, ρ=0.9)
#   loose_prior   α=0.1 — does a weaker rubber-band buy task-1 plasticity?
#   tight_prior   α=10.0 — does a stronger rubber-band reduce forgetting?
#   no_momentum   ρ=0 — is momentum × rubber-band interaction biasing things?
#
# damping is held at 1e-3 (numerical safeguard only — the p_w = α·I prior
# is what actually bounds Λ⁻¹). lr is held at 0.1 to match the ER baselines.
#
# Override SEED to repeat with a different seed:
#   SEED=2 bash scripts/probe_ncl_hparams.sh

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

SEED="${SEED:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="outputs/_probe/ncl_sweep_${STAMP}"
mkdir -p "$SWEEP_DIR"

# Each row: name | prior_init | lr | momentum
# damping is fixed at 1e-3, fisher_samples at the cfg default (1000).
CONFIGS=(
    "baseline    | 1.0  | 0.1 | 0.9"
    "loose_prior | 0.1  | 0.1 | 0.9"
    "tight_prior | 10.0 | 0.1 | 0.9"
    "no_momentum | 1.0  | 0.1 | 0.0"
)

echo "Sweep root: $SWEEP_DIR"
echo "Seed:       $SEED"
echo

for row in "${CONFIGS[@]}"; do
    IFS='|' read -r name alpha lr mom <<< "$row"
    name="$(echo "$name" | xargs)"
    alpha="$(echo "$alpha" | xargs)"
    lr="$(echo "$lr" | xargs)"
    mom="$(echo "$mom" | xargs)"

    OUT_DIR="$SWEEP_DIR/$name"
    LOG="$SWEEP_DIR/${name}.stdout"
    echo "=== $name ===  prior_init=$alpha  lr=$lr  momentum=$mom"

    if "$PYTHON" scripts/train.py \
            method=ncl \
            dataset=rot_mnist \
            dataset.num_tasks=2 \
            'dataset.rotations_deg=[0,90]' \
            model=mlp \
            "method.ncl.prior_init=$alpha" \
            method.ncl.damping=1e-3 \
            "training.lr=$lr" \
            "training.momentum=$mom" \
            "seed=$SEED" \
            eval.stability_gap.eval_freq_steps=1 \
            eval.stability_gap.window_steps=250 \
            tracking.backend=csv_only \
            "hydra.run.dir=$OUT_DIR" \
            > "$LOG" 2>&1; then
        echo "  done."
    else
        echo "  FAILED — see $LOG"
    fi
done

echo
echo "=== Summary ==="
"$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("$SWEEP_DIR")
keys = ("ACC", "FORG", "stability_gap_depth", "stability_gap_area",
        "stability_gap_max_drop", "stability_gap_recovery_steps")

rows = []
for sub in sorted(p for p in root.iterdir() if p.is_dir()):
    js = next(sub.rglob("metrics_summary.json"), None)
    if js is None:
        rows.append((sub.name, None))
        continue
    with open(js) as f:
        rows.append((sub.name, json.load(f)))

if not rows:
    print("No runs found.")
else:
    name_w = max(len(n) for n, _ in rows)
    header = f"{'config':<{name_w}}  " + "".join(f"{k:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for n, m in rows:
        if m is None:
            print(f"{n:<{name_w}}  " + "".join(f"{'(no metrics)':>22}" for _ in keys))
            continue
        cells = []
        for k in keys:
            v = m.get(k)
            if isinstance(v, float):
                cells.append(f"{v:>22.4f}")
            elif v is None:
                cells.append(f"{'—':>22}")
            else:
                cells.append(f"{str(v):>22}")
        print(f"{n:<{name_w}}  " + "".join(cells))
PY
