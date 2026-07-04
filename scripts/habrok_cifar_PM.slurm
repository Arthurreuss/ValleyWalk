#!/bin/bash
# scripts/habrok_cifar_PM.slurm
#
# Habrok (RUG) SLURM launcher for the CIFAR "P" and "M" blocks of run_cifar.sh.
#
#   P  = per_asym       block: P1, P2         (asym PER δ=0.1, BN/GN) — SLOW (CG)
#   M  = momentum_cross block: M1, M2, M3     (µ=0 legs, norm=CROSS_NORM)
#
# One SLURM array task == one (condition, seed) == one train.py run on one GPU.
# All Hydra overrides are owned by scripts/run_cifar.sh — this launcher just
# selects which condition+seed each array task runs, so cluster runs stay in
# lock-step with local runs (no override duplication to drift).
#
# Array index → task map (built below):
#   1 – 10   P block  (P1,P2 × seeds 1..5)   ← CG-heavy, give it more --time
#   11 – 25  M block  (M1,M2,M3 × seeds 1..5)
#
# Checkpoints + results are uploaded to W&B automatically (already enabled in
# configs/config.yaml: tracking.wandb.artifacts.log_checkpoints/log_results).
# Auth comes from the ~/.netrc written by `wandb login` on the login node
# (shared home FS → visible on compute nodes). No key needs to live in here.
#
# ─── USAGE ────────────────────────────────────────────────────────────────
# Submit FROM THE REPO ROOT (git tree must be clean — run_cifar.sh enforces it).
#
#   0) ONE-TIME, on the login node, pre-download CIFAR-10 so 25 jobs don't race
#      the download/extract (the script also flock-guards this, but doing it
#      once up front is the clean fix):
#        python -c "import torchvision as t; \
#          t.datasets.CIFAR10('data', train=True,  download=True); \
#          t.datasets.CIFAR10('data', train=False, download=True)"
#
#   1) Cost-probe ONE P seed first (per_asym runs ~25 Fisher-vector products
#      per step — time it before committing the block, then tune --time):
#        sbatch --array=1 scripts/habrok_cifar_PM.slurm
#
#   2) Launch everything (all 25):
#        sbatch scripts/habrok_cifar_PM.slurm
#
#      …or split so each block gets a fitting wall limit:
#        sbatch --array=1-10  --time=1-00:00:00 scripts/habrok_cifar_PM.slurm   # P (slow)
#        sbatch --array=11-25 --time=08:00:00   scripts/habrok_cifar_PM.slurm   # M
#
#      …or throttle concurrency (max 8 at a time) to be nice to the queue:
#        sbatch --array=1-25%8 scripts/habrok_cifar_PM.slurm
#
# ─── SBATCH DIRECTIVES ─────────────────────────────────────────────────────
#SBATCH --job-name=vw_cifar_PM
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-25
#SBATCH --output=logs/cifar_PM_%A_%a.out
#SBATCH --error=logs/cifar_PM_%A_%a.err

set -euo pipefail

# ─── Environment ──────────────────────────────────────────────────────────
# We are using the local .venv built by `uv sync` on the login node. 
# This environment already contains the correct Linux/CUDA wheels.

# Ensure the cluster's base modules don't interfere with our uv environment
module purge

# Explicitly set the Python path to our local virtual environment
export PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

# ─── CIFAR-10 download guard (safety net for step 0 above) ────────────────
# Serialise first-time download/extract across array tasks sharing this FS.
# Once the data is present torchvision's integrity check makes this a fast
# no-op, so the lock is only ever contended on a truly cold start.
(
    flock -x 200
    "$PYTHON" - <<'PY'
import torchvision as t
t.datasets.CIFAR10("data", train=True,  download=True)
t.datasets.CIFAR10("data", train=False, download=True)
PY
) 200>data/.cifar_download.lock

# ─── Array index → (block, condition, seed) ───────────────────────────────
SEEDS=(1 2 3 4 5)
TASKS=()
for c in P1 P2;    do for s in "${SEEDS[@]}"; do TASKS+=("per_asym $c $s");       done; done  # idx 1..10
for c in M1 M2 M3; do for s in "${SEEDS[@]}"; do TASKS+=("momentum_cross $c $s"); done; done  # idx 11..25

idx=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#TASKS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (1..${#TASKS[@]})" >&2
    exit 1
fi
read -r BLOCK COND SEED <<< "${TASKS[$idx]}"

echo "=========================================================="
echo " Array task : $SLURM_ARRAY_TASK_ID / ${#TASKS[@]}"
echo " Block      : $BLOCK"
echo " Condition  : $COND"
echo " Seed       : $SEED"
echo " Node       : $(hostname)"
echo " Python     : $PYTHON"
echo "=========================================================="

# ─── Dispatch ─────────────────────────────────────────────────────────────
# One condition, one seed. run_cifar.sh applies every Hydra override for the
# selected block/condition. SEEDS_PER drives per_asym, SEEDS_CROSS drives
# momentum_cross; setting both to $SEED is harmless (only the chosen block
# runs). CROSS_NORM/PER_CG_ITERS keep run_cifar.sh's defaults unless overridden
# in the submit environment.
CONDITIONS="$COND" \
BLOCKS="$BLOCK" \
SEEDS_PER="$SEED" \
SEEDS_CROSS="$SEED" \
CROSS_NORM="${CROSS_NORM:-gn}" \
PER_CG_ITERS="${PER_CG_ITERS:-25}" \
N_JOBS=1 \
bash scripts/run_cifar.sh

echo "=== array task $SLURM_ARRAY_TASK_ID ($BLOCK/$COND seed $SEED) done ==="
