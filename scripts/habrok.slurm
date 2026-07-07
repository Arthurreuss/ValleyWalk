#!/bin/bash
# scripts/habrok.slurm
#
# Habrok (RUG) SLURM launcher — general dispatcher for every experiment script.
# (Formerly habrok_cifar_PM.slurm, which only ran the CIFAR P/M blocks.)
#
# One SLURM array task == one (condition, seed) == one train.py run on one GPU.
# All Hydra overrides are owned by the run_*.sh scripts; this launcher only
# selects which script + block + condition + seed each array task runs, so
# cluster runs stay in lock-step with local runs (no override duplication).
#
# ─── SUITES ─────────────────────────────────────────────────────────────────
# Pick one or more suites via the SUITE env var (space-separated).  Each suite
# expands to a fixed list of (condition, seed) array tasks:
#
#   D           decomposition (run_D.sh, rot-MNIST):  D1..D6 × 5 seeds, µ both  = 30
#   C           curriculum    (run_C.sh, rot-MNIST):  C1..C7 × 5 seeds, µ both  = 35
#   L           long sequence (run_L.sh, rot-MNIST):  L1..L3 × 5 seeds, µ both  = 15
#   P           PER sweep     (run_P.sh, rot-MNIST):  δ∈{1.0,0.3,0.1,0.03} × 5  = 20
#   G-headline  CIFAR headline(run_G.sh):             G1..G6 × 5 seeds          = 30
#   G-gen       CIFAR general.(run_G.sh):             Gs1..Gt1v × 3 seeds       = 18
#   G-per       CIFAR asym PER(run_G.sh):             G7,G8 × 5 seeds  (CG SLOW) = 10
#   G-mom       CIFAR momentum(run_G.sh):             G9,G10,G11 × 5 seeds      = 15
#
# Default SUITE is "G-per G-mom" (25 tasks) — the old habrok_cifar_PM behaviour.
#
# The rot-MNIST suites (D, C, L, P) run an MLP in seconds and do not need
# a GPU, but running them here is harmless if you want one submission path.
#
# ─── USAGE ────────────────────────────────────────────────────────────────
# Submit FROM THE REPO ROOT (git tree must be clean — the run_*.sh enforce it).
#
#   0) Preview the task list + count for a suite WITHOUT SLURM (sets --array):
#        SUITE="G-per G-mom" bash scripts/habrok.slurm
#        SUITE="D C P"       bash scripts/habrok.slurm
#
#   1) ONE-TIME on the login node, pre-download CIFAR-10 (only needed for G-*):
#        python -c "import torchvision as t; \
#          t.datasets.CIFAR10('data', train=True,  download=True); \
#          t.datasets.CIFAR10('data', train=False, download=True)"
#
#   2) Cost-probe a single slow task first (G-per is CG-heavy — tune --time):
#        SUITE="G-per" sbatch --array=1 scripts/habrok.slurm
#
#   3) Launch a suite (set --array to the count the preview printed):
#        SUITE="G-per"  sbatch --array=1-10 --time=1-00:00:00 scripts/habrok.slurm
#        SUITE="G-mom"  sbatch --array=1-15 --time=08:00:00   scripts/habrok.slurm
#        SUITE="D"      sbatch --array=1-30 scripts/habrok.slurm
#      …or throttle concurrency (max 8 at a time):
#        SUITE="G-headline" sbatch --array=1-30%8 scripts/habrok.slurm
#
# SUITE must be exported to sbatch: prefix it (SUITE=... sbatch ...) or use
# `sbatch --export=ALL,SUITE="..."`.  --array must match the printed count.
#
# Checkpoints + results upload to W&B automatically (configs/config.yaml).
# Auth comes from ~/.netrc written by `wandb login` on the login node.
#
# ─── SBATCH DIRECTIVES ─────────────────────────────────────────────────────
#SBATCH --job-name=vw_habrok
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-25
#SBATCH --output=logs/habrok_%A_%a.out
#SBATCH --error=logs/habrok_%A_%a.err

set -euo pipefail

# ─── Suite selection ──────────────────────────────────────────────────────
SUITE="${SUITE:-G-per G-mom}"
SEEDS_5=(1 2 3 4 5)
SEEDS_3=(1 2 3)

# Each TASKS entry is "SCRIPT|ENV ASSIGNMENTS|human-label".  The middle field
# may contain spaces (multiple VAR=val); IFS='|' read keeps it intact.
TASKS=()
add() { TASKS+=("$1"); }

for suite in $SUITE; do
    case "$suite" in
        D)
            for c in D1 D2 D3 D4 D5 D6; do for s in "${SEEDS_5[@]}"; do
                add "run_D.sh|CONDITIONS=$c SEEDS=$s MOMENTUM_SET=both|D/$c/s$s"
            done; done ;;
        C)
            for c in C1 C2 C3 C4 C5 C6 C7; do for s in "${SEEDS_5[@]}"; do
                add "run_C.sh|CONDITIONS=$c SEEDS=$s MOMENTUM_SET=both|C/$c/s$s"
            done; done ;;
        L)
            for c in L1 L2 L3; do for s in "${SEEDS_5[@]}"; do
                add "run_L.sh|CONDITIONS=$c SEEDS=$s MOMENTUM_SET=both|L/$c/s$s"
            done; done ;;
        P)
            for d in 1.0 0.3 0.1 0.03; do for s in "${SEEDS_5[@]}"; do
                add "run_P.sh|DAMPINGS=$d BUFFER_SET=both MOMENTUM_SET=off SEEDS=$s|P/d$d/s$s"
            done; done ;;
        G-headline)
            for c in G1 G2 G3 G4 G5 G6; do for s in "${SEEDS_5[@]}"; do
                add "run_G.sh|BLOCKS=headline CONDITIONS=$c SEEDS_HEADLINE=$s|G/$c/s$s"
            done; done ;;
        G-gen)
            for c in Gs1 Gs1v Gs2 Gs2v Gt1 Gt1v; do for s in "${SEEDS_3[@]}"; do
                add "run_G.sh|BLOCKS=generalization CONDITIONS=$c SEEDS_GEN=$s|G/$c/s$s"
            done; done ;;
        G-per)
            for c in G7 G8; do for s in "${SEEDS_5[@]}"; do
                add "run_G.sh|BLOCKS=per_asym CONDITIONS=$c SEEDS_PER=$s|G/$c/s$s"
            done; done ;;
        G-mom)
            for c in G9 G10 G11; do for s in "${SEEDS_5[@]}"; do
                add "run_G.sh|BLOCKS=momentum_cross CONDITIONS=$c SEEDS_CROSS=$s|G/$c/s$s"
            done; done ;;
        *)
            echo "ERROR: unknown SUITE '$suite'. Valid: D C L P G-headline G-gen G-per G-mom" >&2
            exit 1 ;;
    esac
done

NTASKS="${#TASKS[@]}"

# ─── Preview mode (no SLURM) ──────────────────────────────────────────────
# Running this file directly (not under sbatch) prints the plan and the
# --array range to use, then exits without launching anything.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SUITE = ${SUITE}"
    echo "${NTASKS} array tasks  ->  submit with  --array=1-${NTASKS}"
    echo "---"
    i=1
    for t in "${TASKS[@]}"; do
        IFS='|' read -r script envs label <<< "$t"
        printf "  %3d  %-10s  %-14s  %s\n" "$i" "$script" "$label" "$envs"
        i=$((i+1))
    done
    exit 0
fi

# ─── Environment ──────────────────────────────────────────────────────────
# Use the local .venv built by `uv sync` on the login node (correct CUDA wheels).
module purge
export PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

# ─── Select this array task ───────────────────────────────────────────────
idx=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "$NTASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (1..$NTASKS for SUITE='$SUITE')" >&2
    exit 1
fi
IFS='|' read -r SCRIPT ENVS LABEL <<< "${TASKS[$idx]}"

echo "=========================================================="
echo " Array task : $SLURM_ARRAY_TASK_ID / $NTASKS"
echo " Suite      : $SUITE"
echo " Script     : $SCRIPT"
echo " Task       : $LABEL"
echo " Env        : $ENVS"
echo " Node       : $(hostname)"
echo " Python     : $PYTHON"
echo "=========================================================="

# ─── CIFAR-10 download guard (only for the G-* suites) ────────────────────
# Serialise first-time download/extract across array tasks sharing this FS.
# Once present, torchvision's integrity check makes this a fast no-op.
if [ "$SCRIPT" = "run_G.sh" ]; then
    (
        flock -x 200
        "$PYTHON" - <<'PY'
import torchvision as t
t.datasets.CIFAR10("data", train=True,  download=True)
t.datasets.CIFAR10("data", train=False, download=True)
PY
    ) 200>data/.cifar_download.lock
fi

# ─── Dispatch ─────────────────────────────────────────────────────────────
# One condition, one seed.  The run_*.sh applies every Hydra override.  Pass
# CROSS_NORM/PER_CG_ITERS through if set in the submit environment (they only
# affect the CIFAR per_asym / momentum blocks; harmless otherwise).
env $ENVS \
    N_JOBS=1 \
    CROSS_NORM="${CROSS_NORM:-gn}" \
    PER_CG_ITERS="${PER_CG_ITERS:-25}" \
    bash "scripts/$SCRIPT"

echo "=== array task $SLURM_ARRAY_TASK_ID ($LABEL) done ==="
