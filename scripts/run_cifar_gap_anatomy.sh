#!/usr/bin/env bash
# scripts/run_cifar_gap_anatomy.sh
#
# CIFAR-10 stability-gap anatomy.  Tests three candidate explanations for
# why D3-D5 in run_curriculum.sh leave residual gap depth on dom_cifar10
# despite an adaptive λ-curriculum that fully closes the gap on rot-MNIST.
#
# Base condition is D5 (adaptive curriculum, λ_min=0.0, the smallest-floor
# variant from run_curriculum.sh) at µ=0.9.  Every H-condition modifies
# exactly one knob relative to that base.  Eval cadence is fine-grained
# (eval_freq_steps=1) on the whole sweep to capture step-by-step gap shape.
#
# Hypotheses + conditions (ablation_key = "cifar_gap_anatomy")
# ------------------------------------------------------------
#   H1 — D5 baseline, eval_freq_steps=1                       (high-res reference)
#   H2 — H1 + model.norm_type=group  (GroupNorm-ResNet-18)    BN-drives-gap test
#   H3 — H1 + memory.total_budget=5000 (5× buffer)            G_est test
#   H4 — H1 + training.momentum=0.0  (no momentum)            momentum sanity check
#   H5 — H1 + GroupNorm + buffer 5000  (combines H2 and H3)   "best of both" test
#
# H5 is the combined intervention: GroupNorm's faster post-transition
# recovery (seen in H2 at n=1) plus the larger buffer that uniformly
# helps in H3.  If GN's task-0 baseline deficit closes with a richer
# replay distribution, H5 should beat H3.  If GN's baseline cost
# persists, H3 (BN + 5k) ships as the headline configuration.
#
# All five conditions share method/dataset/eval otherwise → directly
# paired-by-seed against H1.
#
# Total runs
# ----------
#   5 conditions × 5 seeds = 25  (default SEEDS="1,2,3,4,5")
#
# Compute cost
# ------------
# eval_freq_steps=1 over a 250-step window adds ~250 extra forward passes
# on the CIFAR-10 test set per task transition relative to D5's standard
# eval_freq_steps=50.  Empirically ~+10-15 min wall-clock per seed on MPS.
#
# Usage
# -----
#   bash scripts/run_cifar_gap_anatomy.sh                          # all 4 × all seeds
#   CONDITIONS="H1 H2" bash scripts/run_cifar_gap_anatomy.sh       # subset
#   SEEDS="1,2,3" bash scripts/run_cifar_gap_anatomy.sh            # fewer seeds
#   N_JOBS=1     bash scripts/run_cifar_gap_anatomy.sh             # sequential (default)
#   DRY_RUN=1    bash scripts/run_cifar_gap_anatomy.sh             # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

# CIFAR runs on a single MPS GPU — default sequential.
N_JOBS="${N_JOBS:-1}"
SEEDS="${SEEDS:-1}"

ALL_CONDITIONS_DEFAULT="H1 H2 H3 H4 H5"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Base CIFAR overrides — two-task dom_cifar10 (clean → gaussian_noise) on
# ResNet-18, 10 epochs/task.  Identical to the D-block of run_curriculum.sh.
CIFAR_OVERRIDES=(
    dataset=dom_cifar10
    dataset.num_tasks=2
    'dataset.corruption_types=[none,gaussian_noise]'
    model=resnet18
    training.epochs_per_task=10
)

# Fine-grained eval at every step over a 250-step window — twice the default
# CIFAR window in run_curriculum.sh, so we capture the full recovery curve.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=50
    eval.eval_every_n_steps=50
)

# Adaptive λ-curriculum knobs (matches D5).
CURRICULUM_OVERRIDES=(
    method=er
    method.mode=standard
    method.lambda_curriculum.enabled=true
    method.lambda_curriculum.schedule=adaptive
    method.lambda_curriculum.ema_alpha=0.05
    method.lambda_curriculum.lambda_min=0.0
)

# Base momentum for the sweep (overridden by H4 only).
MOM_DEFAULT=0.9

ABLATION_KEY="cifar_gap_anatomy"

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────────

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:    ${GIT_COMMIT}"
echo "Python:        ${PYTHON}"
echo "Conditions:    ${CONDITIONS}"
echo "Seeds:         ${SEEDS}"
echo "N_JOBS:        ${N_JOBS}"
echo "Dry run:       ${DRY_RUN}"
echo "---"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_PIDS=()

spawn_job() {
    echo "  + $PYTHON scripts/train.py $*"
    if [ "$DRY_RUN" = "1" ]; then return; fi

    if [ "$N_JOBS" != "-1" ] && [ "${#BLOCK_PIDS[@]}" -ge "$N_JOBS" ]; then
        wait "${BLOCK_PIDS[0]}" || true
        BLOCK_PIDS=("${BLOCK_PIDS[@]:1}")
    fi

    "$PYTHON" scripts/train.py "$@" &
    BLOCK_PIDS+=($!)
}

wait_block() {
    local label="${1:-block}"
    [ "${#BLOCK_PIDS[@]}" -eq 0 ] && return
    echo "  Waiting for ${#BLOCK_PIDS[@]} jobs (${label})..."
    local failed=0
    for pid in "${BLOCK_PIDS[@]}"; do
        wait "$pid" || failed=1
    done
    BLOCK_PIDS=()
    if [ $failed -ne 0 ]; then
        echo "ERROR: some jobs in ${label} failed" >&2
        exit 1
    fi
}

should_run() {
    local label="$1"
    for c in $CONDITIONS; do
        [ "$c" = "$label" ] && return 0
    done
    return 1
}

# Spawns one (condition × all seeds) block.  All H-conditions share the
# base curriculum + CIFAR setup; the variable arguments at the end carry
# the single-knob override that defines the hypothesis.
spawn_h_block() {
    local label="$1"; shift
    local ablation_value="$1"; shift
    local family="$1"; shift           # bn | gn | buffer | nomom (wandb tag)
    local momentum="$1"; shift
    local tags_csv="${ABLATION_KEY},${label},${family}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            "${CIFAR_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "${CURRICULUM_OVERRIDES[@]}" \
            "training.momentum=${momentum}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label}"
}

# ──────────────────────────────────────────────────────────────────────────────
# H-series — CIFAR gap-anatomy sweep
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=================================================================="
echo "  CIFAR-10 gap-anatomy sweep (base = D5 adaptive, λ_min=0.0)"
echo "=================================================================="

if should_run H1; then
    echo "=== H1: D5 baseline, eval_freq=1, BN, buffer=1000, µ=${MOM_DEFAULT} ==="
    spawn_h_block H1 H1_baseline reference "${MOM_DEFAULT}"
fi

if should_run H2; then
    echo "=== H2: H1 + GroupNorm (norm_type=group) ==="
    spawn_h_block H2 H2_groupnorm gn "${MOM_DEFAULT}" \
        model.norm_type=group \
        model.norm_groups=32
fi

if should_run H3; then
    echo "=== H3: H1 + buffer 5000 (5× the default) ==="
    spawn_h_block H3 H3_buffer5k buffer "${MOM_DEFAULT}" \
        memory.total_budget=5000
fi

if should_run H4; then
    echo "=== H4: H1 + no momentum (µ=0.0) ==="
    spawn_h_block H4 H4_nomomentum nomom 0.0
fi

if should_run H5; then
    echo "=== H5: H1 + GroupNorm + buffer 5000 (combines H2 & H3) ==="
    spawn_h_block H5 H5_groupnorm_buffer5k gn_buffer "${MOM_DEFAULT}" \
        model.norm_type=group \
        model.norm_groups=32 \
        memory.total_budget=5000
fi

echo ""
echo "=== CIFAR gap-anatomy sweep complete ==="
