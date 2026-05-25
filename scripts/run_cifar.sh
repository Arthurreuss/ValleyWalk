#!/usr/bin/env bash
# scripts/run_cifar.sh
#
# Unified CIFAR-10 sweep — replaces the old D-block in run_curriculum.sh and
# the H-block in run_cifar_gap_anatomy.sh.  All new CIFAR work happens here.
#
# Two opt-in blocks
# -----------------
#   headline        — the final 6 conditions for the §5.6 / §6.5 tables
#                     (3 methods × {BN, GN}, all at buffer 5000, µ = 0.9).
#   generalization  — does the headline finding carry to other corruptions
#                     and to 3-task sequences?  Vanilla-ER controls included
#                     so the comparison stays interpretable on the new shifts.
#
# Headline block (ablation_key = "cifar_headline")
# ------------------------------------------------
# All at: dom_cifar10, 2 tasks [none, gaussian_noise], ResNet-18,
#         10 epochs/task, µ = 0.9, eval_freq_steps = 1, window_steps = 250.
#
#   Label  Method                            Norm   Buffer
#   D1     Vanilla ER                        BN     5000
#   D2     NCL (α = 0.1, damp = 1e-3)        BN     n/a    ← NCL has no buffer
#   D3     Adaptive curr.  λ_min = 0.20      BN     5000
#   D4     Vanilla ER                        GN     5000
#   D5     NCL (α = 0.1, damp = 1e-3)        GN     n/a
#   D6     Adaptive curr.  λ_min = 0.20      GN     5000
#
# λ_min = 0.20 is the headline curriculum setting (rot-MNIST C7 / old D3)
# after the multi-seed CIFAR sweep showed it gave the best final accuracies.
#
# NCL's "buffer" lines are silently identical between D2 and any buffer
# setting — NCL uses a precision-prior path-finding mechanism, not replay.
# Listed at "n/a" for clarity; the actual run is the same code path.
#
# Generalization block (ablation_key = "cifar_generalization")
# ------------------------------------------------------------
# Each condition runs the SHIP config (BN + buffer 5k + adaptive curriculum,
# λ_min = 0.20, µ = 0.9) plus a paired vanilla-ER control on the same shift.
# The control answers "does our method still beat vanilla on this new shift?"
#
#   Label  Setup                                              Method
#   S1     2 tasks [none, shot_noise]                         adaptive (D3 config)
#   S1v    2 tasks [none, shot_noise]                         vanilla ER (D1 config)
#   S2     2 tasks [none, contrast]                           adaptive (D3 config)
#   S2v    2 tasks [none, contrast]                           vanilla ER (D1 config)
#   T1     3 tasks [none, gaussian_noise, shot_noise]         adaptive (D3 config)
#   T1v    3 tasks [none, gaussian_noise, shot_noise]         vanilla ER (D1 config)
#
# If the headline block reveals GN beats BN, swap SHIP_NORM=gn and rerun the
# generalization block; the env var changes only the normalisation on the
# six gen conditions, nothing else.
#
# Total runs (when both blocks selected with default seed counts)
# ---------------------------------------------------------------
#   headline:        6 × 5 seeds = 30
#   generalization:  6 × 3 seeds = 18
#   Grand total                  = 48  (≈ 58 h on MPS)
#
# Usage
# -----
# BLOCKS is required — every block is opt-in.  Valid block names:
#   headline | generalization
#
#   BLOCKS="headline"                      bash scripts/run_cifar.sh
#   BLOCKS="generalization"                bash scripts/run_cifar.sh
#   BLOCKS="headline generalization"       bash scripts/run_cifar.sh
#   CONDITIONS="D3 D6" BLOCKS="headline"   bash scripts/run_cifar.sh   # subset
#   SEEDS_HEADLINE="1,2,3" BLOCKS="headline" bash scripts/run_cifar.sh
#   SEEDS_GEN="1,2,3,4,5" BLOCKS="generalization" bash scripts/run_cifar.sh
#   SHIP_NORM=gn BLOCKS="generalization"   bash scripts/run_cifar.sh
#   DRY_RUN=1 BLOCKS="headline" bash scripts/run_cifar.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# CIFAR runs on a single MPS GPU — default sequential.
N_JOBS="${N_JOBS:-1}"

# Two seed sets — headline runs at 5 seeds (paired Wilcoxon-able), generalization
# at 3 (point-estimate "does it carry over").
SEEDS_HEADLINE="${SEEDS_HEADLINE:-1,2,3,4,5}"
SEEDS_GEN="${SEEDS_GEN:-1,2,3}"

ALL_HEADLINE_CONDITIONS="D1 D2 D3 D4 D5 D6"
ALL_GEN_CONDITIONS="S1 S1v S2 S2v T1 T1v"
ALL_CONDITIONS_DEFAULT="${ALL_HEADLINE_CONDITIONS} ${ALL_GEN_CONDITIONS}"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

BLOCKS="${BLOCKS:-}"
VALID_BLOCKS=(headline generalization)

# Which norm to ship in the generalization block.  Default "bn" — set to
# "gn" after the headline block if GroupNorm wins.
SHIP_NORM="${SHIP_NORM:-gn}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ──────────────────────────────────────────────────────────────────────────────
# Fixed config (shared across every CIFAR condition)
# ──────────────────────────────────────────────────────────────────────────────

CIFAR_BASE=(
    dataset=dom_cifar10
    model=resnet18
    training.epochs_per_task=10
    training.momentum=0.9
)

# Fine-grained eval at every step over a 250-step window — captures the full
# recovery curve from the BN/GN catch-up time-constant.  Outside the window,
# fall back to every-10-step general eval so plots remain continuous.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=250
    eval.eval_every_n_steps=10
)

BUFFER_BIG=5000

# Adaptive λ-curriculum knobs — λ_min = 0.20 is the headline setting from
# the CIFAR sweep (D3 in the old numbering).
ADAPTIVE_CURRICULUM=(
    method=er
    method.mode=standard
    method.lambda_curriculum.enabled=true
    method.lambda_curriculum.schedule=adaptive
    method.lambda_curriculum.ema_alpha=0.05
    method.lambda_curriculum.lambda_min=0.20
)

VANILLA_ER=(
    method=er
    method.mode=standard
)

NCL_CONFIG=(
    method=ncl
    method.ncl.damping=0.001
    method.ncl.fisher_samples=1000
    method.ncl.prior_init=0.1
    method.ncl.trust_radius=1.0
)

GN_OVERRIDES=(
    model.norm_type=group
    model.norm_groups=32
)

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "$BLOCKS" ]; then
    echo "ERROR: BLOCKS environment variable not set — every block is opt-in." >&2
    echo "Usage: BLOCKS=\"<block> [<block> ...]\" bash scripts/run_cifar.sh" >&2
    echo "       Valid blocks: ${VALID_BLOCKS[*]}" >&2
    exit 1
fi

for _b in $BLOCKS; do
    _ok=0
    for _v in "${VALID_BLOCKS[@]}"; do
        [ "$_b" = "$_v" ] && { _ok=1; break; }
    done
    if [ "$_ok" = "0" ]; then
        echo "ERROR: unknown block '${_b}'. Valid: ${VALID_BLOCKS[*]}" >&2
        exit 1
    fi
done

if [ "$SHIP_NORM" != "bn" ] && [ "$SHIP_NORM" != "gn" ]; then
    echo "ERROR: SHIP_NORM='${SHIP_NORM}' (expected 'bn' or 'gn')" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:       ${GIT_COMMIT}"
echo "Python:           ${PYTHON}"
echo "Blocks:           ${BLOCKS}"
echo "Conditions:       ${CONDITIONS}"
echo "Seeds (headline): ${SEEDS_HEADLINE}"
echo "Seeds (gen):      ${SEEDS_GEN}"
echo "Ship norm (gen):  ${SHIP_NORM}"
echo "N_JOBS:           ${N_JOBS}"
echo "Dry run:          ${DRY_RUN}"
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

has_block() {
    local target="$1"
    for b in $BLOCKS; do
        [ "$b" = "$target" ] && return 0
    done
    return 1
}

# spawn_condition — single condition × all listed seeds.
#   $1: label                ($1=D1, D2, …, S1, …)
#   $2: ablation_value       (descriptive)
#   $3: family tag           (vanilla | ncl | adaptive)
#   $4: ablation_key         (cifar_headline | cifar_generalization)
#   $5: seeds CSV
#   $6...: hydra overrides specific to this condition
spawn_condition() {
    local label="$1"; shift
    local ablation_value="$1"; shift
    local family="$1"; shift
    local ablation_key="$1"; shift
    local seeds_csv="$1"; shift
    local tags_csv="${ablation_key},${label},${family}"
    local -a seed_array
    IFS=',' read -ra seed_array <<< "$seeds_csv"
    for seed in "${seed_array[@]}"; do
        spawn_job \
            "${CIFAR_BASE[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "seed=${seed}" \
            "+ablation_key=${ablation_key}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Block A — headline (ablation_key=cifar_headline)
# ──────────────────────────────────────────────────────────────────────────────

if has_block headline; then
    echo ""
    echo "=================================================================="
    echo "  CIFAR headline — 3 methods × {BN, GN}, buffer 5000, µ=0.9"
    echo "=================================================================="

    TWO_TASK_GAUSSIAN=(
        dataset.num_tasks=2
        'dataset.corruption_types=[none,gaussian_noise]'
    )

    if should_run D1; then
        echo "=== D1: vanilla ER, BN, buffer 5k ==="
        spawn_condition D1 D1_vanilla_BN_buf5k vanilla cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${VANILLA_ER[@]}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run D2; then
        echo "=== D2: NCL, BN  (buffer setting irrelevant — NCL has no replay) ==="
        spawn_condition D2 D2_NCL_BN ncl cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${NCL_CONFIG[@]}"
    fi

    if should_run D3; then
        echo "=== D3: adaptive curriculum (λ_min=0.20), BN, buffer 5k ==="
        spawn_condition D3 D3_adaptive_lmin0.20_BN_buf5k adaptive cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${ADAPTIVE_CURRICULUM[@]}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run D4; then
        echo "=== D4: vanilla ER, GN, buffer 5k ==="
        spawn_condition D4 D4_vanilla_GN_buf5k vanilla cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${VANILLA_ER[@]}" "${GN_OVERRIDES[@]}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run D5; then
        echo "=== D5: NCL, GN ==="
        spawn_condition D5 D5_NCL_GN ncl cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${NCL_CONFIG[@]}" "${GN_OVERRIDES[@]}"
    fi

    if should_run D6; then
        echo "=== D6: adaptive curriculum (λ_min=0.20), GN, buffer 5k ==="
        spawn_condition D6 D6_adaptive_lmin0.20_GN_buf5k adaptive cifar_headline "$SEEDS_HEADLINE" \
            "${TWO_TASK_GAUSSIAN[@]}" "${ADAPTIVE_CURRICULUM[@]}" "${GN_OVERRIDES[@]}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Block B — generalization (ablation_key=cifar_generalization)
# ──────────────────────────────────────────────────────────────────────────────
#
# Each S*/T* condition pairs the ship config (adaptive curriculum + buffer 5k
# + SHIP_NORM-controlled norm) with a vanilla-ER control on the same shift.

if has_block generalization; then
    echo ""
    echo "=================================================================="
    echo "  CIFAR generalization — ship config (norm=${SHIP_NORM}) vs vanilla controls"
    echo "=================================================================="

    # Build the norm overrides list once (empty for BN, GN flags for GN).
    SHIP_NORM_OVERRIDES=()
    if [ "$SHIP_NORM" = "gn" ]; then SHIP_NORM_OVERRIDES=("${GN_OVERRIDES[@]}"); fi

    TWO_TASK_SHOT=(
        dataset.num_tasks=2
        'dataset.corruption_types=[none,shot_noise]'
    )
    TWO_TASK_CONTRAST=(
        dataset.num_tasks=2
        'dataset.corruption_types=[none,contrast]'
    )
    THREE_TASK=(
        dataset.num_tasks=3
        'dataset.corruption_types=[none,gaussian_noise,shot_noise]'
    )

    if should_run S1; then
        echo "=== S1: ship config on 2-task [none, shot_noise] ==="
        spawn_condition S1 "S1_ship_shot_noise_${SHIP_NORM}" adaptive cifar_generalization "$SEEDS_GEN" \
            "${TWO_TASK_SHOT[@]}" "${ADAPTIVE_CURRICULUM[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi
    if should_run S1v; then
        echo "=== S1v: vanilla-ER control on 2-task [none, shot_noise] ==="
        spawn_condition S1v "S1v_vanilla_shot_noise_${SHIP_NORM}" vanilla cifar_generalization "$SEEDS_GEN" \
            "${TWO_TASK_SHOT[@]}" "${VANILLA_ER[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run S2; then
        echo "=== S2: ship config on 2-task [none, contrast] ==="
        spawn_condition S2 "S2_ship_contrast_${SHIP_NORM}" adaptive cifar_generalization "$SEEDS_GEN" \
            "${TWO_TASK_CONTRAST[@]}" "${ADAPTIVE_CURRICULUM[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi
    if should_run S2v; then
        echo "=== S2v: vanilla-ER control on 2-task [none, contrast] ==="
        spawn_condition S2v "S2v_vanilla_contrast_${SHIP_NORM}" vanilla cifar_generalization "$SEEDS_GEN" \
            "${TWO_TASK_CONTRAST[@]}" "${VANILLA_ER[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run T1; then
        echo "=== T1: ship config on 3-task [none, gaussian_noise, shot_noise] ==="
        spawn_condition T1 "T1_ship_3task_${SHIP_NORM}" adaptive cifar_generalization "$SEEDS_GEN" \
            "${THREE_TASK[@]}" "${ADAPTIVE_CURRICULUM[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi
    if should_run T1v; then
        echo "=== T1v: vanilla-ER control on 3-task [none, gaussian_noise, shot_noise] ==="
        spawn_condition T1v "T1v_vanilla_3task_${SHIP_NORM}" vanilla cifar_generalization "$SEEDS_GEN" \
            "${THREE_TASK[@]}" "${VANILLA_ER[@]}" \
            "${SHIP_NORM_OVERRIDES[@]:+${SHIP_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi
fi

echo ""
echo "=== CIFAR sweep complete ==="
