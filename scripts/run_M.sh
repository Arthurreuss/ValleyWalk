#!/usr/bin/env bash
# scripts/run_M.sh
#
# M-series: global learning-rate warm-up (the step-size accountability block).
#
# Aim
# ---
# Determine how much of the boundary overshoot is attributable to the step
# size alone.  At a task switch the minimum teleports and the first discrete
# steps ride a large, unopposed new-task gradient out of the old basin; the
# acute driver of that overshoot is the step size meeting the boundary
# gradient (Level Two of the two-level account).  The cleanest step-size-only
# intervention is a global LR warm-up: scale the optimiser's lr by
# min((task_step − 1) / N, 1) at the start of every task with task_id > 0,
# slowing the new-task gradient AND the replay gradient equally — no change
# to the update direction, no reweighting of the mix.
#
# Two comparisons hang off this block:
#   vs D1 (run_D.sh)      — how much depth/area does the pure step-size
#                           intervention remove from the total gap?
#   vs C1–C3 (run_C.sh)   — falsification of the λ-curriculum: same ramp
#                           lengths N ∈ {50, 100, 200}, but symmetric.  If LR
#                           warm-up matches the curriculum's depth reduction
#                           at matched N, the curriculum's operative mechanism
#                           is "small steps early", not asymmetric scaling.
#                           If it falls short, the asymmetry (slowing only
#                           the new task, keeping the restoring force whole)
#                           matters per se.
#
# Everything lands in the same master_index.csv, so both comparisons are
# analysis-time joins — no coupling to the other scripts.
#
# Conditions (default set: M1 M2 M3)
# ----------------------------------
#   M1 — LR warm-up, N = 50     (matches C1's ramp length)
#   M2 — LR warm-up, N = 100    (matches C2)
#   M3 — LR warm-up, N = 200    (matches C3)
#
# All conditions: standard-mode ER, 1 k reservoir buffer — identical to D1 /
# the C-series base configuration except for lr_warmup.
#
# Momentum cross
# --------------
# Each condition runs at training.momentum=0.0 and 0.9 ("_M" suffix), like
# the C and D blocks.
#
# Total (default): 3 conditions × 2 momentum × 5 seeds = 30 runs.
# rot-MNIST + MLP runs in seconds; the block finishes in <15 min at N_JOBS=5.
#
# Usage
# -----
#   bash scripts/run_M.sh                          # default (M1 M2 M3)
#   CONDITIONS="M2" bash scripts/run_M.sh          # subset
#   MOMENTUM_SET="off" bash scripts/run_M.sh       # only the µ=0 leg
#   MOMENTUM_SET="on"  bash scripts/run_M.sh       # only the µ=0.9 leg
#   SEEDS="1,2,3,4,5"  bash scripts/run_M.sh       # custom seeds
#   N_JOBS=4           bash scripts/run_M.sh       # cap parallelism
#   DRY_RUN=1          bash scripts/run_M.sh       # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_CONDITIONS_DEFAULT="M1 M2 M3"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Two-task rot-MNIST: a single, clean task transition (matches run_D.sh).
DATASET_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=2
    'dataset.rotations_deg=[0,90]'
)

# Dense per-step eval over a 250-step window starting at the transition
# (matches run_D.sh so depth/area are directly comparable to D1).
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=250
)

ABLATION_KEY="lr_warmup"

# Momentum settings.
MOM_OFF=0.0
MOM_ON=0.9

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────────

if [ "$DRY_RUN" != "1" ] && [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:    ${GIT_COMMIT}"
echo "Python:        ${PYTHON}"
echo "Conditions:    ${CONDITIONS}"
echo "Momentum set:  ${MOMENTUM_SET}"
echo "Seeds:         ${SEEDS}"
echo "N_JOBS:        ${N_JOBS}"
echo "Dry run:       ${DRY_RUN}"
echo "---"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirrors run_D.sh)
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

momentum_values() {
    case "${MOMENTUM_SET}" in
        off)  echo "${MOM_OFF}" ;;
        on)   echo "${MOM_ON}"  ;;
        both) echo "${MOM_OFF} ${MOM_ON}" ;;
        *) echo "ERROR: unknown MOMENTUM_SET=${MOMENTUM_SET} (expected off|on|both)" >&2; exit 1 ;;
    esac
}

mom_suffix() {
    local mom="$1"
    if [ "$mom" = "${MOM_ON}" ]; then echo "_M"; else echo ""; fi
}

# LR warm-up condition launcher: standard-mode ER + lr_warmup at the given N.
run_warmup() {
    local label="$1"; shift
    local warmup_steps="$1"; shift
    local mom="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${label}_lrwarmup_N${warmup_steps}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${mom_tag}"
    echo "=== ${label}: std ER + global LR warm-up, N=${warmup_steps}, µ=${mom} ==="
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=er \
            "${DATASET_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            method.mode=standard \
            method.lr_warmup.enabled=true \
            "method.lr_warmup.warmup_steps=${warmup_steps}" \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]"
    done
    wait_block "${label} (µ=${mom})"
}

# ──────────────────────────────────────────────────────────────────────────────
# M-series — global LR warm-up
# ──────────────────────────────────────────────────────────────────────────────

for mu in $(momentum_values); do
    echo ""
    echo "=================================================================="
    echo "  LR warm-up block — training.momentum = ${mu}"
    echo "=================================================================="

    if should_run M1; then run_warmup M1  50 "${mu}"; fi
    if should_run M2; then run_warmup M2 100 "${mu}"; fi
    if should_run M3; then run_warmup M3 200 "${mu}"; fi
done

echo ""
echo "=== LR warm-up sweep complete ==="
