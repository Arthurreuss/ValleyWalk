#!/usr/bin/env bash
# scripts/run_C.sh
#
# Implements the λ-curriculum sweep, the λ_min refinement, and the momentum
# cross-cut over those conditions on two-task rot-MNIST (Chapter 4 §4.5–§4.6).
#
# This script is two-task rot-MNIST only: the five-task long-sequence block
# lives in scripts/run_L.sh (L-series), CIFAR-10 generalisation in
# scripts/run_G.sh (G-series).
#
# Conditions (rot-MNIST, applied on top of standard ER, buffer 1 k reservoir)
# --------------------------------------------------------------------------
#   C1  — Linear curriculum, N = 50                            RQ1
#   C2  — Linear curriculum, N = 100                           RQ1
#   C3  — Linear curriculum, N = 200                           RQ1
#   C4  — Adaptive curriculum (closed-loop EMA ratio)          RQ2
#   C5  — Adaptive + λ_min = 0.05                              RQ4
#   C6  — Adaptive + λ_min = 0.10                              RQ4
#   C7  — Adaptive + λ_min = 0.20                              RQ4
#   C8  — C7 + full 60 k buffer (opt-in probe; not in default set)
#
# Momentum cross (§4.6)
# ---------------------
# Each curriculum condition above is run twice — once at training.momentum=0.0
# (the plain-SGD baseline against which the envelope-theorem bound is
# stated), and once at training.momentum=0.9 (the momentum-on condition that
# tests the noise-suppression prediction of §3.6).  Momentum-on names get a
# "_M" suffix in ablation_value.
#
# The "momentum-alone, no curriculum" falsifier (RQ3) lives in
# scripts/run_D.sh as D1_M (vanilla ER at µ=0.9): if D1_M shows no
# depth reduction relative to D1, the prediction depth(no curr., µ=0.9) ≈
# depth(no curr., µ=0.0) is confirmed.  No separate condition is needed here.
#
# Total (default): 7 conditions (C1-C7) × 2 momentum × 5 seeds = 70 runs.
#
# Per-step instrumentation
# ------------------------
# Buffer-fidelity diagnostics (cos(g_replay, g_true), ‖g_replay‖/‖g_true‖)
# are NOT enabled by default for the curriculum sweep — they cost one extra
# forward+backward per step on a large past-task batch.  The decomposition
# already establishes the estimator-bias contributor; curriculum runs need
# only the gap-depth / gap-area / accuracy curves.  Enable on demand via
# GRAD_DIAG=on if a specific contrast needs the fidelity signal.
#
# Usage
# -----
#   bash scripts/run_C.sh                                # C1-C7, both momentum
#   CONDITIONS="C1 C4 C6" bash scripts/run_C.sh          # subset
#   CONDITIONS="C8" bash scripts/run_C.sh                # the opt-in probe
#   MOMENTUM_SET="off" bash scripts/run_C.sh             # µ=0 leg only
#   GRAD_DIAG=on CONDITIONS="C4 C7" bash scripts/run_C.sh
#   SEEDS="1,2,3,4,5" bash scripts/run_C.sh
#   N_JOBS=4     bash scripts/run_C.sh                   # parallelism cap
#   DRY_RUN=1    bash scripts/run_C.sh                   # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_CONDITIONS_DEFAULT="C1 C2 C3 C4 C5 C6 C7"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

# Enable per-step g_true buffer-fidelity diagnostics?  Off by default for the
# curriculum sweep (see header comment).
GRAD_DIAG="${GRAD_DIAG:-off}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Two-task rot-MNIST: a single, clean task transition.
ROTMNIST_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=2
    'dataset.rotations_deg=[0,90]'
)

EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=250
)

# Optional g_true diagnostics — applied only when GRAD_DIAG=on.
GRAD_DIAG_OVERRIDES=()
if [ "${GRAD_DIAG}" = "on" ]; then
    GRAD_DIAG_OVERRIDES=(eval.stability_gap.grad_diagnostics.enabled=true)
fi

ABLATION_KEY="curriculum"

# Curriculum hyperparameters.
EMA_ALPHA=0.05    # EMA smoothing for the adaptive schedule
LAMBDA_MIN_C5=0.05
LAMBDA_MIN_C6=0.10
LAMBDA_MIN_C7=0.20

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
echo "Grad diag:     ${GRAD_DIAG}"
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

# Returns the list of momentum values to iterate, based on MOMENTUM_SET.
momentum_values() {
    case "${MOMENTUM_SET}" in
        off)  echo "${MOM_OFF}" ;;
        on)   echo "${MOM_ON}"  ;;
        both) echo "${MOM_OFF} ${MOM_ON}" ;;
        *) echo "ERROR: unknown MOMENTUM_SET=${MOMENTUM_SET} (expected off|on|both)" >&2; exit 1 ;;
    esac
}

# Suffix for ablation_value reflecting the momentum leg.
mom_suffix() {
    local mom="$1"
    if [ "$mom" = "${MOM_ON}" ]; then echo "_M"; else echo ""; fi
}

# Generic ER condition launcher for the rot-MNIST block.
spawn_rotmnist_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    local family="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${family},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=er \
            "${ROTMNIST_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            ${GRAD_DIAG_OVERRIDES[@]:+"${GRAD_DIAG_OVERRIDES[@]}"} \
            "training.momentum=${mom}" \
            method.mode=standard \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom})"
}

# Wrappers for the two schedule families.
run_linear() {
    local label="$1"; shift
    local ramp_steps="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    echo "=== ${label}: std ER + λ-curriculum linear, N=${ramp_steps}, µ=${mom} ==="
    spawn_rotmnist_block "${label}" "${base_ablation}" "${mom}" linear \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${ramp_steps}" \
        method.lambda_curriculum.schedule=linear
}

run_adaptive() {
    local label="$1"; shift
    local lambda_min="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    echo "=== ${label}: std ER + λ-curriculum adaptive, λ_min=${lambda_min}, µ=${mom} ==="
    spawn_rotmnist_block "${label}" "${base_ablation}" "${mom}" adaptive \
        method.lambda_curriculum.enabled=true \
        method.lambda_curriculum.schedule=adaptive \
        "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}" \
        "method.lambda_curriculum.lambda_min=${lambda_min}" \
        "$@"
}

# ──────────────────────────────────────────────────────────────────────────────
# C-series — λ-curriculum sweep on standard ER (rot-MNIST, §4.5)
# ──────────────────────────────────────────────────────────────────────────────

for mom in $(momentum_values); do
    echo ""
    echo "=================================================================="
    echo "  Curriculum block (rot-MNIST) — training.momentum = ${mom}"
    echo "=================================================================="

    if should_run C1; then run_linear C1  50 C1_linear_N50  "${mom}"; fi
    if should_run C2; then run_linear C2 100 C2_linear_N100 "${mom}"; fi
    if should_run C3; then run_linear C3 200 C3_linear_N200 "${mom}"; fi

    if should_run C4; then run_adaptive C4 0.0                "C4_adaptive"           "${mom}"; fi
    if should_run C5; then run_adaptive C5 "${LAMBDA_MIN_C5}" "C5_adaptive_lmin0.05" "${mom}"; fi
    if should_run C6; then run_adaptive C6 "${LAMBDA_MIN_C6}" "C6_adaptive_lmin0.10" "${mom}"; fi
    if should_run C7; then run_adaptive C7 "${LAMBDA_MIN_C7}" "C7_adaptive_lmin0.20" "${mom}"; fi

    # C8 — variance-vs-acceleration probe (opt-in; not in the default set).
    # C7 curriculum (λ_min=0.20) with the *estimator* contributor removed at
    # source: full 60 k buffer + exact replay gradient (no per-step sampling
    # noise), mirroring D2/D4's replay setup. Standard mode (the curriculum
    # already tempers magnitude; balancing would confound). Run at µ=0 to ask
    # whether the tail momentum closes (C7 2.71 → C7_M 0.035) is killed by
    # variance reduction alone (→ tail collapses, momentum interchangeable) or
    # needs momentum's acceleration (→ tail stays ~2). Run: CONDITIONS="C8".
    if should_run C8; then
        run_adaptive C8 "${LAMBDA_MIN_C7}" "C8_adaptive_lmin0.20_fullbuf" "${mom}" \
            method.replay_full_buffer=true \
            memory.total_budget=60000
    fi
done

echo ""
echo "=== Curriculum sweep complete ==="
