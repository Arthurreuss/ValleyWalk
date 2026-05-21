#!/usr/bin/env bash
# scripts/run_curriculum.sh
#
# Implements the λ-curriculum sweep, the λ_min refinement, and the momentum
# cross-cut over those conditions on rot-MNIST (Chapter 4 §4.5–§4.6), plus
# the five-task rot-MNIST long-sequence block (§5.7).
#
# CIFAR-10 generalisation lives in scripts/run_cifar.sh (the unified CIFAR
# sweep — headline + generalization blocks).  This script is rot-MNIST only.
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
# scripts/run_decomposition.sh as G1_M (vanilla ER at µ=0.9): if G1_M shows no
# depth reduction relative to G1, the prediction depth(no curr., µ=0.9) ≈
# depth(no curr., µ=0.0) is confirmed.  No separate condition is needed here.
#
# Long-sequence rot-MNIST (§5.7, TODO 1.2)
# ----------------------------------------
# Five-task rot-MNIST (rotations [0,30,60,90,120]) to test whether the
# curriculum carries past a single transition, and whether NCL recovers its
# advantage at the task counts where Kao et al. (2021) report SOTA.
#   L1 — Vanilla ER (no curriculum)
#   L2 — Standard NCL (α = prior_init = 0.1)
#   L3 — Adaptive curriculum, λ_min = 0.00
#   L4 — Adaptive curriculum, λ_min = 0.10
# Single momentum setting (MOM_LSEQ, default 0.9) — the C-series and the
# G1/G1_M decomposition already establish that momentum-on dominates µ=0 on
# stability-gap depth, so running both legs here would burn 20 extra runs
# without addressing the long-sequence research question.
#
# Total runs (when both blocks selected)
# --------------------------------------
#   rot-MNIST 2-task (C1-C7): 7 conditions × 2 momentum × 5 seeds = 70
#   rot-MNIST 5-task (L1-L4): 4 conditions × 1 momentum × 5 seeds = 20
#   ──────────────────────────────────────────────────────────────────
#   Grand total                                                  = 90
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
# BLOCKS is required — every block is opt-in.  Valid block names:
#   rot_mnist          — C-series, two-task rot-MNIST
#   rot_mnist_5task    — L-series, five-task rot-MNIST (long-sequence)
#
#   BLOCKS="rot_mnist" bash scripts/run_curriculum.sh                       # C-series only
#   BLOCKS="rot_mnist_5task" bash scripts/run_curriculum.sh                 # L-series only
#   BLOCKS="rot_mnist rot_mnist_5task" bash scripts/run_curriculum.sh       # both
#   CONDITIONS="C1 C4 C6" BLOCKS="rot_mnist" bash scripts/run_curriculum.sh # subset
#   CONDITIONS="L3 L4" BLOCKS="rot_mnist_5task" bash scripts/run_curriculum.sh
#   MOMENTUM_SET="off" BLOCKS="rot_mnist" bash scripts/run_curriculum.sh    # µ=0 leg only
#   GRAD_DIAG=on CONDITIONS="C4 C7" BLOCKS="rot_mnist" bash scripts/run_curriculum.sh
#   SEEDS="1,2,3,4,5" BLOCKS="rot_mnist" bash scripts/run_curriculum.sh
#   N_JOBS=4     BLOCKS="rot_mnist" bash scripts/run_curriculum.sh          # parallelism cap
#   DRY_RUN=1    BLOCKS="rot_mnist" bash scripts/run_curriculum.sh          # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_C_CONDITIONS="C1 C2 C3 C4 C5 C6 C7"
ALL_L_CONDITIONS="L1 L2 L3 L4"
ALL_CONDITIONS_DEFAULT="${ALL_C_CONDITIONS} ${ALL_L_CONDITIONS}"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

# BLOCKS: space-separated list of dataset blocks to run.  Every block is
# opt-in — no default.  Valid values: rot_mnist | rot_mnist_5task
BLOCKS="${BLOCKS:-}"
VALID_BLOCKS=(rot_mnist rot_mnist_5task)

# Momentum used for the 5-task long-sequence block.  Default 0.9 — the
# C-series already characterises the µ=0 vs. µ=0.9 trade-off at two tasks,
# so we ship the headline configuration here instead of running both legs.
MOM_LSEQ="${MOM_LSEQ:-0.9}"

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

# Five-task rot-MNIST (§5.7, TODO 1.2): four consecutive transitions over a
# uniform 30°-step rotation schedule.  Same MLP backbone / online regime as
# the two-task block.
ROTMNIST_5TASK_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=5
    'dataset.rotations_deg=[0,30,60,90,120]'
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

if [ -z "$BLOCKS" ]; then
    echo "ERROR: BLOCKS environment variable not set — every block is opt-in." >&2
    echo "Usage: BLOCKS=\"<block> [<block> ...]\" bash scripts/run_curriculum.sh" >&2
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

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:    ${GIT_COMMIT}"
echo "Python:        ${PYTHON}"
echo "Blocks:        ${BLOCKS}"
echo "Conditions:    ${CONDITIONS}"
echo "Momentum set:  ${MOMENTUM_SET}"
echo "MOM_LSEQ:      ${MOM_LSEQ}"
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

has_block() {
    local target="$1"
    for b in $BLOCKS; do
        [ "$b" = "$target" ] && return 0
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
        "method.lambda_curriculum.lambda_min=${lambda_min}"
}

# ──────────────────────────────────────────────────────────────────────────────
# C-series — λ-curriculum sweep on standard ER (rot-MNIST, §4.5)
# ──────────────────────────────────────────────────────────────────────────────

if has_block rot_mnist; then
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
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# L-series — five-task rot-MNIST long-sequence sweep (§5.7, TODO 1.2)
# ──────────────────────────────────────────────────────────────────────────────
#
# Same MLP / online regime as the C-series, but with four consecutive task
# transitions over a uniform 30° rotation schedule.  Tests (i) whether the
# adaptive curriculum carries past a single transition and (ii) whether NCL
# recovers its rot-MNIST advantage at the task counts where Kao et al.
# (2021) report SOTA.

spawn_rotmnist_5task_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local method_name="$1"; shift
    local mom="$1"; shift
    local family="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    # Tag with the plain ABLATION_KEY ("curriculum") so the L-series groups
    # with the C-series in wandb; the ablation_key carries the "_5task"
    # suffix so per-block grouping stays distinct.
    local tags_csv="${ABLATION_KEY},${label},${family},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            "method=${method_name}" \
            "${ROTMNIST_5TASK_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            ${GRAD_DIAG_OVERRIDES[@]:+"${GRAD_DIAG_OVERRIDES[@]}"} \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}_5task" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom}, 5-task)"
}

if has_block rot_mnist_5task; then
    mom="${MOM_LSEQ}"
    echo ""
    echo "=================================================================="
    echo "  Long-sequence block (rot-MNIST, 5 tasks) — training.momentum = ${mom}"
    echo "=================================================================="

    if should_run L1; then
        echo "=== L1: vanilla ER on 5-task rot-MNIST, µ=${mom} ==="
        spawn_rotmnist_5task_block L1 L1_vanilla_ER er "${mom}" vanilla \
            method.mode=standard
    fi

    if should_run L2; then
        echo "=== L2: standard NCL (α=0.1) on 5-task rot-MNIST, µ=${mom} ==="
        # prior_init=0.1 is the tuned winner from the rot-MNIST α sweep
        # (thesis_draft/notes/ncl_implementation_findings.md §2.1 +
        # outputs/_probe/ncl_sweep_20260511_201851): α=1.0 over-regularises,
        # α≤0.03 diverges at lr=0.1, α=0.1 wins on ACC/FORG and gap-depth.
        spawn_rotmnist_5task_block L2 L2_NCL ncl "${mom}" ncl \
            method.ncl.damping=0.001 \
            method.ncl.fisher_samples=1000 \
            method.ncl.prior_init=0.1 \
            method.ncl.trust_radius=1.0
    fi

    if should_run L3; then
        echo "=== L3: adaptive curriculum (λ_min=0.0) on 5-task rot-MNIST, µ=${mom} ==="
        spawn_rotmnist_5task_block L3 L3_adaptive_lmin0.00 er "${mom}" adaptive \
            method.mode=standard \
            method.lambda_curriculum.enabled=true \
            method.lambda_curriculum.schedule=adaptive \
            "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}" \
            method.lambda_curriculum.lambda_min=0.0
    fi

    if should_run L4; then
        echo "=== L4: adaptive curriculum (λ_min=0.10) on 5-task rot-MNIST, µ=${mom} ==="
        spawn_rotmnist_5task_block L4 L4_adaptive_lmin0.10 er "${mom}" adaptive \
            method.mode=standard \
            method.lambda_curriculum.enabled=true \
            method.lambda_curriculum.schedule=adaptive \
            "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}" \
            method.lambda_curriculum.lambda_min=0.10
    fi
fi

echo ""
echo "=== Curriculum sweep complete ==="
