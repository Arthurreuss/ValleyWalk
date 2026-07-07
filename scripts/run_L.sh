#!/usr/bin/env bash
# scripts/run_L.sh
#
# L-series: two-gate long-sequence ablation on five-task rot-MNIST.
# (Split out of run_C.sh / run_curriculum.sh, which now covers the two-task
# C-series only.  The old L-series — vanilla / NCL / λ_min∈{0.0,0.10} — is
# replaced by the clean baseline-vs-both-gates ablation below.)
#
# Aim
# ---
# Ask whether the single-transition result — the two gates trace a
# stability--plasticity frontier over vanilla ER — carries past one task
# boundary.  Five-task rot-MNIST (rotations [0,30,60,90,120]) gives four
# consecutive transitions on the same MLP / online regime as the C-series.
#
# Conditions (ablation_key = "longseq")
# -------------------------------------
#   L1 — Vanilla ER                          no gate                 baseline
#   L2 — Adaptive curriculum, λ_min = 0.20   scalar feedback gate    (= C7 / G3 headline)
#   L3 — Asymmetric PER, δ = 0.1             directional feedforward gate
#
# All on standard ER over a 1 k reservoir buffer (config default).  L3 uses the
# rot-MNIST asym-PER settings: apply_to=current, fisher.target=replay,
# warm-started CG at 25 iters, δ = 0.1 (the strong single-δ rot-MNIST setting).
#
# Momentum cross
# --------------
# Each condition runs at both momentum legs — µ = 0.0 (plain SGD) and µ = 0.9;
# momentum-on runs carry a "_M" suffix in ablation_value.  The directional gate
# (L3) is strongest at µ = 0 and re-inflates under momentum, so both legs are
# needed to read the frontier on the long sequence.  Set MOMENTUM_SET=on to
# ship only the µ = 0.9 leg (the old L-series regime).
#
# Total (default): 3 conditions × 2 momentum × 5 seeds = 30 runs.  rot-MNIST +
# MLP; L3's CG makes it the slow condition, but the block still finishes fast.
#
# Usage
# -----
#   bash scripts/run_L.sh                         # L1 L2 L3, both momentum
#   CONDITIONS="L3" bash scripts/run_L.sh         # subset
#   MOMENTUM_SET="off" bash scripts/run_L.sh      # only the µ=0 leg
#   MOMENTUM_SET="on"  bash scripts/run_L.sh      # only the µ=0.9 leg
#   SEEDS="1,2,3,4,5"  bash scripts/run_L.sh      # custom seeds
#   N_JOBS=4           bash scripts/run_L.sh      # cap parallelism
#   DRY_RUN=1          bash scripts/run_L.sh      # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_CONDITIONS_DEFAULT="L1 L2 L3"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Five-task rot-MNIST: four consecutive transitions over a uniform 30° schedule.
ROTMNIST_5TASK_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=5
    'dataset.rotations_deg=[0,30,60,90,120]'
)

# Dense per-step eval over a 250-step window starting at each transition.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=250
)

ABLATION_KEY="longseq"

# Curriculum knobs — λ_min = 0.20 is the headline setting (rot-MNIST C7).
EMA_ALPHA=0.05
LAMBDA_MIN_L2=0.20

# Asymmetric PER knobs (mirror the rot-MNIST P-series): filter the current-task
# gradient only, through the replay Fisher, warm-started CG.  δ = 0.1.
PER_DAMPING=0.1
PER_CG_ITERS=25

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

# Generic 5-task launcher.  Loops over seeds; caller passes the method and any
# method-specific Hydra overrides as further arguments.
#   $1 label   $2 base_ablation   $3 method   $4 mom   $5 family   $6... overrides
spawn_lseq_block() {
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
    local tags_csv="${ABLATION_KEY},${label},${family},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            "method=${method_name}" \
            "${ROTMNIST_5TASK_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom}, 5-task)"
}

# ──────────────────────────────────────────────────────────────────────────────
# L-series — five-task rot-MNIST, baseline vs both gates
# ──────────────────────────────────────────────────────────────────────────────

for mom in $(momentum_values); do
    echo ""
    echo "=================================================================="
    echo "  Long-sequence block (rot-MNIST, 5 tasks) — training.momentum = ${mom}"
    echo "=================================================================="

    if should_run L1; then
        echo "=== L1: vanilla ER (no gate) on 5-task rot-MNIST, µ=${mom} ==="
        spawn_lseq_block L1 L1_vanilla_ER er "${mom}" vanilla \
            method.mode=standard
    fi

    if should_run L2; then
        echo "=== L2: adaptive curriculum (λ_min=0.20) on 5-task rot-MNIST, µ=${mom} ==="
        spawn_lseq_block L2 L2_adaptive_lmin0.20 er "${mom}" adaptive \
            method.mode=standard \
            method.lambda_curriculum.enabled=true \
            method.lambda_curriculum.schedule=adaptive \
            "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}" \
            "method.lambda_curriculum.lambda_min=${LAMBDA_MIN_L2}"
    fi

    if should_run L3; then
        echo "=== L3: asymmetric PER (δ=0.1) on 5-task rot-MNIST, µ=${mom} ==="
        spawn_lseq_block L3 L3_PER_asym_d0.1 precond_er "${mom}" per_asym \
            method.apply_to=current \
            method.fisher.target=replay \
            "method.fisher.damping=${PER_DAMPING}" \
            "method.cg.iters=${PER_CG_ITERS}" \
            method.cg.warm_start=true
    fi
done

echo ""
echo "=== Long-sequence (L-series) sweep complete ==="
