#!/usr/bin/env bash
# scripts/run_curriculum.sh
#
# Implements the λ-curriculum sweep, the λ_min refinement, the momentum cross-
# cut over those conditions (Chapter 4 §4.5–§4.6), and the CIFAR-10
# generalisation block (§4.7).  Supersedes the old scripts/run_gap_anatomy.sh
# (which is now stale and should be ignored).
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
# CIFAR-10 generalisation (§4.7)
# ------------------------------
# Headline carry-over only — not the full sweep.  Four conditions:
#   D1 — Vanilla ER (ResNet-18, dom_cifar10)
#   D2 — Standard NCL
#   D3 — Best linear curriculum (defaults to N = 200; override via BEST_N env)
#   D4 — Best adaptive curriculum
# The CIFAR block uses the default ConvNet / ResNet-18 backbone via
# configs/model/resnet18.yaml.  Momentum cross is not run here — pick the
# better-performing momentum setting from the rot-MNIST sweep and lock it
# via MOM_CIFAR (default 0.0).
#
# Total runs
# ----------
#   rot-MNIST:  7 conditions × 2 momentum × 5 seeds = 70
#   CIFAR-10:   4 conditions × 1 momentum × 5 seeds = 20
#   ────────────────────────────────────────────────────
#   Grand total                                      = 90
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
#   bash scripts/run_curriculum.sh                            # rot-MNIST + CIFAR
#   BLOCK="rot_mnist" bash scripts/run_curriculum.sh          # rot-MNIST only
#   BLOCK="cifar10"   bash scripts/run_curriculum.sh          # CIFAR only
#   CONDITIONS="C1 C4 C6" bash scripts/run_curriculum.sh      # subset
#   MOMENTUM_SET="off" bash scripts/run_curriculum.sh         # only the µ=0 leg
#   GRAD_DIAG=on bash scripts/run_curriculum.sh               # enable g_true diag
#   BEST_N=100   bash scripts/run_curriculum.sh               # different best-N
#   SEEDS="1,2,3,4,5" bash scripts/run_curriculum.sh          # custom seeds
#   N_JOBS=4     bash scripts/run_curriculum.sh               # cap parallelism
#   DRY_RUN=1    bash scripts/run_curriculum.sh               # preview only

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

# BLOCK: which dataset block to run.  "rot_mnist" | "cifar10" | "both"
BLOCK="${BLOCK:-both}"

# Headline N for the "best linear curriculum" CIFAR condition.  Override
# after the rot-MNIST sweep tells us which N actually wins.
BEST_N="${BEST_N:-200}"

# Momentum used for the CIFAR block.  Default 0.9 — CIFAR is the
# generalisation block, where we ship the headline configuration (curriculum
# + momentum) rather than the no-momentum reference.  Override via env var
# if a different setting is needed.
MOM_CIFAR="${MOM_CIFAR:-0.9}"

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

# CIFAR overrides — 3-task dom_cifar10 with the ResNet-18 backbone.  Number
# of tasks is taken from configs/dataset/dom_cifar10.yaml (currently 3).
CIFAR_OVERRIDES=(
    dataset=dom_cifar10
    model=resnet18
)

# Dense per-step eval over a 500-step window starting at the transition.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=500
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

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:    ${GIT_COMMIT}"
echo "Python:        ${PYTHON}"
echo "Block:         ${BLOCK}"
echo "Conditions:    ${CONDITIONS}"
echo "Momentum set:  ${MOMENTUM_SET}"
echo "Best-N:        ${BEST_N}  (CIFAR headline linear condition)"
echo "MOM_CIFAR:      ${MOM_CIFAR}"
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
    local mu="$1"; shift
    local family="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mu" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${family},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=er \
            "${ROTMNIST_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "${GRAD_DIAG_OVERRIDES[@]}" \
            "training.momentum=${mu}" \
            method.mode=standard \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mu})"
}

# Wrappers for the two schedule families.
run_linear() {
    local label="$1"; shift
    local ramp_steps="$1"; shift
    local base_ablation="$1"; shift
    local mu="$1"; shift
    echo "=== ${label}: std ER + λ-curriculum linear, N=${ramp_steps}, µ=${mu} ==="
    spawn_rotmnist_block "${label}" "${base_ablation}" "${mu}" linear \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${ramp_steps}" \
        method.lambda_curriculum.schedule=linear
}

run_adaptive() {
    local label="$1"; shift
    local lambda_min="$1"; shift
    local base_ablation="$1"; shift
    local mu="$1"; shift
    echo "=== ${label}: std ER + λ-curriculum adaptive, λ_min=${lambda_min}, µ=${mu} ==="
    spawn_rotmnist_block "${label}" "${base_ablation}" "${mu}" adaptive \
        method.lambda_curriculum.enabled=true \
        method.lambda_curriculum.schedule=adaptive \
        "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}" \
        "method.lambda_curriculum.lambda_min=${lambda_min}"
}

# ──────────────────────────────────────────────────────────────────────────────
# C-series — λ-curriculum sweep on standard ER (rot-MNIST, §4.5)
# ──────────────────────────────────────────────────────────────────────────────

if [ "$BLOCK" = "rot_mnist" ] || [ "$BLOCK" = "both" ]; then
    for mu in $(momentum_values); do
        echo ""
        echo "=================================================================="
        echo "  Curriculum block (rot-MNIST) — training.momentum = ${mu}"
        echo "=================================================================="

        if should_run C1; then run_linear C1  50 C1_linear_N50  "${mu}"; fi
        if should_run C2; then run_linear C2 100 C2_linear_N100 "${mu}"; fi
        if should_run C3; then run_linear C3 200 C3_linear_N200 "${mu}"; fi

        if should_run C4; then run_adaptive C4 0.0                "C4_adaptive"           "${mu}"; fi
        if should_run C5; then run_adaptive C5 "${LAMBDA_MIN_C5}" "C5_adaptive_lmin0.05" "${mu}"; fi
        if should_run C6; then run_adaptive C6 "${LAMBDA_MIN_C6}" "C6_adaptive_lmin0.10" "${mu}"; fi
        if should_run C7; then run_adaptive C7 "${LAMBDA_MIN_C7}" "C7_adaptive_lmin0.20" "${mu}"; fi
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# D-series — CIFAR-10 generalisation (§4.7)
# ──────────────────────────────────────────────────────────────────────────────
#
# Single momentum setting (MOM_CIFAR).  No further momentum cross.  Best-N
# defaults to 200 — override after the rot-MNIST sweep tells us better.

spawn_cifar_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local method_name="$1"; shift
    local family="$1"; shift
    local mom_tag
    if [ "${MOM_CIFAR}" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    # Use plain ABLATION_KEY ("curriculum") as a tag so CIFAR runs group with
    # the rot-MNIST C-series; the wandb `group` field still uses the
    # _cifar-suffixed ablation_key, which keeps the per-block grouping intact.
    local tags_csv="${ABLATION_KEY},${label},${family},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            "method=${method_name}" \
            "${CIFAR_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "training.momentum=${MOM_CIFAR}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}_cifar" \
            "+ablation_value=${base_ablation}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (CIFAR)"
}

if [ "$BLOCK" = "cifar10" ] || [ "$BLOCK" = "both" ]; then
    echo ""
    echo "=================================================================="
    echo "  CIFAR-10 generalisation block — training.momentum = ${MOM_CIFAR}"
    echo "=================================================================="

    echo "=== D1: vanilla ER on dom_cifar10 ==="
    spawn_cifar_block D1 D1_vanilla_ER er vanilla method.mode=standard

    echo "=== D2: standard NCL on dom_cifar10 ==="
    # prior_init=0.1 is the tuned winner from the rot-MNIST α sweep
    # (thesis_draft/notes/ncl_implementation_findings.md §2.1 + iteration 2 in
    # outputs/_probe/ncl_sweep_20260511_201851): α=1.0 is over-regularising,
    # α≤0.03 diverges at lr=0.1, α=0.1 wins on ACC/FORG and on gap-depth.
    spawn_cifar_block D2 D2_NCL ncl ncl method.ncl.prior_init=0.1

    echo "=== D3: best linear curriculum (N=${BEST_N}) on dom_cifar10 ==="
    spawn_cifar_block D3 "D3_linear_N${BEST_N}" er linear \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${BEST_N}" \
        method.lambda_curriculum.schedule=linear

    echo "=== D4: adaptive curriculum on dom_cifar10 ==="
    spawn_cifar_block D4 D4_adaptive er adaptive \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        method.lambda_curriculum.schedule=adaptive \
        "method.lambda_curriculum.ema_alpha=${EMA_ALPHA}"
fi

echo ""
echo "=== Curriculum sweep complete ==="
