#!/usr/bin/env bash
# scripts/run_P.sh
#
# P-series: damping sweep for asymmetric preconditioned ER (PER_ASYM).
# (Formerly run_per_asym_sweep.sh.)
#
# Aim
# ---
# In the asymmetric update  d = δ(F_rep + δI)⁻¹ g_cur + g_rep  the damping δ is
# a *pure interference knob*: it filters only the current-task gradient and
# never touches the restorative replay gradient or the overall step scale.
# The sweep therefore interpolates the method between its two limits:
#
#   δ → ∞   plain ER            (g_cur passes through unfiltered)
#   δ → 0   valley walk         (g_cur projected onto the replay-Fisher null
#                                space; motion only in task-A-flat directions,
#                                full-strength restoration)
#
# The decomposition sweep only probed δ=1, which leaves every direction with
# λ ≲ 1 essentially unfiltered.  This sweep turns the filter up and asks the
# actual question: does the task-0 dip approach the D4 trajectory floor as
# δ → 0, and at what plasticity (WP) cost — or does interference through the
# Fisher's blind spots put a floor under the dip?
#
# Prediction: stability-gap depth falls monotonically with δ while WP10/WP100
# degrade.  The money plot is depth (and area) vs δ against the D4 reference.
#
# Conditions
# ----------
# Cross of DAMPINGS × buffer legs:
#   std     — 1 k reservoir buffer (matches the decomposition PER conditions).
#             The raw g_rep carries buffer sampling noise, which inflates
#             per-seed depth minima (see PER_asym vs PER_replay at µ=0).
#   fullbuf — replay gradient over the full 60 k buffer (zero estimator noise,
#             mirrors D2/D4).  The Fisher metric stays on a sampled subset
#             (see precond_er.py).  Isolates mechanism from estimator variance.
#
# Fixed across all runs: apply_to=current, fisher.target=replay, warm-started
# CG.  cg.iters is raised to 25 (default; override via CG_ITERS): the system
# (F+δI) has condition number ~λ_max/δ, so small δ needs more CG iterations —
# check the logged cg_residual if you push δ below 0.03.
#
# Default: 4 dampings × 2 buffer legs × 1 momentum leg (µ=0) × 5 seeds
# = 40 runs.  Momentum legs opt-in via MOMENTUM_SET (see below): the µ=0.9
# transient confounds the δ readout, so answer the µ=0 question first.
#
# Usage
# -----
#   bash scripts/run_P.sh                          # full default sweep
#   DAMPINGS="0.1 0.03" bash scripts/run_P.sh      # subset of δ
#   BUFFER_SET=full      bash scripts/run_P.sh     # fullbuf leg only
#   MOMENTUM_SET=both    bash scripts/run_P.sh     # add the µ=0.9 leg
#   SEEDS="1,2,3,4,5"    bash scripts/run_P.sh     # custom seeds
#   N_JOBS=4             bash scripts/run_P.sh     # cap parallelism
#   CG_ITERS=40          bash scripts/run_P.sh     # tighter CG solves
#   DRY_RUN=1            bash scripts/run_P.sh     # preview only
#
# CG-convergence control
# ----------------------
# A non-default CG_ITERS is appended to the ablation_value as "_cg<N>" so
# control runs never mix with the main sweep during aggregation.  The solver
# control for the δ=0.03 fullbuf relaxation-oscillation cliffs (mean residual
# ~0.2 at 25 iters; rule out under-convergence before claiming curvature
# blindness) is:
#
#   CG_ITERS=60 DAMPINGS="0.03" BUFFER_SET=full bash scripts/run_P.sh
#
# → PER_asym_d0.03_fullbuf_cg60, 5 seeds.  Readout: cliffs persist at the same
# window steps (~206-219) → curvature-blindness claim stands; cliffs vanish →
# it was solver error.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

DAMPINGS="${DAMPINGS:-1.0 0.3 0.1 0.03}"

# BUFFER_SET: "std" → 1 k reservoir only, "full" → 60 k full-buffer only,
# "both" → both legs.
BUFFER_SET="${BUFFER_SET:-both}"

# MOMENTUM_SET: "off" → µ=0.0 only (default — clean δ readout), "on" → µ=0.9
# only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-off}"

CG_ITERS_DEFAULT=25
CG_ITERS="${CG_ITERS:-${CG_ITERS_DEFAULT}}"

# Non-default CG iteration counts are control runs — suffix the ablation_value
# so they aggregate separately from the main sweep.
if [ "$CG_ITERS" != "$CG_ITERS_DEFAULT" ]; then
    CG_SUFFIX="_cg${CG_ITERS}"
else
    CG_SUFFIX=""
fi

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Two-task rot-MNIST: a single, clean task transition (matches decomposition).
DATASET_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=2
    'dataset.rotations_deg=[0,90]'
)

# Dense per-step eval over a 250-step window starting at the transition.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=250
)

ABLATION_KEY="per_asym_sweep"

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
echo "Dampings:      ${DAMPINGS}"
echo "Buffer set:    ${BUFFER_SET}"
echo "Momentum set:  ${MOMENTUM_SET}"
echo "CG iters:      ${CG_ITERS}"
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

momentum_values() {
    case "${MOMENTUM_SET}" in
        off)  echo "${MOM_OFF}" ;;
        on)   echo "${MOM_ON}"  ;;
        both) echo "${MOM_OFF} ${MOM_ON}" ;;
        *) echo "ERROR: unknown MOMENTUM_SET=${MOMENTUM_SET} (expected off|on|both)" >&2; exit 1 ;;
    esac
}

buffer_values() {
    case "${BUFFER_SET}" in
        std)  echo "std" ;;
        full) echo "full" ;;
        both) echo "std full" ;;
        *) echo "ERROR: unknown BUFFER_SET=${BUFFER_SET} (expected std|full|both)" >&2; exit 1 ;;
    esac
}

# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────

for mu in $(momentum_values); do
    if [ "$mu" = "${MOM_ON}" ]; then mom_suffix="_M"; mom_tag="mom_on"
    else mom_suffix=""; mom_tag="mom_off"; fi

    for buf in $(buffer_values); do
        if [ "$buf" = "full" ]; then
            buf_suffix="_fullbuf"
            BUF_OVERRIDES=(method.replay_full_buffer=true memory.total_budget=60000)
        else
            buf_suffix=""
            BUF_OVERRIDES=()
        fi

        for delta in $DAMPINGS; do
            ablation_value="PER_asym_d${delta}${buf_suffix}${CG_SUFFIX}${mom_suffix}"
            label="P_d${delta}${buf_suffix}${CG_SUFFIX}"
            echo ""
            echo "=== ${ablation_value}: asymmetric PER, δ=${delta}, buffer=${buf}, µ=${mu} ==="
            tags_csv="${ABLATION_KEY},${label},${mom_tag}"
            for seed in "${SEED_ARRAY[@]}"; do
                spawn_job \
                    method=precond_er \
                    "${DATASET_OVERRIDES[@]}" \
                    "${EVAL_OVERRIDES[@]}" \
                    "training.momentum=${mu}" \
                    "seed=${seed}" \
                    "+ablation_key=${ABLATION_KEY}" \
                    "+ablation_value=${ablation_value}" \
                    "tracking.wandb.tags=[${tags_csv}]" \
                    method.apply_to=current \
                    method.fisher.target=replay \
                    "method.fisher.damping=${delta}" \
                    "method.cg.iters=${CG_ITERS}" \
                    method.cg.warm_start=true \
                    "${BUF_OVERRIDES[@]+"${BUF_OVERRIDES[@]}"}"
            done
            wait_block "${ablation_value}"
        done
    done
done

echo ""
echo "=== PER_ASYM damping sweep complete ==="
