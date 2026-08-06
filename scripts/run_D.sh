#!/usr/bin/env bash
# scripts/run_D.sh
#
# D-series: the three-contributor decomposition experiments described in
# Chapter 4 §4.4 of the thesis, plus the momentum-cross rows for those
# conditions described in §4.6.
#
# Aim
# ---
# Empirically isolate the three contributors to the stability gap — magnitude
# bias, estimator bias, trajectory effect — and quantify each one's share of
# the total gap.  A path of conditions removes each contributor by
# construction:
#
#   D1 → D3 → D4   removes magnitude, then estimator
#   D1 → D2 → D4   removes estimator, then magnitude
#
# D4 is the cleanest isolation of the trajectory contributor (no buffer noise,
# no magnitude asymmetry, only the SGD-trajectory geometry of Kao et al.).
#
# REVISION (feedback round, 2026-07): D3/D4 now use ``balanced_direction``
# mode — equal-magnitude mixing rescaled to the raw joint gradient norm, so
# the step length matches vanilla ER exactly.  The original ``balanced`` mode
# unit-normalised the joint direction (fixed step length η), confounding the
# rebalancing with a step-size intervention.  The old runs remain in
# outputs/ under ablation values D3_balanced / D4_fulldata_balanced
# (appendix); the new runs land under D3_balanced_dir /
# D4_fulldata_balanced_dir.  The step-size question itself is tested
# separately by the global LR warm-up block, scripts/run_M.sh.
#
# Conditions (default set: D1 D2 D3 D4 D5 D6)
# -------------------------------------------
#   D1 — Vanilla ER          (buffer 1 k, standard mode)         total gap
#   D2 — Full-data ER        (buffer 60 k, standard mode)        − estimator noise
#   D3 — Balanced-dir ER     (buffer 1 k, balanced_direction)    − magnitude asymmetry,
#              step length matched to vanilla
#   D4 — Full-data bal-dir   (buffer 60 k, balanced_direction)   − magnitude + estimator
#   D5 — Asymmetric PER      (buffer 1 k; replay Fisher filters the current-task
#              gradient only, replay gradient added raw)          preconditioner reference
#   D6 — Averaged GEM        (buffer 1 k; gradient projection)   projection reference
#
# Momentum cross (§4.6)
# ---------------------
# Each active condition is run twice — once at training.momentum=0.0 (plain
# SGD baseline) and once at training.momentum=0.9 (the momentum-on condition).
# Momentum-on names get a "_M" suffix in ablation_value.
#
# Total (default): 6 base × 2 momentum × 5 seeds = 60 runs.  rot-MNIST + MLP
# runs in seconds; the full block finishes in <30 min at N_JOBS=5.
#
# Per-step instrumentation enabled for the ER conditions D1–D4:
#   - cos(g_replay, g_true)          (directional buffer fidelity)
#   - ‖g_replay‖ / ‖g_true‖          (magnitude buffer fidelity)
#   - ‖g_new‖   / ‖g_replay‖         (update-side magnitude asymmetry)
#   - replay loss, per-step T₀ accuracy
#
# Usage
# -----
#   bash scripts/run_D.sh                         # default conditions (D1–D6)
#   CONDITIONS="D1 D3 D4" bash scripts/run_D.sh   # subset
#   MOMENTUM_SET="off"   bash scripts/run_D.sh    # only the µ=0 leg
#   MOMENTUM_SET="on"    bash scripts/run_D.sh    # only the µ=0.9 leg
#   SEEDS="1,2,3,4,5"    bash scripts/run_D.sh    # custom seeds
#   N_JOBS=4             bash scripts/run_D.sh    # cap parallelism
#   DRY_RUN=1            bash scripts/run_D.sh    # preview only

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_CONDITIONS_DEFAULT="D1 D2 D3 D4 D5 D6"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Two-task rot-MNIST: a single, clean task transition.
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

# Buffer-fidelity diagnostics are essential for the D-series: they provide
# the per-step decomposition evidence (estimator-bias contributor).
GRAD_DIAG_ON=(
    eval.stability_gap.grad_diagnostics.enabled=true
)

ABLATION_KEY="decomposition"

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

# Generic ER condition launcher.  Loops over seeds inside; caller passes any
# additional Hydra overrides as further arguments.
spawn_er_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=er \
            "${DATASET_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom})"
}

# Preconditioned-ER launcher.  Replay-based (1 k reservoir), but the g_true
# buffer-fidelity diagnostics are implemented only by the ER method, so the
# caller leaves that toggle off (it would be inert here).
spawn_precond_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=precond_er \
            "${DATASET_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom})"
}

# A-GEM launcher.  Replay-based, but the projection (not an ER-style g_new +
# g_replay combine) means the buffer-fidelity g_true diagnostics — which only
# the ER method implements — do not apply; the toggle is left off, as for NCL.
spawn_agem_block() {
    local label="$1"; shift
    local base_ablation="$1"; shift
    local mom="$1"; shift
    local suffix
    suffix="$(mom_suffix "${mom}")"
    local ablation_value="${base_ablation}${suffix}"
    local mom_tag
    if [ "$mom" = "${MOM_ON}" ]; then mom_tag="mom_on"; else mom_tag="mom_off"; fi
    local tags_csv="${ABLATION_KEY},${label},${mom_tag}"
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=agem \
            "${DATASET_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "training.momentum=${mom}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "tracking.wandb.tags=[${tags_csv}]" \
            "$@"
    done
    wait_block "${label} (µ=${mom})"
}

# ──────────────────────────────────────────────────────────────────────────────
# D-series — Three-contributor decomposition (§4.4)
# ──────────────────────────────────────────────────────────────────────────────

for mu in $(momentum_values); do
    echo ""
    echo "=================================================================="
    echo "  Decomposition block — training.momentum = ${mu}"
    echo "=================================================================="

    if should_run D1; then
        echo "=== D1: vanilla ER (1 k reservoir, standard, unbalanced) ==="
        spawn_er_block D1 D1_vanilla "${mu}" \
            method.mode=standard \
            "${GRAD_DIAG_ON[@]}"
    fi

    if should_run D2; then
        echo "=== D2: full-data ER (60 k buffer, standard, unbalanced) — removes estimator noise ==="
        # replay_full_buffer=true: every step computes the replay gradient on
        # every stored sample exactly once (no with-replacement sampling).
        # Combined with memory.total_budget=60000 (= the full MNIST training
        # set per task), this yields the exact empirical past-task gradient
        # at the current parameters — zero sampling noise.  Peak MLP memory
        # ≈ 0.6 GB; fits comfortably on 64 GB RAM.
        spawn_er_block D2 D2_fulldata "${mu}" \
            method.mode=standard \
            method.replay_full_buffer=true \
            memory.total_budget=60000 \
            "${GRAD_DIAG_ON[@]}"
    fi

    if should_run D3; then
        echo "=== D3: balanced-direction ER (1 k reservoir) — removes magnitude asymmetry at vanilla's step length ==="
        # Direction: ĝ_new + ĝ_rep (equal magnitude); step length rescaled to
        # ‖g_new + g_replay‖, bit-for-bit what vanilla ER would step.  Pure
        # mixing intervention — no step-size change to confess.
        spawn_er_block D3 D3_balanced_dir "${mu}" \
            method.mode=balanced_direction \
            "${GRAD_DIAG_ON[@]}"
    fi

    if should_run D4; then
        echo "=== D4: full-data balanced-direction ER (60 k buffer) — isolates trajectory residual ==="
        # See D2 for the rationale on replay_full_buffer=true, D3 for the
        # balanced_direction update.
        spawn_er_block D4 D4_fulldata_balanced_dir "${mu}" \
            method.mode=balanced_direction \
            method.replay_full_buffer=true \
            memory.total_budget=60000 \
            "${GRAD_DIAG_ON[@]}"
    fi

    # Preconditioned-ER / projection reference conditions.  The PER solve is the
    # damped natural-gradient δ·(F+δI)⁻¹ (CG over Fisher-vector products) with
    # settings pinned for reproducibility: δ=1.0 damping, 10 CG iters, warm-start.
    # No grad_diagnostics: the g_true buffer-fidelity hooks are ER-only.
    #
    # D5 (asymmetric PER) is the preconditioner condition: the replay Fisher
    # filters ONLY the current-task gradient and the replay gradient is added
    # raw — d = δ(F_rep+δI)⁻¹ g_cur + g_rep.  A symmetric variant (Fisher
    # applied to the whole joint gradient) would rescale interference and
    # restoration identically, preserving ER's drift equilibrium and only
    # shrinking the spike; the asymmetric update keeps the full restoring
    # force on task-A-sharp directions and targets the trajectory bend.

    if should_run D5; then
        echo "=== D5: asymmetric preconditioned ER — replay Fisher on g_cur only, g_rep raw ==="
        spawn_precond_block D5 D5_PER_asym "${mu}" \
            method.apply_to=current \
            method.fisher.target=replay \
            method.fisher.damping=1.0 \
            method.cg.iters=10 \
            method.cg.warm_start=true
    fi

    if should_run D6; then
        echo "=== D6: averaged GEM (1 k reservoir, single-constraint projection) — projection reference ==="
        # Original A-GEM (Chaudhry et al. 2019): projects the current-task
        # gradient so it does not increase average loss on a buffer mini-batch.
        # reference_gradient=joint is the single-constraint A-GEM; margin=0.0 is
        # the paper formulation (project only when g̃·g_ref < 0) — pinned here
        # because configs/method/agem.yaml defaults to a non-standard margin=0.5.
        spawn_agem_block D6 D6_AGEM_reference "${mu}" \
            method.gem.reference_gradient=joint \
            method.gem.margin=0.0
    fi
done

echo ""
echo "=== Decomposition sweep complete ==="
