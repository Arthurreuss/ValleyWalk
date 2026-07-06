#!/usr/bin/env bash
# scripts/run_D.sh
#
# D-series: the three-contributor decomposition experiments described in
# Chapter 4 §4.4 of the thesis, plus the momentum-cross rows for those
# conditions described in §4.6.  (Formerly run_decomposition.sh; the G-series
# label was reassigned to the CIFAR generalisation block in scripts/run_G.sh.)
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
# Conditions (default set: D1 D2 D3 D4 D5 D6)
# -------------------------------------------
#   D1 — Vanilla ER          (buffer 1 k, standard mode)         total gap
#   D2 — Full-data ER        (buffer 60 k, standard mode)        − estimator noise
#   D3 — Balanced ER         (buffer 1 k, balanced mode)         − magnitude asymmetry
#   D4 — Full-data balanced  (buffer 60 k, balanced mode)        − magnitude + estimator
#   D5 — Asymmetric PER      (buffer 1 k; replay Fisher filters the current-task
#              gradient only, replay gradient added raw)          preconditioner reference
#   D6 — Averaged GEM        (buffer 1 k; gradient projection)   projection reference
#
# Deprecated reference conditions (D7–D10, commented out below; the collected
# results are kept on disk under these labels but are no longer run by default):
#   D7  — Standard NCL       (no replay buffer; precision prior).  Cut from the
#              final story; the α-tuned NCL path-finding reference.
#   D8  — PER_JOINT   \ symmetric preconditioned ER (Fisher of the joint / replay
#   D9  — PER_REPLAY  / loss applied to the whole joint gradient).  Near-identical
#              by construction; superseded by D5 (asymmetric PER).
#   D10 — PER_JOINT_FULLBUF — joint-Fisher PER with a full-buffer replay gradient
#              (variance-vs-trajectory probe).
# To re-run any deprecated condition, uncomment its block and pass it via
# CONDITIONS (e.g. CONDITIONS="D7").
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

# NCL condition launcher.  No replay buffer; grad_diagnostics toggle is set
# but silently ignored by NCL (no g_replay to compare against).
spawn_ncl_block() {
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
            method=ncl \
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
        echo "=== D3: balanced ER (1 k reservoir, balanced, normalised) — removes magnitude asymmetry ==="
        spawn_er_block D3 D3_balanced "${mu}" \
            method.mode=balanced \
            method.grad_balance.normalize_components=true \
            method.grad_balance.task_weighted=false \
            "${GRAD_DIAG_ON[@]}"
    fi

    if should_run D4; then
        echo "=== D4: full-data balanced ER (60 k buffer, balanced) — isolates trajectory residual ==="
        # See D2 for the rationale on replay_full_buffer=true.
        spawn_er_block D4 D4_fulldata_balanced "${mu}" \
            method.mode=balanced \
            method.replay_full_buffer=true \
            memory.total_budget=60000 \
            method.grad_balance.normalize_components=true \
            method.grad_balance.task_weighted=false \
            "${GRAD_DIAG_ON[@]}"
    fi

    # Preconditioned-ER / projection reference conditions.  The PER solve is the
    # damped natural-gradient δ·(F+δI)⁻¹ (CG over Fisher-vector products) with
    # settings pinned for reproducibility: δ=1.0 damping, 10 CG iters, warm-start.
    # No grad_diagnostics: the g_true buffer-fidelity hooks are ER-only.
    #
    # D5 (asymmetric PER) is the default preconditioner condition: the replay
    # Fisher filters ONLY the current-task gradient and the replay gradient is
    # added raw — d = δ(F_rep+δI)⁻¹ g_cur + g_rep.  The deprecated symmetric
    # variants (D8/D9) rescale interference and restoration identically, so they
    # preserve ER's drift equilibrium and can only shrink the spike; the
    # asymmetric update keeps the full restoring force on task-A-sharp directions
    # and is the one that targets the trajectory bend.

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

    # ──────────────────────────────────────────────────────────────────────
    # DEPRECATED reference conditions D7–D10 (commented out; not in the default
    # set).  The collected 5-seed results live on disk under these labels; the
    # blocks are kept for reproducibility.  Uncomment a block and pass its label
    # via CONDITIONS (e.g. CONDITIONS="D7") to re-run it.
    # ──────────────────────────────────────────────────────────────────────
    #
    # if should_run D7; then
    #     echo "=== D7 (DEPRECATED): standard NCL — precision-matrix path-finding reference ==="
    #     # prior_init=0.1 is the tuned winner from the α sweep (matches the
    #     # global default in configs/method/ncl.yaml).  NCL has no replay buffer,
    #     # so the g_replay diagnostics are silently ignored.  Cut from the final
    #     # story; see thesis_draft/notes/ncl_implementation_findings.md.
    #     spawn_ncl_block D7 D7_NCL_reference "${mu}" method.ncl.prior_init=0.1
    # fi
    #
    # if should_run D8; then
    #     echo "=== D8 (DEPRECATED): symmetric PER, Fisher of the JOINT (current+replay) batch ==="
    #     spawn_precond_block D8 D8_PER_joint "${mu}" \
    #         method.fisher.target=joint \
    #         method.fisher.damping=1.0 \
    #         method.cg.iters=10 \
    #         method.cg.warm_start=true
    # fi
    #
    # if should_run D9; then
    #     echo "=== D9 (DEPRECATED): symmetric PER, Fisher of the REPLAY batch only ==="
    #     spawn_precond_block D9 D9_PER_replay "${mu}" \
    #         method.fisher.target=replay \
    #         method.fisher.damping=1.0 \
    #         method.cg.iters=10 \
    #         method.cg.warm_start=true
    # fi
    #
    # if should_run D10; then
    #     echo "=== D10 (DEPRECATED): joint-Fisher PER + full-buffer replay gradient — variance/trajectory probe ==="
    #     spawn_precond_block D10 D10_PER_joint_fullbuf "${mu}" \
    #         method.fisher.target=joint \
    #         method.fisher.damping=1.0 \
    #         method.cg.iters=10 \
    #         method.cg.warm_start=true \
    #         method.replay_full_buffer=true \
    #         memory.total_budget=60000
    # fi
done

echo ""
echo "=== Decomposition sweep complete ==="
