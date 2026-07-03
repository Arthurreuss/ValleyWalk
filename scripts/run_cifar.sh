#!/usr/bin/env bash
# scripts/run_cifar.sh
#
# Unified CIFAR-10 sweep — replaces the old D-block in run_curriculum.sh and
# the H-block in run_cifar_gap_anatomy.sh.  All new CIFAR work happens here.
#
# Four opt-in blocks
# ------------------
#   headline        — the final 6 conditions for the §5.6 / §6.5 tables
#                     (3 methods × {BN, GN}, all at buffer 5000, µ = 0.9).
#   generalization  — does the headline finding carry to other corruptions
#                     and to 3-task sequences?  Vanilla-ER controls included
#                     so the comparison stays interpretable on the new shifts.
#   per_asym        — asymmetric PER on CIFAR (story-v2 RQ4: the directional
#                     gate's deep-backbone leg; see thesis_draft/story_v2/04).
#   momentum_cross  — µ = 0 legs for {vanilla, curriculum, asym PER} so the
#                     momentum-composability rule can be read on the ResNet.
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
# Asym-PER block (ablation_key = "cifar_per_asym")   — STILL TO RUN
# -----------------------------------------------------------------
# The directional feedforward gate  d = δ(F_rep+δI)⁻¹ g_cur + g_rep  on the
# deep backbone, matched to the headline protocol (dom_cifar10 2-task
# gaussian_noise, buffer 5000, µ = 0.9) so rows are directly comparable to
# D1/D3/D4/D6.  Single damping δ = 0.1: strong rot-MNIST performer on both
# axes (fullbuf µ=0 depth 0.022 / area 0.44; best momentum leg 0.076 depth at
# 0.965 ACC) while staying clear of the δ = 0.03 cliff regime.  Fixed like
# the rot-MNIST sweep: apply_to=current, fisher.target=replay, warm-started
# CG at 25 iters.
#
#   Label  Method                     δ      Norm   Buffer
#   P1     Asymmetric PER             0.1    BN     5000
#   P2     Asymmetric PER             0.1    GN     5000
#
# Vanilla / curriculum comparators are D1/D3 (BN) and D4/D6 (GN) — no new
# controls needed.  Note µ = 0.9 is the directional gate's *weakest* regime
# (momentum re-inflation), which is the honest test; rot-MNIST predicts depth
# between vanilla and curriculum at ≥ vanilla ACC.
#
# COST PROBE FIRST: CG runs ~25 Fisher-vector products per step on the
# ResNet-18.  Before committing the block, time a single seed:
#   CONDITIONS="P1" SEEDS_PER="1" BLOCKS="per_asym" bash scripts/run_cifar.sh
# If cost forces cuts, drop a norm before dropping seeds.
#
# A non-default PER_CG_ITERS is appended to the ablation_value as "_cg<N>"
# (same convention as run_per_asym_sweep.sh) so solver-control runs never mix
# with the main block during aggregation.
#
# Momentum-cross block (ablation_key = "cifar_momentum_cross")   — STILL TO RUN
# -----------------------------------------------------------------------------
# Unwraps momentum's effect per gate type on the deep backbone.  The
# rot-MNIST composability rule (source vs per-step attenuation,
# thesis_draft/story_v2/03) predicts, µ = 0 → µ = 0.9:
#   vanilla ER  — depth ~unchanged, tail cut          (nothing tamed, noise averaged)
#   curriculum  — gap helped/held, ACC up             (push attenuated at source)
#   asym PER    — depth re-inflated, ACC up           (push attenuated per step)
# The µ = 0.9 legs already exist (D4, D6 / P2 for gn; D1, D3 / P1 for bn);
# this block supplies the µ = 0 legs, at ONE norm.  Default CROSS_NORM=gn:
# the GN transient is pure optimisation dynamics (no cross-batch statistics
# to re-adapt), so the momentum mechanism reads out clean.  Set CROSS_NORM=bn
# to replicate.
#
#   Label  Method                          δ      µ     Norm         Buffer
#   M1     Vanilla ER                      —      0.0   $CROSS_NORM  5000
#   M2     Adaptive curr.  λ_min = 0.20    —      0.0   $CROSS_NORM  5000
#   M3     Asymmetric PER                  0.1    0.0   $CROSS_NORM  5000
#
# CAVEAT — probe before committing: µ = 0 SGD may under-converge task 0 in
# 10 epochs (lower pre-switch baseline).  Gap metrics are baseline-relative,
# so the cross stays readable, but check T0 convergence on one seed first:
#   CONDITIONS="M1" SEEDS_CROSS="1" BLOCKS="momentum_cross" bash scripts/run_cifar.sh
#
# Total runs (all blocks at default seed counts)
# ----------------------------------------------
#   headline:        6 × 5 seeds = 30
#   generalization:  6 × 3 seeds = 18
#   per_asym:        2 × 5 seeds = 10   (CG makes these slower than D-runs)
#   momentum_cross:  3 × 5 seeds = 15
#   Grand total                  = 73
#
# Usage
# -----
# BLOCKS is required — every block is opt-in.  Valid block names:
#   headline | generalization | per_asym | momentum_cross
#
#   BLOCKS="headline"                      bash scripts/run_cifar.sh
#   BLOCKS="generalization"                bash scripts/run_cifar.sh
#   BLOCKS="per_asym"                      bash scripts/run_cifar.sh
#   BLOCKS="per_asym momentum_cross"       bash scripts/run_cifar.sh
#   CONDITIONS="D3 D6" BLOCKS="headline"   bash scripts/run_cifar.sh   # subset
#   CONDITIONS="P1" SEEDS_PER="1" BLOCKS="per_asym" bash scripts/run_cifar.sh  # cost probe
#   CONDITIONS="M1" SEEDS_CROSS="1" BLOCKS="momentum_cross" bash scripts/run_cifar.sh  # µ=0 probe
#   SEEDS_HEADLINE="1,2,3" BLOCKS="headline" bash scripts/run_cifar.sh
#   SEEDS_GEN="1,2,3,4,5" BLOCKS="generalization" bash scripts/run_cifar.sh
#   PER_CG_ITERS=40 BLOCKS="per_asym"      bash scripts/run_cifar.sh   # solver control
#   CROSS_NORM=bn BLOCKS="momentum_cross"  bash scripts/run_cifar.sh
#   SHIP_NORM=gn BLOCKS="generalization"   bash scripts/run_cifar.sh
#   DRY_RUN=1 BLOCKS="headline" bash scripts/run_cifar.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# CIFAR runs on a single MPS GPU — default sequential.
N_JOBS="${N_JOBS:-1}"

# Seed sets — headline and per_asym run at 5 seeds (paired Wilcoxon-able),
# generalization at 3 (point-estimate "does it carry over").
SEEDS_HEADLINE="${SEEDS_HEADLINE:-1,2,3,4,5}"
SEEDS_GEN="${SEEDS_GEN:-1,2,3}"
SEEDS_PER="${SEEDS_PER:-${SEEDS_HEADLINE}}"
SEEDS_CROSS="${SEEDS_CROSS:-${SEEDS_HEADLINE}}"

ALL_HEADLINE_CONDITIONS="D1 D2 D3 D4 D5 D6"
ALL_GEN_CONDITIONS="S1 S1v S2 S2v T1 T1v"
ALL_PER_CONDITIONS="P1 P2"
ALL_CROSS_CONDITIONS="M1 M2 M3"
ALL_CONDITIONS_DEFAULT="${ALL_HEADLINE_CONDITIONS} ${ALL_GEN_CONDITIONS} ${ALL_PER_CONDITIONS} ${ALL_CROSS_CONDITIONS}"
CONDITIONS="${CONDITIONS:-$ALL_CONDITIONS_DEFAULT}"

BLOCKS="${BLOCKS:-}"
VALID_BLOCKS=(headline generalization per_asym momentum_cross)

# Which norm the momentum-cross block runs at (its µ=0.9 counterparts are
# D4/D6/P2 for gn, D1/D3/P1 for bn).
CROSS_NORM="${CROSS_NORM:-gn}"

# CG iterations for the per_asym block (rot-MNIST sweep default).  Non-default
# values suffix the ablation_value with "_cg<N>" so control runs aggregate
# separately (matches run_per_asym_sweep.sh).
PER_CG_ITERS_DEFAULT=25
PER_CG_ITERS="${PER_CG_ITERS:-${PER_CG_ITERS_DEFAULT}}"
if [ "$PER_CG_ITERS" != "$PER_CG_ITERS_DEFAULT" ]; then
    PER_CG_SUFFIX="_cg${PER_CG_ITERS}"
else
    PER_CG_SUFFIX=""
fi

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
)

# Momentum is injected per condition by spawn_condition (Hydra rejects
# duplicate overrides, so it cannot live in CIFAR_BASE).  µ = 0.9 everywhere
# except the momentum_cross block, which flips this to 0.0.
COND_MOMENTUM=0.9

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

# Asymmetric PER — fixed choices mirror the rot-MNIST sweep
# (run_per_asym_sweep.sh): gate the current-task gradient only, through the
# replay Fisher, warm-started CG.  δ comes per condition.
PER_ASYM_BASE=(
    method=precond_er
    method.apply_to=current
    method.fisher.target=replay
    "method.cg.iters=${PER_CG_ITERS}"
    method.cg.warm_start=true
)

GN_OVERRIDES=(
    model.norm_type=group
    model.norm_groups=32
)

# The headline task pair — shared by the headline and per_asym blocks.
TWO_TASK_GAUSSIAN=(
    dataset.num_tasks=2
    'dataset.corruption_types=[none,gaussian_noise]'
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

if [ "$CROSS_NORM" != "bn" ] && [ "$CROSS_NORM" != "gn" ]; then
    echo "ERROR: CROSS_NORM='${CROSS_NORM}' (expected 'bn' or 'gn')" >&2
    exit 1
fi

if [ "$DRY_RUN" != "1" ] && [ -n "$(git status --porcelain)" ]; then
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
echo "Seeds (per_asym): ${SEEDS_PER}"
echo "Seeds (cross):    ${SEEDS_CROSS}"
echo "PER CG iters:     ${PER_CG_ITERS}"
echo "Ship norm (gen):  ${SHIP_NORM}"
echo "Cross norm:       ${CROSS_NORM}"
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
            "training.momentum=${COND_MOMENTUM}" \
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

# ──────────────────────────────────────────────────────────────────────────────
# Block C — asymmetric PER (ablation_key=cifar_per_asym)
# ──────────────────────────────────────────────────────────────────────────────
#
# Directional feedforward gate on the headline protocol.  Comparators are the
# existing D1/D3 (BN) and D4/D6 (GN) rows — same tasks, buffer, µ, eval.

if has_block per_asym; then
    echo ""
    echo "=================================================================="
    echo "  CIFAR asym PER — δ=0.1 × {BN, GN}, buffer 5000, µ=0.9"
    echo "=================================================================="

    if should_run P1; then
        echo "=== P1: asym PER δ=0.1, BN, buffer 5k ==="
        spawn_condition P1 "PER_asym_d0.1_BN_buf5k${PER_CG_SUFFIX}" per_asym cifar_per_asym "$SEEDS_PER" \
            "${TWO_TASK_GAUSSIAN[@]}" "${PER_ASYM_BASE[@]}" \
            method.fisher.damping=0.1 \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run P2; then
        echo "=== P2: asym PER δ=0.1, GN, buffer 5k ==="
        spawn_condition P2 "PER_asym_d0.1_GN_buf5k${PER_CG_SUFFIX}" per_asym cifar_per_asym "$SEEDS_PER" \
            "${TWO_TASK_GAUSSIAN[@]}" "${PER_ASYM_BASE[@]}" "${GN_OVERRIDES[@]}" \
            method.fisher.damping=0.1 \
            "memory.total_budget=${BUFFER_BIG}"
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Block D — momentum cross (ablation_key=cifar_momentum_cross)
# ──────────────────────────────────────────────────────────────────────────────
#
# µ = 0 legs for the three gate types at CROSS_NORM; their µ = 0.9 pairs are
# the existing headline/per_asym rows.  Composability predictions in header.

if has_block momentum_cross; then
    echo ""
    echo "=================================================================="
    echo "  CIFAR momentum cross — µ=0 legs, norm=${CROSS_NORM}, buffer 5000"
    echo "=================================================================="

    CROSS_NORM_OVERRIDES=()
    if [ "$CROSS_NORM" = "gn" ]; then CROSS_NORM_OVERRIDES=("${GN_OVERRIDES[@]}"); fi

    COND_MOMENTUM=0.0

    if should_run M1; then
        echo "=== M1: vanilla ER, ${CROSS_NORM}, µ=0, buffer 5k ==="
        spawn_condition M1 "M1_vanilla_${CROSS_NORM}_buf5k_m0" vanilla cifar_momentum_cross "$SEEDS_CROSS" \
            "${TWO_TASK_GAUSSIAN[@]}" "${VANILLA_ER[@]}" \
            "${CROSS_NORM_OVERRIDES[@]:+${CROSS_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run M2; then
        echo "=== M2: adaptive curriculum (λ_min=0.20), ${CROSS_NORM}, µ=0, buffer 5k ==="
        spawn_condition M2 "M2_adaptive_lmin0.20_${CROSS_NORM}_buf5k_m0" adaptive cifar_momentum_cross "$SEEDS_CROSS" \
            "${TWO_TASK_GAUSSIAN[@]}" "${ADAPTIVE_CURRICULUM[@]}" \
            "${CROSS_NORM_OVERRIDES[@]:+${CROSS_NORM_OVERRIDES[@]}}" \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    if should_run M3; then
        echo "=== M3: asym PER δ=0.1, ${CROSS_NORM}, µ=0, buffer 5k ==="
        spawn_condition M3 "M3_PER_asym_d0.1_${CROSS_NORM}_buf5k_m0${PER_CG_SUFFIX}" per_asym cifar_momentum_cross "$SEEDS_CROSS" \
            "${TWO_TASK_GAUSSIAN[@]}" "${PER_ASYM_BASE[@]}" \
            "${CROSS_NORM_OVERRIDES[@]:+${CROSS_NORM_OVERRIDES[@]}}" \
            method.fisher.damping=0.1 \
            "memory.total_budget=${BUFFER_BIG}"
    fi

    COND_MOMENTUM=0.9
fi

echo ""
echo "=== CIFAR sweep complete ==="
