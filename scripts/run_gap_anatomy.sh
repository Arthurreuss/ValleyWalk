#!/usr/bin/env bash
# scripts/run_gap_anatomy.sh
#
# Coherent end-to-end experiment script for the curriculum-centric stability-
# gap thesis.  Supersedes earlier decomposition / curriculum scripts.
#
# Narrative anchor (rewritten for the new draft):
#   The stability gap is caused by the discontinuous switch in optimisation
#   target at task boundary (landscape teleportation).  The principled fix is
#   the asymmetric λ-curriculum, which restores continuity to the schedule.
#   Two refinements pin down the residual fluctuation that survives the basic
#   curriculum:
#     • momentum dampens SGD-noise-driven oscillation along the optimum path;
#     • λ_min keeps a non-zero new-task signal from step 1, preventing the
#       early-task progress that pure-replay starts forfeit.
#   Gradient *balancing* is demoted: it does reduce depth, but it destroys
#   SGD's natural step-size adaptation and pays for it in final accuracy.
#   It is reported as a *foil* — what NOT to do — rather than the headline.
#
# Four blocks:
#
#   ┌───────────────────────────────────────────────────────────────────────┐
#   │ G-series — Gap diagnostic (3 conditions)                              │
#   │   G1  vanilla ER (standard, buf=1k)        — G_total baseline         │
#   │   G2  full-data joint (standard, buf=60k)  — perfect estimator;       │
#   │                                              gap persists ⇒ not a    │
#   │                                              buffer problem alone     │
#   │   G3  balanced ER (buf=1k)                 — foil: depth↓ but ACC↓    │
#   │                                                                       │
#   │ Three-contributor diagnostic compressed into the smallest design that │
#   │ tells the story: there IS a gap (G1), better estimation is not a     │
#   │ complete fix (G2), magnitude balancing has the wrong tradeoff (G3).   │
#   │ The full 2×2 decomposition table moves to an appendix.                │
#   ├───────────────────────────────────────────────────────────────────────┤
#   │ C-series — λ-curriculum on standard ER (4 conditions, headline)       │
#   │   C1  std ER + curriculum N=50  linear                                │
#   │   C2  std ER + curriculum N=100 linear                                │
#   │   C3  std ER + curriculum N=200 linear                                │
#   │   C4  std ER + curriculum adaptive (closed-loop EMA ratio)            │
#   │                                                                       │
#   │ Tests H1: depth(N) is monotone non-increasing in N on standard (un-   │
#   │ balanced) ER, with no ACC collapse.  C4 vs the linear sweep tests     │
#   │ whether the closed-loop schedule reaches the linear-N Pareto front.   │
#   ├───────────────────────────────────────────────────────────────────────┤
#   │ R-series — Refinements at best N=200 (4 conditions)                   │
#   │   R1  C3 + momentum=0.9                    — fluctuation suppression  │
#   │   R2  C3 + λ_min=0.1                       — early-task progress      │
#   │   R3  C3 + momentum=0.9 + λ_min=0.1        — combined                 │
#   │   R4  std ER + momentum=0.9 (no curriculum) — falsifies "momentum     │
#   │                                              alone fixes it"          │
#   │                                                                       │
#   │ Tests H2 (momentum × curriculum): momentum reduces residual depth     │
#   │ variance but only WITH the curriculum; alone it cannot remove the    │
#   │ discontinuity.  Tests H3 (λ_min): a small floor on λ raises end-of-   │
#   │ task ACC at modest cost to depth, moving the Pareto frontier outward. │
#   ├───────────────────────────────────────────────────────────────────────┤
#   │ F-series — Mechanism falsification on standard ER (6 conditions)      │
#   │   F1a/F1b/F1c  symmetric lr warm-up at N=50/100/200       (M2)        │
#   │   F2a/F2b/F2c  step delay (λ=0 then snap to 1) at same N  (M3)        │
#   │                                                                       │
#   │ Tests H4: the depth reduction in C is specifically due to the         │
#   │ ASYMMETRIC scaling of ∇L_current.  M2 (symmetric scale) preserves the │
#   │ direction ratio and should leave depth flat in N.  M3 (binary delay)  │
#   │ reintroduces a discontinuity at step N and either matches C in depth  │
#   │ at small N or starves the budget at large N.                          │
#   └───────────────────────────────────────────────────────────────────────┘
#
# Per-step instrumentation (logged automatically when task_id > 0):
#   • cos(g_replay, g_true)          — directional alignment
#   • ‖g_replay‖ / ‖g_true‖          — magnitude ratio
#   • L_replay (replay loss)          — direct envelope-theorem readout
#   • per-step accuracy on T_0        — the gap itself
# These appear in W&B step logs and summaries on the run object.  g_true
# diagnostics are enabled only for the G-series (the only block where
# per-step decomposition evidence is load-bearing).
#
# Total: 3 + 4 + 4 + 6 = 17 conditions × 5 seeds = 85 runs.  rot-MNIST + MLP
# runs in seconds; the full sweep finishes in <1 hour at N_JOBS=4.
#
# Usage:
#   bash scripts/run_gap_anatomy.sh                         # all conditions
#   CONDITIONS="G1 G2"  bash scripts/run_gap_anatomy.sh     # subset
#   SERIES=C            bash scripts/run_gap_anatomy.sh     # one block
#   SEEDS="1,2,3,4,5"   bash scripts/run_gap_anatomy.sh     # custom seeds
#   N_JOBS=4            bash scripts/run_gap_anatomy.sh     # cap parallelism
#   DRY_RUN=1           bash scripts/run_gap_anatomy.sh     # preview commands

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

ALL_G="G1 G2 G3"
ALL_C="C1 C2 C3 C4"
ALL_R="R1 R2 R3 R4"
ALL_F="F1a F1b F1c F2a F2b F2c"
ALL_CONDITIONS="${ALL_G} ${ALL_C} ${ALL_R} ${ALL_F}"

# SERIES env var: shortcut to run one block at a time.  Example: SERIES=C runs
# only the C-series.  Ignored if CONDITIONS is set explicitly.
case "${SERIES:-}" in
    G) DEFAULT_CONDITIONS="${ALL_G}" ;;
    C) DEFAULT_CONDITIONS="${ALL_C}" ;;
    R) DEFAULT_CONDITIONS="${ALL_R}" ;;
    F) DEFAULT_CONDITIONS="${ALL_F}" ;;
    "") DEFAULT_CONDITIONS="${ALL_CONDITIONS}" ;;
    *) echo "ERROR: unknown SERIES=${SERIES} (expected one of G/C/R/F)" >&2; exit 1 ;;
esac
CONDITIONS="${CONDITIONS:-$DEFAULT_CONDITIONS}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Two-task rot-MNIST: a single, clean task transition.  Same for every series.
DATASET_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=2
    'dataset.rotations_deg=[0,90]'
)

# Window of 500 steps after the task transition.  eval_freq_steps=1 gives the
# per-step accuracy curves required for depth/area metrics.
EVAL_OVERRIDES=(
    eval.stability_gap.eval_freq_steps=1
    eval.stability_gap.window_steps=500
)

# g_true diagnostics — enabled only for the G-series.  C/R/F runs read the
# toggle as off (default) and run at full speed.
GRAD_DIAG_ON=(
    eval.stability_gap.grad_diagnostics.enabled=true
)

# Best-N for the R-series.  Picked from the prior E-series (the linear sweep
# bottoms out near N=200 on rot-MNIST + MLP).  R1/R2/R3 stack momentum and
# λ_min on top of this curriculum.
BEST_N=200
LAMBDA_MIN=0.1
MOMENTUM=0.9

# Single +ablation_key for the whole study so aggregate_results.py can group
# all conditions in one table; +ablation_value carries the series prefix
# (G1_*, C1_*, R1_*, F2c_*) for sub-grouping.
ABLATION_KEY="gap_anatomy"

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
echo "Git commit: ${GIT_COMMIT}"
echo "Python:     ${PYTHON}"
echo "Conditions: ${CONDITIONS}"
echo "Seeds:      ${SEEDS}"
echo "N_JOBS:     ${N_JOBS}"
echo "Best-N:     ${BEST_N} (R-series uses this curriculum length)"
echo "Dry run:    ${DRY_RUN}"
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

# Common base: ER, two-task rot-MNIST, dense eval.  Per-condition overrides
# are appended.
spawn_er_job() {
    local label="$1"; shift
    local ablation_value="$1"; shift
    for seed in "${SEED_ARRAY[@]}"; do
        spawn_job \
            method=er \
            "${DATASET_OVERRIDES[@]}" \
            "${EVAL_OVERRIDES[@]}" \
            "seed=${seed}" \
            "+ablation_key=${ABLATION_KEY}" \
            "+ablation_value=${ablation_value}" \
            "$@"
    done
    wait_block "${label}"
}

# ──────────────────────────────────────────────────────────────────────────────
# G-series — Gap diagnostic (3 conditions)
# ──────────────────────────────────────────────────────────────────────────────
#
# G1: vanilla ER on a small buffer — establishes the gap exists.
# G2: same vanilla ER but the "buffer" is the full past-task dataset (60k).
#     A perfect estimator does NOT close the gap to zero — by design, this
#     is the empirical readout of the irreducible trajectory + magnitude
#     contribution that no replay-quality intervention can remove.
# G3: balanced ER (per-component magnitude normalisation).  Reduces depth
#     dramatically by hiding magnitude information from SGD; pays for it in
#     ACC.  Reported as a foil/counter-example.

if should_run G1; then
    echo "=== G1: vanilla ER (standard, buffer=1k) — G_total baseline ==="
    spawn_er_job G1 G1_vanilla_std \
        method.mode=standard \
        "${GRAD_DIAG_ON[@]}"
fi

if should_run G2; then
    echo "=== G2: full-data joint, standard scheme (buffer=60k) — perfect estimator ==="
    spawn_er_job G2 G2_fulldata_std \
        method.mode=standard \
        method.replay_batch_size=4096 \
        memory.total_budget=60000 \
        "${GRAD_DIAG_ON[@]}"
fi

if should_run G3; then
    echo "=== G3: balanced ER (foil; depth↓ but ACC↓) ==="
    spawn_er_job G3 G3_balanced_foil \
        method.mode=balanced \
        method.grad_balance.normalize_components=true \
        method.grad_balance.task_weighted=false \
        "${GRAD_DIAG_ON[@]}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# C-series — λ-curriculum on STANDARD ER (headline)
# ──────────────────────────────────────────────────────────────────────────────
#
# Standard ER (no balancing).  Pure SGD on g_new + g_replay, with the new-
# task gradient asymmetrically weighted by λ(t) over the first N steps.
# This is the headline experimental block: it demonstrates the curriculum
# gives a Pareto improvement *without* the side effects of balancing.

run_std_curriculum() {
    local label="$1"           # C1 / C2 / C3
    local ramp_steps="$2"      # 50 / 100 / 200
    local ablation_value="$3"
    echo "=== ${label}: std ER + λ-curriculum, N=${ramp_steps} (linear) ==="
    spawn_er_job "${label}" "${ablation_value}" \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${ramp_steps}" \
        method.lambda_curriculum.schedule=linear
}

if should_run C1; then run_std_curriculum C1  50 C1_std_curriculum_N50;  fi
if should_run C2; then run_std_curriculum C2 100 C2_std_curriculum_N100; fi
if should_run C3; then run_std_curriculum C3 200 C3_std_curriculum_N200; fi

if should_run C4; then
    echo "=== C4: std ER + λ-curriculum, ADAPTIVE (closed-loop EMA ratio) ==="
    spawn_er_job C4 C4_std_curriculum_adaptive \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        method.lambda_curriculum.schedule=adaptive \
        method.lambda_curriculum.ema_alpha=0.05
fi

# ──────────────────────────────────────────────────────────────────────────────
# R-series — Refinements at best N (4 conditions)
# ──────────────────────────────────────────────────────────────────────────────
#
# Stacks momentum and λ_min on top of the best linear curriculum (N=200).
# R4 is the falsifier: momentum without the curriculum should NOT close the
# gap, because momentum does not address the schedule discontinuity.

if should_run R1; then
    echo "=== R1: C3 + momentum=${MOMENTUM} (curriculum + momentum) ==="
    spawn_er_job R1 R1_curriculum_momentum \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${BEST_N}" \
        method.lambda_curriculum.schedule=linear \
        "training.momentum=${MOMENTUM}"
fi

if should_run R2; then
    echo "=== R2: C3 + λ_min=${LAMBDA_MIN} (curriculum + min-λ floor) ==="
    spawn_er_job R2 R2_curriculum_lambda_min \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${BEST_N}" \
        method.lambda_curriculum.schedule=linear \
        "method.lambda_curriculum.lambda_min=${LAMBDA_MIN}"
fi

if should_run R3; then
    echo "=== R3: C3 + momentum=${MOMENTUM} + λ_min=${LAMBDA_MIN} (combined) ==="
    spawn_er_job R3 R3_curriculum_momentum_lambda_min \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${BEST_N}" \
        method.lambda_curriculum.schedule=linear \
        "method.lambda_curriculum.lambda_min=${LAMBDA_MIN}" \
        "training.momentum=${MOMENTUM}"
fi

if should_run R4; then
    echo "=== R4: std ER + momentum=${MOMENTUM} (NO curriculum — falsifier) ==="
    spawn_er_job R4 R4_momentum_only \
        method.mode=standard \
        method.lambda_curriculum.enabled=false \
        "training.momentum=${MOMENTUM}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# F-series — Mechanism falsification on STANDARD ER (6 conditions)
# ──────────────────────────────────────────────────────────────────────────────
#
# F1 (M2): symmetric lr warm-up — both g_new and g_replay slowed equally.
#          Direction ratio unchanged.  Predicted: depth flat in N.
# F2 (M3): step delay — λ=0 for N steps, then snaps to 1.  Reintroduces the
#          discontinuity at step N.  Predicted: comparable to C at small N,
#          budget-degenerate at large N (depth↓ but ACC collapse).
# Both run on the headline base (standard ER) so the comparison is direct.

run_lr_warmup() {
    local label="$1"
    local warmup_steps="$2"
    local ablation_value="$3"
    echo "=== ${label}: std ER + symmetric lr warm-up, warmup_steps=${warmup_steps} ==="
    spawn_er_job "${label}" "${ablation_value}" \
        method.mode=standard \
        method.lambda_curriculum.enabled=false \
        method.lr_warmup.enabled=true \
        "method.lr_warmup.warmup_steps=${warmup_steps}"
}

if should_run F1a; then run_lr_warmup F1a  50 F1a_lr_warmup_N50;  fi
if should_run F1b; then run_lr_warmup F1b 100 F1b_lr_warmup_N100; fi
if should_run F1c; then run_lr_warmup F1c 200 F1c_lr_warmup_N200; fi

run_step_delay() {
    local label="$1"
    local ramp_steps="$2"
    local ablation_value="$3"
    echo "=== ${label}: std ER + step delay (λ=0 then snap to 1), N=${ramp_steps} ==="
    spawn_er_job "${label}" "${ablation_value}" \
        method.mode=standard \
        method.lambda_curriculum.enabled=true \
        "method.lambda_curriculum.ramp_steps=${ramp_steps}" \
        method.lambda_curriculum.schedule=step
}

if should_run F2a; then run_step_delay F2a  50 F2a_step_delay_N50;  fi
if should_run F2b; then run_step_delay F2b 100 F2b_step_delay_N100; fi
if should_run F2c; then run_step_delay F2c 200 F2c_step_delay_N200; fi

echo ""
echo "=== Gap-anatomy sweep complete ==="
