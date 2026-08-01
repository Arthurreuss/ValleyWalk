#!/usr/bin/env bash
# scripts/run_S.sh
#
# S-series: the learning-rate ladder (RQ1).
#
# Aim
# ---
# Separate the two levels of the account by driving the one knob that gates
# all of Level Two — the step size — towards zero, and seeing what is left.
#
# A uniform rescaling of eta does not change the intended trajectory: forward
# Euler at step eta_k integrates the same ODE dtheta/dtau = -g(theta) on a
# finer grid, so a smaller eta follows the same path more accurately.  Anything
# that vanishes down the ladder is discretisation (Level Two); anything that
# survives belongs to the intended trajectory (Level One) — the arc.
#
# Four design constraints make the ladder read as a limit rather than as three
# unrelated runs:
#
#   1. MATCHED FLOW TIME.  Flow time is tau = sum of eta over steps, so the
#      epoch count scales as 1/eta.  Without this, small-eta cells simply do
#      not finish the trajectory and their small gap is under-training, not a
#      result.  (The M-series has exactly this defect: its ramp covers
#      sum(eta_t) = 13.45 against vanilla's 23.5, i.e. 57 %.)
#
#   2. FROZEN theta*_0.  Every cell warm-starts from one and the same task-0
#      checkpoint (init.checkpoint / init.skip_tasks, see scripts/train.py).
#      Training task 0 at the cell's own eta would land in a differently-sharp
#      minimum — less SGD noise means less implicit flattening — and since
#      d^2 L_rep/dt^2 = theta_dot^T H_rep theta_dot, a sharper past-task
#      Hessian *enlarges* the arc.  That confound points in the direction of
#      the hypothesis, so it has to go.  It also halves the compute and makes
#      every cross-rung comparison exactly paired.
#
#   3. MATCHED RECORD GRID.  The eval cadence scales with the epoch count, so
#      every cell records the same number of samples at the same flow-time
#      spacing.  gap_area_end (a sum over records) is therefore directly
#      comparable across the ladder with no rescaling; gap_depth is
#      eta-invariant by construction.
#
#   4. CADENCE COUNTED FROM THE BOUNDARY.  Inside the dense window the eval
#      cadence fires on task_step % eval_freq_steps rather than on global_step
#      (scripts/train.py).  Otherwise a stretch factor that does not divide the
#      235-batch epoch would put the first post-switch sample at an arbitrary
#      phase offset into the task — skipping exactly the first steps the spike
#      lives in.  The change is a no-op at eval_freq_steps = 1, which is every
#      run in the archive.
#
# Grid
# ----
#   eta in {0.1, 0.01, 0.1/33}  x  momentum in {0.0, 0.9}  x  buffer in {1k, 60k}
#   = 12 cells x 5 seeds = 60 runs, plus 10 task-0 anchors.
#
# The eta values are written out exactly (0.1/c) rather than rounded, so that
# eta * epochs is 0.1 in every rung; the thesis may report ~0.003.
#
# Two readings the grid supports beyond the ladder itself:
#
#   * MATCHED EFFECTIVE STEP.  Momentum multiplies the asymptotic step by
#     1/(1-mu), so the mu=0.9 rungs sit at effective 1.0, 0.1, 0.03 against the
#     mu=0 rungs' 0.1, 0.01, 0.003: (eta=0.01, mu=0.9) and (eta=0.1, mu=0) are
#     an exactly matched pair at 0.1, as are (0.1/33, 0.9) and (0.01, 0) at
#     0.03 vs 0.01 — one exact pair and one near pair.  If a momentum cell
#     behaves like its matched mu=0 partner, the momentum signature is an
#     effective-step effect; if it does not, it is the accumulator.  Plot
#     against effective flow time tau = t * eta/(1-mu) so the differing run
#     lengths line up.  NOTE: momentum is *not* a reparametrisation of the
#     flow, so the mu=0.9 leg is flow-matched only within itself — make
#     comparisons within a momentum setting and treat the diagonal as a
#     secondary read.
#
#   * O(eta) EXTRAPOLATION.  Forward Euler's global error is O(eta), so a
#     residual metric should behave as A(eta) = A_0 + k*eta over the small-eta
#     rungs.  Fitting it gives the eta -> 0 limit with an error bar instead of
#     asserting that the smallest rung is the limit.  A departure from
#     linearity at eta = 0.1 is itself the signature of the discrete
#     instability that produces the first-step spike.
#
# Conditions
# ----------
#   S1 — eta = 0.1        1 epoch    (the standard operating point)
#   S2 — eta = 0.01      10 epochs
#   S3 — eta = 0.1/33    33 epochs
# each x {1k reservoir, 60k exact} x {mu=0.0, mu=0.9}.
#
# ablation_value: S{1,2,3}_eta<val>[_exact][_M]   ablation_key: lr_ladder
# Deliberately NOT reusing the D2/D3 labels: those mean "full-data" and
# "balanced-direction" in 436 archived rows, in Appendix C and in every
# plotting script.
#
# The eta = 0.1 rung is re-run here rather than taken from the archived D1/D2
# rows, because every cell of the ladder has to share theta*_0.
#
# Buffer-fidelity diagnostics are OFF: they cost a full past-task
# forward+backward per step, unaffordable at 33 epochs, and D1/D2 already
# carry those numbers.
#
# Cost (task 0 is skipped, so this is T1 only)
# --------------------------------------------
#   1k arm:     ~3 / ~8 / ~25 min per run
#   exact arm:  ~2 / ~20 / ~65 min per run
#   whole grid at N_JOBS=5: roughly 4 h, i.e. one overnight batch.
#
# Usage
# -----
#   bash scripts/run_S.sh                         # anchors + full ladder
#   PHASES="anchors"     bash scripts/run_S.sh    # task-0 checkpoints only
#   PHASES="ladder"      bash scripts/run_S.sh    # ladder only (anchors exist)
#   CONDITIONS="S1 S2"   bash scripts/run_S.sh    # subset of rungs
#   ARMS="exact"         bash scripts/run_S.sh    # subset of buffer arms
#   MOMENTUM_SET="off"   bash scripts/run_S.sh    # only the mu=0 leg
#   SEEDS="1,2,3,4,5"    bash scripts/run_S.sh
#   N_JOBS=4             bash scripts/run_S.sh
#   DRY_RUN=1            bash scripts/run_S.sh    # preview only (works dirty)

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

N_JOBS="${N_JOBS:-5}"
SEEDS="${SEEDS:-1,2,3,4,5}"

CONDITIONS="${CONDITIONS:-S1 S2 S3}"

# ARMS: "standard" → 1 k reservoir, "exact" → 60 k full-buffer gradient.
ARMS="${ARMS:-standard exact}"

# MOMENTUM_SET: "off" → µ=0.0 only, "on" → µ=0.9 only, "both" → both legs.
MOMENTUM_SET="${MOMENTUM_SET:-both}"

# PHASES: "anchors", "ladder", or both.
PHASES="${PHASES:-anchors ladder}"

DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Where the shared task-0 checkpoints live.  One per (momentum, seed): the
# ladder is flow-matched within a momentum leg, so each leg gets a theta*_0
# trained at that momentum.
ANCHOR_ROOT="${PROJECT_ROOT}/outputs/S_anchors"

# Two-task rot-MNIST: a single, clean task transition.  Identical to run_D.sh
# so the rows join against the D-series in master_index.csv.
DATASET_OVERRIDES=(
    dataset=rot_mnist
    dataset.num_tasks=2
    'dataset.rotations_deg=[0,90]'
)

ABLATION_KEY="lr_ladder"

MOM_OFF=0.0
MOM_ON=0.9

# ──────────────────────────────────────────────────────────────────────────────
# Rung table
# ──────────────────────────────────────────────────────────────────────────────
#
# c is the flow-time stretch factor: eta = 0.1/c and epochs = c, so eta*epochs
# is 0.1 in every rung.  eval_freq = c and every_n = 10c keep the record grid
# identical across rungs; window = 250c always exceeds the 235c steps of the
# task, so every cell records the whole of T1 at the dense cadence — 235
# samples, evenly spaced in flow time, in every rung.
#
# c need not divide the 235-batch epoch: the dense cadence counts from the task
# boundary (constraint 4 in the header), so the first post-switch step is
# recorded in every rung regardless.
#
#              eta                    c (= epochs = eval_freq)   every_n
rung_spec() {
    case "$1" in
        S1) echo "0.1 1 10" ;;
        S2) echo "0.01 10 100" ;;
        S3) echo "0.0030303030303030303 33 330" ;;
    esac
}

# Short, stable tag for the ablation_value (the exact eta lives in the config).
rung_tag() {
    case "$1" in
        S1) echo "S1_eta0.1" ;;
        S2) echo "S2_eta0.01" ;;
        S3) echo "S3_eta0.003" ;;
    esac
}

# Buffer-arm overrides.  "exact" pairs replay_full_buffer with a budget equal
# to the per-task training-set size, giving the exact empirical past-task
# gradient at the current parameters (zero sampling noise on g_rep).
arm_overrides() {
    case "$1" in
        standard) echo "memory.total_budget=1000" ;;
        exact)    echo "method.replay_full_buffer=true memory.total_budget=60000" ;;
    esac
}

arm_suffix() {
    if [ "$1" = "exact" ]; then echo "_exact"; else echo ""; fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────────
#
# Validate the env-var vocabularies here rather than inside the lookup
# functions: those run in command substitutions, where an exit would only kill
# the subshell and leave the caller with an empty string.

for cond in $CONDITIONS; do
    case "$cond" in
        S1|S2|S3) ;;
        *) echo "ERROR: unknown condition '$cond' (expected S1|S2|S3)" >&2; exit 1 ;;
    esac
done

for arm in $ARMS; do
    case "$arm" in
        standard|exact) ;;
        *) echo "ERROR: unknown arm '$arm' (expected standard|exact)" >&2; exit 1 ;;
    esac
done

for phase in $PHASES; do
    case "$phase" in
        anchors|ladder) ;;
        *) echo "ERROR: unknown phase '$phase' (expected anchors|ladder)" >&2; exit 1 ;;
    esac
done

case "${MOMENTUM_SET}" in
    off|on|both) ;;
    *) echo "ERROR: unknown MOMENTUM_SET='${MOMENTUM_SET}' (expected off|on|both)" >&2; exit 1 ;;
esac

if [ "$DRY_RUN" != "1" ] && [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit:    ${GIT_COMMIT}"
echo "Python:        ${PYTHON}"
echo "Phases:        ${PHASES}"
echo "Conditions:    ${CONDITIONS}"
echo "Arms:          ${ARMS}"
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

has_phase() {
    for p in $PHASES; do [ "$p" = "$1" ] && return 0; done
    return 1
}

momentum_values() {
    case "${MOMENTUM_SET}" in
        off)  echo "${MOM_OFF}" ;;
        on)   echo "${MOM_ON}"  ;;
        both) echo "${MOM_OFF} ${MOM_ON}" ;;
    esac
}

mom_suffix() {
    if [ "$1" = "${MOM_ON}" ]; then echo "_M"; else echo ""; fi
}

mom_dir() {
    if [ "$1" = "${MOM_ON}" ]; then echo "mom_on"; else echo "mom_off"; fi
}

anchor_dir()  { echo "${ANCHOR_ROOT}/$(mom_dir "$1")/seed_$2"; }
anchor_ckpt() { echo "$(anchor_dir "$1" "$2")/checkpoints/model_task_00.pt"; }

# ──────────────────────────────────────────────────────────────────────────────
# Phase 0 — task-0 anchors
# ──────────────────────────────────────────────────────────────────────────────
#
# One task-0 run per (momentum, seed) at the standard operating point
# (eta = 0.1, one epoch): this is theta*_0 as ordinary training produces it,
# and every rung of the ladder starts from it.  Single-task runs
# (num_tasks=1), so no stability gap is measured and there is nothing to
# aggregate — tracking is csv_only to keep 10 bookkeeping runs out of W&B.
#
# The buffer these runs save is never used: each ladder cell refills its own
# via end_task() at its own memory.total_budget.  Reservoir sampling with
# budget k over a stream of n yields a uniform random k-subset either way.

if has_phase anchors; then
    echo ""
    echo "=================================================================="
    echo "  Phase 0 — task-0 anchors (eta=0.1, 1 epoch)"
    echo "=================================================================="

    for mu in $(momentum_values); do
        for seed in "${SEED_ARRAY[@]}"; do
            ckpt="$(anchor_ckpt "${mu}" "${seed}")"
            if [ -f "${ckpt}" ]; then
                echo "  = anchor exists, skipping: ${ckpt}"
                continue
            fi
            echo "=== anchor: mu=${mu} seed=${seed} ==="
            spawn_job \
                method=er \
                method.mode=standard \
                dataset=rot_mnist \
                dataset.num_tasks=1 \
                'dataset.rotations_deg=[0]' \
                training.lr=0.1 \
                training.epochs_per_task=1 \
                "training.momentum=${mu}" \
                memory.total_budget=1000 \
                "seed=${seed}" \
                tracking.backend=csv_only \
                "hydra.run.dir=$(anchor_dir "${mu}" "${seed}")"
        done
    done
    wait_block "anchors"

    if [ "$DRY_RUN" != "1" ]; then
        for mu in $(momentum_values); do
            for seed in "${SEED_ARRAY[@]}"; do
                ckpt="$(anchor_ckpt "${mu}" "${seed}")"
                [ -f "${ckpt}" ] || { echo "ERROR: anchor missing: ${ckpt}" >&2; exit 1; }
            done
        done
        echo "  All anchors present."
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — the ladder
# ──────────────────────────────────────────────────────────────────────────────

if has_phase ladder; then
    for mu in $(momentum_values); do
        for arm in $ARMS; do
            echo ""
            echo "=================================================================="
            echo "  Ladder — momentum=${mu}, buffer arm=${arm}"
            echo "=================================================================="

            for cond in $CONDITIONS; do
                read -r eta c every_n <<< "$(rung_spec "${cond}")"
                window=$(( 250 * c ))
                ablation_value="$(rung_tag "${cond}")$(arm_suffix "${arm}")$(mom_suffix "${mu}")"
                tags_csv="${ABLATION_KEY},${cond},${arm},$(mom_dir "${mu}")"

                echo "=== ${ablation_value}: eta=${eta}, ${c} epochs, dense eval every ${c} steps ==="

                for seed in "${SEED_ARRAY[@]}"; do
                    ckpt="$(anchor_ckpt "${mu}" "${seed}")"
                    if [ "$DRY_RUN" != "1" ] && [ ! -f "${ckpt}" ]; then
                        echo "ERROR: anchor missing for mu=${mu} seed=${seed}: ${ckpt}" >&2
                        echo "       run PHASES=anchors first." >&2
                        exit 1
                    fi
                    # arm_overrides is an intentionally-split word list.
                    # shellcheck disable=SC2046
                    spawn_job \
                        method=er \
                        method.mode=standard \
                        "${DATASET_OVERRIDES[@]}" \
                        $(arm_overrides "${arm}") \
                        "training.lr=${eta}" \
                        "training.epochs_per_task=${c}" \
                        "training.momentum=${mu}" \
                        "eval.eval_every_n_steps=${every_n}" \
                        "eval.stability_gap.eval_freq_steps=${c}" \
                        "eval.stability_gap.window_steps=${window}" \
                        "init.checkpoint=${ckpt}" \
                        init.skip_tasks=1 \
                        "seed=${seed}" \
                        "+ablation_key=${ABLATION_KEY}" \
                        "+ablation_value=${ablation_value}" \
                        "tracking.wandb.tags=[${tags_csv}]"
                done
                wait_block "${ablation_value}"
            done
        done
    done
fi

echo ""
echo "=== S-series learning-rate ladder complete ==="
echo "Next: $PYTHON scripts/aggregate_results.py"
