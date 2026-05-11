#!/usr/bin/env bash
# scripts/run_ablations.sh
#
# Ablation sweep launcher for the CACL experiment.
#
# Blocks run sequentially (A1 → A2 → A3a → A3b → A4 → B1 → B2 → B3).
# Within each block all jobs (seed × dataset × parameter combo) run in
# parallel as independent Python processes — no joblib, no pickling.
#
# Blocks:
#   A1  – grad_balance: normalize_components × task_weighted (4 combos)
#   A2  – cone angle sweep: alpha_deg ∈ {0, 15, 30, 45, 60, 75, 90}
#   A3  – Lanczos: k sweep (d fixed) + d sweep (k fixed)
#   A4  – trust region: enabled ∈ {true, false}
#   B1  – baselines: GEM, A-GEM, NCL
#   B2  – ER standard mode
#   B3  – ER balanced mode (all 4 grad_balance combos)
#
# Each block uses all non-swept params at their config defaults.
# hessian.target is always replay (the default; never varied).
#
# Usage:
#   bash scripts/run_ablations.sh                          # run everything
#   ABLATIONS="A1 A2"    bash scripts/run_ablations.sh    # subset of blocks
#   DATASETS=rot_mnist   bash scripts/run_ablations.sh    # single dataset (default)
#   DATASETS="rot_mnist,dom_cifar10" bash scripts/run_ablations.sh
#   N_JOBS=4             bash scripts/run_ablations.sh    # cap parallel workers per block
#   DRY_RUN=1            bash scripts/run_ablations.sh    # print commands, don't run

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────

# Max parallel jobs per block. -1 = unlimited (all at once).
N_JOBS="${N_JOBS:--1}"

ALL_ABLATIONS="A1 A2 A3 A4 B"
ABLATIONS="${ABLATIONS:-$ALL_ABLATIONS}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-6942069}"
DATASETS="${DATASETS:-dom_cifar10}"

# Resolve project root (the directory containing this script's parent).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the venv Python if present, otherwise fall back to whatever is on PATH.
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="python"; fi

# Ensure src/ is importable.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Parse comma-separated seeds/datasets into arrays.
IFS=',' read -ra SEED_ARRAY   <<< "$SEEDS"
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: dirty git working tree — commit or stash all changes before" \
         "launching the sweep." >&2
    echo "       Dirty files:" >&2
    git status --porcelain >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
echo "Git commit: ${GIT_COMMIT}"
echo "Python:     ${PYTHON}"
echo "Ablations:  ${ABLATIONS}"
echo "Datasets:   ${DATASETS}"
echo "Seeds:      ${SEEDS}"
echo "N_JOBS:     ${N_JOBS}"
echo "Dry run:    ${DRY_RUN}"
echo "---"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_PIDS=()

# Spawn a single training job in the background.
# If N_JOBS != -1, blocks until a worker slot is free before spawning.
spawn_job() {
    echo "  + $PYTHON scripts/train.py $*"
    if [ "$DRY_RUN" = "1" ]; then return; fi

    # Throttle: wait for the oldest job if we've hit the cap.
    if [ "$N_JOBS" != "-1" ] && [ "${#BLOCK_PIDS[@]}" -ge "$N_JOBS" ]; then
        wait "${BLOCK_PIDS[0]}" || true
        BLOCK_PIDS=("${BLOCK_PIDS[@]:1}")
    fi

    "$PYTHON" scripts/train.py "$@" &
    BLOCK_PIDS+=($!)
}

# Wait for all spawned jobs in the current block; exit on any failure.
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
    for a in $ABLATIONS; do
        [ "$a" = "$label" ] && return 0
    done
    return 1
}

# ──────────────────────────────────────────────────────────────────────────────
# A1: Grad balance combinations
#
# Sweeps normalize_components × task_weighted (all 4 combos).
# All other params at defaults (alpha_deg=45, k=15, d=5, trust_region=false).
# ──────────────────────────────────────────────────────────────────────────────

if should_run A1; then
    echo "=== A1: Grad balance combinations ==="
    for norm in true false; do
        for weighted in true false; do
            for seed in "${SEED_ARRAY[@]}"; do
                for ds in "${DATASET_ARRAY[@]}"; do
                    spawn_job \
                        method=cacl \
                        "method.grad_balance.normalize_components=${norm}" \
                        "method.grad_balance.task_weighted=${weighted}" \
                        "dataset=${ds}" \
                        "seed=${seed}" \
                        +ablation_key=A1_grad_balance \
                        "+ablation_value=norm_${norm}_w_${weighted}"
                done
            done
        done
    done
    wait_block A1
fi

# ──────────────────────────────────────────────────────────────────────────────
# A2: Cone angle sweep
#
# Varies alpha_deg ∈ {0, 15, 30, 45, 60, 75, 90}.
# All other params at defaults.
# ──────────────────────────────────────────────────────────────────────────────

if should_run A2; then
    echo "=== A2: Cone angle sweep ==="
    for alpha in 0 15 30 45 60 75 90; do
        for seed in "${SEED_ARRAY[@]}"; do
            for ds in "${DATASET_ARRAY[@]}"; do
                spawn_job \
                    method=cacl \
                    "method.cone.alpha_deg=${alpha}" \
                    "dataset=${ds}" \
                    "seed=${seed}" \
                    +ablation_key=A2_cone_sweep \
                    "+ablation_value=alpha_${alpha}"
            done
        done
    done
    wait_block A2
fi

# ──────────────────────────────────────────────────────────────────────────────
# A3: Lanczos setup sweep
#
# A3a: k sweep — k ∈ {5, 10, 15, 20}, d=5 fixed (default)
# A3b: d sweep — d ∈ {3, 5, 10}, k=15 fixed (default)
# All other params at defaults.
# ──────────────────────────────────────────────────────────────────────────────

if should_run A3; then
    echo "=== A3a: Lanczos k sweep ==="
    for k in 5 10 15 20; do
        for seed in "${SEED_ARRAY[@]}"; do
            for ds in "${DATASET_ARRAY[@]}"; do
                spawn_job \
                    method=cacl \
                    "method.lanczos.k=${k}" \
                    method.lanczos.d=5 \
                    "dataset=${ds}" \
                    "seed=${seed}" \
                    +ablation_key=A3a_lanczos_k \
                    "+ablation_value=k_${k}"
            done
        done
    done
    wait_block A3a

    echo "=== A3b: Lanczos d sweep ==="
    for d in 3 5 10; do
        for seed in "${SEED_ARRAY[@]}"; do
            for ds in "${DATASET_ARRAY[@]}"; do
                spawn_job \
                    method=cacl \
                    method.lanczos.k=15 \
                    "method.lanczos.d=${d}" \
                    "dataset=${ds}" \
                    "seed=${seed}" \
                    +ablation_key=A3b_lanczos_d \
                    "+ablation_value=d_${d}"
            done
        done
    done
    wait_block A3b
fi

# ──────────────────────────────────────────────────────────────────────────────
# A4: Trust region on/off
#
# hessian.target stays replay (the default; not varied).
# All other params at defaults.
# ──────────────────────────────────────────────────────────────────────────────

if should_run A4; then
    echo "=== A4: Trust region on/off ==="
    for tr in true false; do
        for seed in "${SEED_ARRAY[@]}"; do
            for ds in "${DATASET_ARRAY[@]}"; do
                spawn_job \
                    method=cacl \
                    "method.trust_region.enabled=${tr}" \
                    "dataset=${ds}" \
                    "seed=${seed}" \
                    +ablation_key=A4_trust_region \
                    "+ablation_value=tr_${tr}"
            done
        done
    done
    wait_block A4
fi

# ──────────────────────────────────────────────────────────────────────────────
# B: Baselines
#
# B1: GEM, A-GEM, NCL — no extra knobs.
# B2: ER standard mode (single combined forward/backward).
# B3: ER balanced mode — all 4 grad_balance combos (normalize × task_weighted).
# ──────────────────────────────────────────────────────────────────────────────

if should_run B; then
    echo "=== B1: Baselines (GEM, A-GEM, NCL) ==="
    for method in gem agem ncl; do
        for seed in "${SEED_ARRAY[@]}"; do
            for ds in "${DATASET_ARRAY[@]}"; do
                spawn_job \
                    "method=${method}" \
                    "dataset=${ds}" \
                    "seed=${seed}" \
                    +ablation_key=baseline \
                    "+ablation_value=${method}"
            done
        done
    done
    wait_block B1

    echo "=== B2: ER standard ==="
    for seed in "${SEED_ARRAY[@]}"; do
        for ds in "${DATASET_ARRAY[@]}"; do
            spawn_job \
                method=er \
                method.mode=standard \
                "dataset=${ds}" \
                "seed=${seed}" \
                +ablation_key=baseline_er \
                +ablation_value=er_standard
        done
    done
    wait_block B2

    echo "=== B3: ER balanced (grad balance combos) ==="
    for norm in true false; do
        for weighted in true false; do
            for seed in "${SEED_ARRAY[@]}"; do
                for ds in "${DATASET_ARRAY[@]}"; do
                    spawn_job \
                        method=er \
                        method.mode=balanced \
                        "method.grad_balance.normalize_components=${norm}" \
                        "method.grad_balance.task_weighted=${weighted}" \
                        "dataset=${ds}" \
                        "seed=${seed}" \
                        +ablation_key=baseline_er_balanced \
                        "+ablation_value=er_bal_norm_${norm}_w_${weighted}"
                done
            done
        done
    done
    wait_block B3
fi

echo ""
echo "=== Sweep complete ==="
