#!/bin/bash
# scripts/habrok_cost.slurm
#
# Computational-cost benchmark on Habrok: curriculum vs asymmetric PER
# (plus the vanilla-ER baseline) on rot-MNIST and CIFAR.
#
# One SLURM array task == one (dataset, family, seed) == one train.py run on
# one GPU, wrapped in resource measurement:
#
#   - wall-clock time, user/sys CPU time, peak host RAM   (GNU time -v)
#   - peak GPU memory, mean GPU utilisation, mean power,
#     estimated GPU energy                                 (nvidia-smi sampler)
#
# Every task appends one row to  outputs/cost/cost_summary.csv  (flock-safe);
# raw time/nvidia-smi logs land in  outputs/cost/raw/.
#
# ─── CONDITIONS ─────────────────────────────────────────────────────────────
# Overrides MIRROR the real experiment scripts and must be kept in sync:
#
#   rotmnist  — 2-task rot-MNIST [0,90] (one transition), MLP, µ=0,
#               1k reservoir buffer (config default)     (run_C.sh / run_P.sh)
#   cifar     — 2-task CIFAR [none,gaussian_noise], ResNet-18 GN, buffer 5k,
#               µ=0.9, 10 epochs/task                    (run_G.sh: G6/G8)
#
#   curriculum  — adaptive λ-curriculum, λ_min=0.20
#   per         — asymmetric PER, δ=0.1, CG 25 iters, warm start
#   vanilla     — plain ER, no gate (opt-in via FAMILIES, off by default)
#
# ONE deliberate deviation from the real scripts: continual evaluation is
# disabled (no stability-gap tracker, no periodic eval), so wall/CPU/GPU
# numbers measure the methods' training compute only.  Safe: the adaptive
# λ reads a train-step gradient-norm EMA (src/methods/er.py), never the
# eval loop, so disabling eval changes neither method's trajectory.
#
# Benchmark runs are tagged +ablation_key=cost and use tracking.backend=
# csv_only, so they never touch W&B and never mix with real aggregates.
#
# ─── USAGE ──────────────────────────────────────────────────────────────────
# Submit FROM THE REPO ROOT.
#
#   0) Preview the task list + count WITHOUT SLURM (sets --array):
#        bash scripts/habrok_cost.slurm
#        SEEDS="1" bash scripts/habrok_cost.slurm
#
#   1) ONE-TIME on the login node, pre-download CIFAR-10 (guard below is a
#      fallback):
#        python -c "import torchvision as t; \
#          t.datasets.CIFAR10('data', train=True,  download=True); \
#          t.datasets.CIFAR10('data', train=False, download=True)"
#
#   2) Launch (default 2 datasets × 2 families × 5 seeds = 20 tasks):
#        sbatch --array=1-20 scripts/habrok_cost.slurm
#      …or subsets:
#        SEEDS="1"           sbatch --array=1-4  scripts/habrok_cost.slurm
#        DATASETS="cifar"    sbatch --array=1-10 scripts/habrok_cost.slurm
#        FAMILIES="vanilla curriculum per" \
#                            sbatch --array=1-30 scripts/habrok_cost.slurm
#      Env vars must reach sbatch: prefix them (VAR=... sbatch ...) or use
#      sbatch --export=ALL,VAR=... .  --array must match the preview count.
#
#   3) After the array finishes, summarise (mean ± sd per condition):
#        SUMMARIZE=1 bash scripts/habrok_cost.slurm
#      and cross-check against SLURM accounting:
#        sacct -j <jobid> --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,State
#
# NOTE: for a fair curriculum-vs-PER comparison keep all tasks on the same
# partition/GPU type (pinned below: one a100).  The rot-MNIST MLP badly
# under-utilises even an A100 — that is fine, the point is relative cost on
# identical hardware.  (A100 is compute capability 8.0, which the installed
# torch build supports; the V100 is CC 7.0 and the default cu128 wheel has no
# sm_70 kernels, so conv2d fails with "GET was unable to find an engine".)
#SBATCH --job-name=vw_cost
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cost_%A_%a.out
#SBATCH --error=logs/cost_%A_%a.err

set -euo pipefail

DATASETS="${DATASETS:-rotmnist cifar}"
FAMILIES="${FAMILIES:-curriculum per}"
SEEDS="${SEEDS:-1 2 3 4 5}"

COST_DIR="outputs/cost"
SUMMARY_CSV="${COST_DIR}/cost_summary.csv"
GPU_SAMPLE_SEC=2

# ─── Summary mode (no SLURM) ──────────────────────────────────────────────
if [ "${SUMMARIZE:-0}" = "1" ]; then
    PYTHON="${PYTHON:-.venv/bin/python}"; [ -x "$PYTHON" ] || PYTHON=python3
    CSV="$SUMMARY_CSV" "$PYTHON" - <<'PY'
import csv, os, statistics as st
path = os.environ["CSV"]
rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
if not rows:
    raise SystemExit(f"no successful rows in {path}")
keys = sorted({(r["dataset"], r["family"]) for r in rows})
cols = [("wall_s", "wall [s]"), ("cpu_total_s", "CPU [s]"),
        ("max_rss_mb", "RAM [MB]"), ("gpu_peak_mem_mb", "GPUmem [MB]"),
        ("gpu_mean_util_pct", "util [%]"), ("gpu_energy_wh", "GPU [Wh]")]
hdr = f"{'dataset':<10} {'family':<11} {'n':>2} " + " ".join(f"{h:>22}" for _, h in cols)
print(hdr); print("-" * len(hdr))
for ds, fam in keys:
    grp = [r for r in rows if (r["dataset"], r["family"]) == (ds, fam)]
    cells = []
    for c, _ in cols:
        vals = [float(r[c]) for r in grp if r[c] not in ("", "nan")]
        cells.append(f"{st.mean(vals):>12.1f} ±{(st.stdev(vals) if len(vals) > 1 else 0.0):>7.1f}"
                     if vals else f"{'—':>21}")
    print(f"{ds:<10} {fam:<11} {len(grp):>2} " + " ".join(cells))
PY
    exit 0
fi

# ─── Task list ────────────────────────────────────────────────────────────
TASKS=()
for ds in $DATASETS; do for fam in $FAMILIES; do for s in $SEEDS; do
    TASKS+=("$ds|$fam|$s")
done; done; done
NTASKS="${#TASKS[@]}"

# ─── Preview mode (no SLURM) ──────────────────────────────────────────────
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "DATASETS = ${DATASETS}   FAMILIES = ${FAMILIES}   SEEDS = ${SEEDS}"
    echo "${NTASKS} array tasks  ->  submit with  --array=1-${NTASKS}"
    echo "---"
    i=1
    for t in "${TASKS[@]}"; do
        IFS='|' read -r ds fam s <<< "$t"
        printf "  %3d  %-9s %-11s seed=%s\n" "$i" "$ds" "$fam" "$s"
        i=$((i+1))
    done
    exit 0
fi

# ─── Environment ──────────────────────────────────────────────────────────
module purge
export PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:${PYTHONPATH}}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data "$COST_DIR/raw"

# ─── Select this array task ───────────────────────────────────────────────
idx=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$idx" -lt 0 ] || [ "$idx" -ge "$NTASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (1..$NTASKS)" >&2
    exit 1
fi
IFS='|' read -r DS FAM SEED <<< "${TASKS[$idx]}"
LABEL="${DS}_${FAM}_s${SEED}"
RAW="$COST_DIR/raw/${LABEL}_${SLURM_ARRAY_JOB_ID}"
TIME_LOG="${RAW}.time"
GPU_LOG="${RAW}.gpu.csv"

echo "=========================================================="
echo " Array task : $SLURM_ARRAY_TASK_ID / $NTASKS"
echo " Benchmark  : dataset=$DS  family=$FAM  seed=$SEED"
echo " Node       : $(hostname)"
echo " Python     : $PYTHON"
echo "=========================================================="

# ─── Hydra overrides (KEEP IN SYNC with run_L.sh / run_G.sh) ──────────────
OVERRIDES=()
case "$DS" in
    rotmnist)  # run_C.sh / run_P.sh single-transition setup, µ=0,
               # 1k reservoir buffer (config default — no override)
        OVERRIDES+=(
            dataset=rot_mnist
            dataset.num_tasks=2
            'dataset.rotations_deg=[0,90]'
            training.momentum=0.0
        ) ;;
    cifar)     # run_G.sh headline protocol, GN legs
               # (CIFAR_BASE + TWO_TASK_GAUSSIAN + GN_OVERRIDES → G6/G8)
        OVERRIDES+=(
            dataset=dom_cifar10
            model=resnet18
            training.epochs_per_task=10
            dataset.num_tasks=2
            'dataset.corruption_types=[none,gaussian_noise]'
            model.norm_type=group
            model.norm_groups=32
            memory.total_budget=5000
            training.momentum=0.9
        ) ;;
    *) echo "ERROR: unknown dataset '$DS' (expected rotmnist|cifar)" >&2; exit 1 ;;
esac

case "$FAM" in
    vanilla)     # C-series baseline / G4 (opt-in, not in default FAMILIES)
        OVERRIDES+=(method=er method.mode=standard) ;;
    curriculum)  # C7 / G6 — adaptive λ-curriculum, λ_min=0.20
        OVERRIDES+=(
            method=er
            method.mode=standard
            method.lambda_curriculum.enabled=true
            method.lambda_curriculum.schedule=adaptive
            method.lambda_curriculum.ema_alpha=0.05
            method.lambda_curriculum.lambda_min=0.20
        ) ;;
    per)         # P-series δ=0.1 / G8 — asym PER, warm-started CG
        OVERRIDES+=(
            method=precond_er
            method.apply_to=current
            method.fisher.target=replay
            method.fisher.damping=0.1
            "method.cg.iters=${PER_CG_ITERS:-25}"
            method.cg.warm_start=true
        ) ;;
    *) echo "ERROR: unknown family '$FAM' (expected vanilla|curriculum|per)" >&2; exit 1 ;;
esac

# Continual evaluation OFF — the real runs' dense per-step eval is the same
# additive load for both methods, but it dilutes the measured method
# overhead, so cost runs strip it: no stability-gap tracker, and the
# periodic-eval interval pushed past any run length (a literal 0 would
# crash — train.py takes `global_step % eval_every_n_steps` every step).
OVERRIDES+=(
    eval.stability_gap.enabled=false
    eval.eval_every_n_steps=1000000000
    "seed=${SEED}"
    +ablation_key=cost
    "+ablation_value=cost_${DS}_${FAM}"
    tracking.backend=csv_only
)

# ─── CIFAR-10 download guard (fallback; serialised across array tasks) ────
if [ "$DS" = "cifar" ]; then
    (
        flock -x 200
        "$PYTHON" - <<'PY'
import torchvision as t
t.datasets.CIFAR10("data", train=True,  download=True)
t.datasets.CIFAR10("data", train=False, download=True)
PY
    ) 200>data/.cifar_download.lock
fi

# ─── GPU sampler ──────────────────────────────────────────────────────────
SAMPLER_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits -l "$GPU_SAMPLE_SEC" > "$GPU_LOG" &
    SAMPLER_PID=$!
else
    GPU_NAME="none"
    : > "$GPU_LOG"
fi
cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ─── Measured run ─────────────────────────────────────────────────────────
echo "  + train.py ${OVERRIDES[*]}"
set +e
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v -o "$TIME_LOG" "$PYTHON" scripts/train.py "${OVERRIDES[@]}"
    RC=$?
else
    echo "WARNING: /usr/bin/time not found — wall/CPU/RSS fields will be empty" >&2
    : > "$TIME_LOG"
    "$PYTHON" scripts/train.py "${OVERRIDES[@]}"
    RC=$?
fi
set -e
cleanup; SAMPLER_PID=""

STATUS=ok; [ "$RC" -ne 0 ] && STATUS="failed(rc=$RC)"
echo "run finished with status: $STATUS"

# ─── Parse + append summary row (flock-safe) ──────────────────────────────
DS="$DS" FAM="$FAM" SEED="$SEED" STATUS="$STATUS" GPU_NAME="$GPU_NAME" \
TIME_LOG="$TIME_LOG" GPU_LOG="$GPU_LOG" SUMMARY_CSV="$SUMMARY_CSV" \
GPU_SAMPLE_SEC="$GPU_SAMPLE_SEC" \
GIT_COMMIT="$(git rev-parse HEAD)" \
"$PYTHON" - <<'PY'
import csv, datetime, fcntl, os, socket

env = os.environ
row = {
    "dataset": env["DS"], "family": env["FAM"], "seed": env["SEED"],
    "status": env["STATUS"], "gpu": env["GPU_NAME"],
    "host": socket.gethostname(),
    "slurm_job": f'{env.get("SLURM_ARRAY_JOB_ID","")}_{env.get("SLURM_ARRAY_TASK_ID","")}',
    "git_commit": env["GIT_COMMIT"][:10],
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
}

# GNU time -v
wall = user = sys_ = rss_kb = None
for line in open(env["TIME_LOG"]):
    line = line.strip()
    if line.startswith("Elapsed (wall clock)"):
        p = line.split()[-1].split(":")
        wall = float(p[-1]) + 60 * int(p[-2]) + (3600 * int(p[0]) if len(p) == 3 else 0)
    elif line.startswith("User time (seconds):"):
        user = float(line.split()[-1])
    elif line.startswith("System time (seconds):"):
        sys_ = float(line.split()[-1])
    elif line.startswith("Maximum resident set size"):
        rss_kb = float(line.split()[-1])
row["wall_s"] = f"{wall:.1f}" if wall is not None else ""
row["cpu_user_s"] = f"{user:.1f}" if user is not None else ""
row["cpu_sys_s"] = f"{sys_:.1f}" if sys_ is not None else ""
row["cpu_total_s"] = f"{user + sys_:.1f}" if None not in (user, sys_) else ""
row["max_rss_mb"] = f"{rss_kb / 1024:.0f}" if rss_kb is not None else ""

# nvidia-smi sampler: util %, mem MiB, power W
util, mem, pwr = [], [], []
for line in open(env["GPU_LOG"]):
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        util.append(float(parts[0])); mem.append(float(parts[1])); pwr.append(float(parts[2]))
    except ValueError:
        continue  # "[N/A]" etc.
row["gpu_peak_mem_mb"] = f"{max(mem):.0f}" if mem else ""
row["gpu_mean_util_pct"] = f"{sum(util) / len(util):.1f}" if util else ""
row["gpu_mean_power_w"] = f"{sum(pwr) / len(pwr):.1f}" if pwr else ""
row["gpu_energy_wh"] = (f"{sum(pwr) * float(env['GPU_SAMPLE_SEC']) / 3600:.1f}"
                        if pwr else "")

fields = ["dataset", "family", "seed", "status", "wall_s", "cpu_user_s",
          "cpu_sys_s", "cpu_total_s", "max_rss_mb", "gpu_peak_mem_mb",
          "gpu_mean_util_pct", "gpu_mean_power_w", "gpu_energy_wh",
          "gpu", "host", "slurm_job", "git_commit", "timestamp"]
path = env["SUMMARY_CSV"]
with open(path, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    w = csv.DictWriter(f, fieldnames=fields)
    if f.tell() == 0:
        w.writeheader()
    w.writerow(row)
print("appended:", {k: row[k] for k in fields[:13]})
PY

echo "=== cost task $SLURM_ARRAY_TASK_ID ($LABEL) done: $STATUS ==="
exit "$RC"
