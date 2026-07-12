# ValleyWalk

Code and thesis for **"Walking the Valley: Mitigating the Stability Gap in Continual Learning"** (BSc AI thesis, University of Groningen).

When a continually-trained network switches tasks, accuracy on old tasks briefly collapses before recovering — the *stability gap*. This repo decomposes the gap into its drivers (magnitude bias, estimator bias, trajectory geometry) and evaluates two mitigation *gates* on top of experience replay:

- **λ-curriculum** — a scalar feedback gate that ramps the new-task loss weight from 0 → 1 at each task boundary.
- **Asymmetric preconditioned ER (PER)** — a directional feedforward gate: the replay Fisher preconditions only the current-task gradient, `d = δ(F_rep + δI)⁻¹ g_cur + g_rep`.

An interactive companion visualisation of the loss landscape at a task transition is published at
[arthurreuss.github.io/ValleyWalk/loss_landscape_visualization.html](https://arthurreuss.github.io/ValleyWalk/loss_landscape_visualization.html)
(served from `loss_landscape_visualization.html` at the repo root).

## Layout

```
configs/            Hydra configs (composed as method × dataset × model)
src/
  data/             Rotated-MNIST & domain-CIFAR task streams, reservoir replay buffer
  methods/          ER (+ λ-curriculum), GEM / A-GEM, NCL, asymmetric PER
  models/           MLP (rot-MNIST) and ResNet-18 (CIFAR)
  eval/             Continual metrics (ACC, FORG, WF/WP, …) and stability-gap tracker
  optim/            Fisher-vector products + conjugate gradient (used by PER)
  utils/            Seeding, dual CSV/W&B tracking, run manifests
scripts/
  train.py          Hydra entry point — one run, one seed
  run_{C,D,G,L,P}.sh  Experiment series (see below)
  habrok*.slurm     SLURM launchers for the RUG Habrok cluster
  aggregate_results.py  Collect run manifests → outputs/master_index.csv
  paired_stats.py   Seed-paired Wilcoxon + bootstrap CIs between conditions
  plotting/         Regenerates every thesis figure from outputs/
tests/              Pytest suite for methods, metrics, buffer, tracking
thesis_latex/       Thesis source; compiled PDF at thesis_latex/main.pdf
```

## Setup

Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest          # sanity check
```

## Running experiments

Single run (Hydra overrides select method / dataset / model):

```bash
uv run python scripts/train.py method=er dataset=rot_mnist model=mlp seed=1
```

Each run writes checkpoints, per-step accuracy curves, `metrics_summary.json`, and a `run_manifest.json` under `outputs/<method>/<dataset>/<timestamp>/`, and optionally logs to W&B (`tracking.backend=csv_only` disables it).

The thesis experiments are organised as series, each a self-documenting script:

| Series | Script | What it runs |
|---|---|---|
| C | `run_C.sh` | λ-curriculum sweep + momentum cross (2-task rot-MNIST) |
| D | `run_D.sh` | Three-contributor gap decomposition + momentum cross |
| G | `run_G.sh` | CIFAR-10 generalisation (headline, corruptions, 3-task) |
| L | `run_L.sh` | Both gates on the 5-task long sequence |
| P | `run_P.sh` | Damping-δ sweep for asymmetric PER |

## From runs to thesis numbers

```bash
python scripts/aggregate_results.py --run-dir outputs/   # → outputs/master_index.csv
python scripts/paired_stats.py --pair <A> <B> --metric stab_gap_depth
python -m scripts.plotting.make_all                      # figures → thesis_latex/img/results/
python -m scripts.plotting.table_compute_cost            # compute-cost table
```

`master_index.csv` is the single source of truth: every figure and table in the thesis is generated from it plus the per-run curve CSVs, traceable back to a `run_manifest.json` (git commit, config, W&B URL).
