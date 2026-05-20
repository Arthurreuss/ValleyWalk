# Appendix A — Conditions Reference

This appendix lists the complete experimental matrix in one place — every condition referenced in §4.3–§4.7, together with its Hydra-config overrides. All conditions are launched from the two sweep scripts under `scripts/`:

- `scripts/run_decomposition.sh` — the three-contributor decomposition (§4.4), rot-MNIST, G-series + NCL reference;
- `scripts/run_curriculum.sh` — the λ-curriculum sweep on rot-MNIST (§4.5–§4.6, C-series) **and** the CIFAR-10 generalisation block (§4.7, D-series).

Every condition uses **5 seeds** (`seed ∈ {1, 2, 3, 4, 5}`) and inherits the global protocol of §4.1.3 unless explicitly overridden. The launchers refuse to run on a dirty git tree, so every result reported in Appendix E is traceable to a single commit recorded in `run_manifest.json`.

## A.1 Decomposition Conditions (§4.4)

Launched via `scripts/run_decomposition.sh`. Buffer-fidelity diagnostics (`eval.stability_gap.grad_diagnostics.enabled=true`) are enabled for every G-condition; NCL has no replay buffer so the toggle is a silent no-op.

| Label | Method | Mode | Buffer | Replay batch | Normalisation | Removed by construction |
|-------|--------|------|--------|--------------|---------------|--------------------------|
| G1 | `er` | standard | 1 k reservoir | matches current (256) | — | (nothing) |
| G2 | `er` | standard | 60 k (full past task) | full buffer (each sample once) | — | estimator noise |
| G3 | `er` | balanced | 1 k reservoir | matches current (256) | per-component + joint | magnitude asymmetry |
| G4 | `er` | balanced | 60 k (full past task) | full buffer (each sample once) | per-component + joint | magnitude + estimator |
| NCL | `ncl` | — | none (Λ-prior + accumulated K-FAC factors) | — | K-FAC precision preconditioning | — (path-finding reference; Kao et al., 2021) |

Each row is repeated at `training.momentum=0.0` and `training.momentum=0.9` (the §4.6 momentum cross). Momentum-on labels carry a `_M` suffix in `ablation_value`.

**Total: 5 base × 2 momentum × 5 seeds = 50 runs.**

### A.1.1 Full Hydra Overrides

Common to every row (rot-MNIST shared block):

```
dataset=rot_mnist
dataset.num_tasks=2
dataset.rotations_deg=[0,90]
eval.stability_gap.eval_freq_steps=1
eval.stability_gap.window_steps=500
+ablation_key=decomposition
```

| Label | Condition-specific overrides |
|-------|------------------------------|
| G1 | `method=er method.mode=standard eval.stability_gap.grad_diagnostics.enabled=true` |
| G2 | `method=er method.mode=standard method.replay_full_buffer=true memory.total_budget=60000 eval.stability_gap.grad_diagnostics.enabled=true` |
| G3 | `method=er method.mode=balanced method.grad_balance.normalize_components=true method.grad_balance.task_weighted=false eval.stability_gap.grad_diagnostics.enabled=true` |
| G4 | `method=er method.mode=balanced method.replay_full_buffer=true memory.total_budget=60000 method.grad_balance.normalize_components=true method.grad_balance.task_weighted=false eval.stability_gap.grad_diagnostics.enabled=true` |
| NCL | `method=ncl method.ncl.prior_init=0.1` |

`prior_init=0.1` is pinned on the NCL row to match the tuned default (see Appendix D for the rationale). The momentum leg is set by `training.momentum=0.0` or `training.momentum=0.9` exclusively at launch.

## A.2 λ-Curriculum Conditions on rot-MNIST (§4.5–§4.6)

Launched via `scripts/run_curriculum.sh` with `BLOCK=rot_mnist`. All curriculum conditions are applied on top of standard ER (1 k reservoir, no balancing) so that the curriculum's effect is isolated. Buffer-fidelity diagnostics are *off* by default — the curriculum claims rest on `gap_depth` and ACC, not on per-step gradient geometry, and the extra forward+backward per step would dominate the runtime of the longer sweep.

| Label | Schedule | Ramp N | EMA α | λ_min | RQ tested |
|-------|----------|--------|-------|-------|-----------|
| C1 | linear | 50 | — | 0.0 | RQ1 |
| C2 | linear | 100 | — | 0.0 | RQ1 |
| C3 | linear | 200 | — | 0.0 | RQ1 |
| C4 | adaptive | — | 0.05 | 0.0 | RQ2 |
| C5 | adaptive | — | 0.05 | 0.05 | RQ4 |
| C6 | adaptive | — | 0.05 | 0.10 | RQ4 |
| C7 | adaptive | — | 0.05 | 0.20 | RQ4 |

Each row is repeated at `training.momentum=0.0` and `training.momentum=0.9` (the §4.6 momentum cross). Momentum-on labels carry a `_M` suffix.

**Total: 7 base × 2 momentum × 5 seeds = 70 runs.**

The RQ3 falsifier (momentum alone, no curriculum) re-uses the `G1_M` run from §A.1 — vanilla ER at momentum 0.9. No separate condition is needed in this block.

### A.2.1 Full Hydra Overrides

Common to every row:

```
method=er
method.mode=standard
method.lambda_curriculum.enabled=true
dataset=rot_mnist
dataset.num_tasks=2
dataset.rotations_deg=[0,90]
eval.stability_gap.eval_freq_steps=1
eval.stability_gap.window_steps=500
+ablation_key=curriculum
```

| Label | Schedule-specific overrides |
|-------|------------------------------|
| C1 | `method.lambda_curriculum.schedule=linear method.lambda_curriculum.ramp_steps=50` |
| C2 | `method.lambda_curriculum.schedule=linear method.lambda_curriculum.ramp_steps=100` |
| C3 | `method.lambda_curriculum.schedule=linear method.lambda_curriculum.ramp_steps=200` |
| C4 | `method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.0` |
| C5 | `method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.05` |
| C6 | `method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.10` |
| C7 | `method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.20` |

## A.3 CIFAR-10 Generalisation Conditions (§4.7)

Launched via `scripts/run_curriculum.sh` with `BLOCK=cifar10`. All four conditions run at `training.momentum=0.9` only (the headline configuration carried over from the rot-MNIST sweep). No momentum cross, no decomposition variants — the CIFAR block is scoped as a *direction-of-effect* check, not a second full sweep.

| Label | Method | Curriculum | Backbone |
|-------|--------|------------|----------|
| D1 | `er` | off (vanilla ER) | ResNet-18 (CIFAR-adapted, Appendix B.1.2) |
| D2 | `ncl` | — | ResNet-18 |
| D3 | `er` | adaptive, EMA α = 0.05, λ_min = 0.20 | ResNet-18 |
| D4 | `er` | adaptive, EMA α = 0.05, λ_min = 0.10 | ResNet-18 |

**Total: 4 conditions × 1 momentum × 5 seeds = 20 runs.**

D3 and D4 mirror the two most successful adaptive variants of the rot-MNIST λ_min refinement (C6 and C7); a linear curriculum is *not* carried over to CIFAR because the adaptive schedule is the hyperparameter-free default that the rest of the thesis recommends.

### A.3.1 Full Hydra Overrides

Common to every row:

```
dataset=dom_cifar10
dataset.num_tasks=2
dataset.corruption_types=[none, gaussian_noise]
model=resnet18
training.epochs_per_task=10
training.momentum=0.9
eval.stability_gap.eval_freq_steps=50
eval.stability_gap.window_steps=250
+ablation_key=curriculum_cifar
```

| Label | Method-specific overrides |
|-------|---------------------------|
| D1 | `method=er method.mode=standard` |
| D2 | `method=ncl method.ncl.damping=0.001 method.ncl.fisher_samples=1000 method.ncl.prior_init=0.1 method.ncl.trust_radius=1.0` |
| D3 | `method=er method.mode=standard method.lambda_curriculum.enabled=true method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.20` |
| D4 | `method=er method.mode=standard method.lambda_curriculum.enabled=true method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05 method.lambda_curriculum.lambda_min=0.10` |

**Why the eval cadence differs on CIFAR.** rot-MNIST trains for one epoch (~235 batches) per task, so a step-1 cadence over 500 steps covers the entire transition window. CIFAR's 10-epoch protocol produces ≈ 1 950 batches per task; the post-switch dip plays out over thousands of steps but with much smaller per-step changes, so we coarsen to every 50 steps and cap the dense window at 250 steps. The coarsening is documented in §4.1.4 and is the only methodological difference between the rot-MNIST and CIFAR evaluation protocols.

## A.4 Aggregate Run Counts

| Block | Section | Runs |
|-------|---------|------|
| Decomposition (G-series, ±momentum, rot-MNIST) | §A.1 | 50 |
| λ-Curriculum (C-series, ±momentum, rot-MNIST) | §A.2 | 70 |
| CIFAR-10 generalisation (D-series, μ = 0.9 only) | §A.3 | 20 |
| **Total** | | **140** |

## A.5 Common Training Protocol

Every condition above inherits the protocol below unless explicitly overridden in §§A.1–A.3.

| Key | Value (rot-MNIST default) | CIFAR override |
|-----|---------------------------|-----------------|
| `training.epochs_per_task` | 1 (online regime) | 10 |
| `training.batch_size` | 256 | (same) |
| `training.optimizer` | `sgd` | (same) |
| `training.lr` | 0.1 | (same) |
| `training.weight_decay` | 0.0 | (same) |
| `memory.sampling` | `reservoir` (Vitter's Algorithm R) | (same) |
| `memory.total_budget` | 1 000 (overridden to 60 000 for G2 / G4) | 1 000 |
| `eval.eval_every_n_steps` | 10 | (same) |
| `eval.stability_gap.enabled` | `true` | (same) |
| `eval.stability_gap.eval_freq_steps` | 1 (dense per-step eval in the boundary window) | 50 |
| `eval.stability_gap.window_steps` | 500 (length of the dense window after each task transition) | 250 |
| `eval.stability_gap.pre_switch_steps` | 10 | (same) |
| `eval.stability_gap.grad_diagnostics.enabled` | `false` by default, `true` for the G-series only; when true, `g_true` is computed on the full past-task training set with no sub-sampling | (same; off for D-series) |

## A.6 Reproducibility Notes

- The launchers `scripts/run_decomposition.sh` and `scripts/run_curriculum.sh` refuse to run on a dirty git tree (`git status --porcelain` is checked at startup), guaranteeing that every manifest can be traced to a single commit.
- Every run writes a `run_manifest.json` containing the git SHA, the full Hydra override list, the resolved seed, the wall-clock time, and the final metric values. Appendix E enumerates one manifest path per (condition, seed).
- The launchers expose env-var overrides for selecting condition subsets (`CONDITIONS="G1 G3"`), restricting the momentum leg (`MOMENTUM_SET=off|on|both`), restricting the dataset block (`BLOCK=rot_mnist|cifar10|both`), changing the seed list (`SEEDS=1,2,3,4,5`), and capping parallelism (`N_JOBS`, `N_JOBS_CIFAR`). See the script header comments for the full list.
- Aggregation is performed by `scripts/aggregate_results.py`, which walks `outputs/`, parses every `run_manifest.json`, and produces a `master_index.csv` plus per-dataset CSV / LaTeX summary tables. Appendix E is generated from the same manifest tree; duplicate manifests for a given (condition, seed) are resolved by keeping the most recent `started_at` (older aborted runs are discarded).
