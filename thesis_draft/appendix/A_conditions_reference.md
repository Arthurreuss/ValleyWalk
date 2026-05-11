# Appendix A — Conditions Reference

This appendix gives the complete experimental matrix in one place — every condition referenced in §4.3–§4.7 with its Hydra-config overrides. All conditions are run with five seeds (`seed ∈ {1, 2, 3, 4, 5}`) and the default training protocol from §4.1.3 unless explicitly overridden.

## A.1 Decomposition Conditions (§4.4)

Launched via `scripts/run_decomposition.sh`. Buffer-fidelity diagnostics (`eval.stability_gap.grad_diagnostics.enabled=true`) are enabled for every G-condition; NCL has no replay buffer so the toggle is a silent no-op.

| Label | Method | Mode | Buffer | Replay batch | Normalisation | Reference |
|-------|--------|------|--------|--------------|----------------|-----------|
| G1 | `er` | standard | 1 k reservoir | matches current (256) | — | total gap |
| G2 | `er` | standard | 60 k (full past task) | full buffer (each sample once, no sampling) | — | − estimator noise |
| G3 | `er` | balanced | 1 k reservoir | matches current (256) | per-component + joint | − magnitude asymmetry |
| G4 | `er` | balanced | 60 k (full past task) | full buffer (each sample once, no sampling) | per-component + joint | − magnitude + estimator |
| NCL | `ncl` | — | none | — | K-FAC precision preconditioning | path-finding reference |

Each row is repeated at `training.momentum=0.0` and `training.momentum=0.9` (the §4.6 momentum cross). Momentum-on labels carry a `_M` suffix in `ablation_value`.

Total: 5 base × 2 momentum × 5 seeds = **50 runs**.

### A.1.1 Full Hydra Overrides

| Label | Hydra overrides (excluding seed / momentum / dataset / eval) |
|-------|---------------------------------------------------------------|
| G1 | `method=er method.mode=standard eval.stability_gap.grad_diagnostics.enabled=true` |
| G2 | `method=er method.mode=standard method.replay_full_buffer=true memory.total_budget=60000 eval.stability_gap.grad_diagnostics.enabled=true` |
| G3 | `method=er method.mode=balanced method.grad_balance.normalize_components=true method.grad_balance.task_weighted=false eval.stability_gap.grad_diagnostics.enabled=true` |
| G4 | `method=er method.mode=balanced method.replay_full_buffer=true memory.total_budget=60000 method.grad_balance.normalize_components=true method.grad_balance.task_weighted=false eval.stability_gap.grad_diagnostics.enabled=true` |
| NCL | `method=ncl` |

Common to every row: `dataset=rot_mnist dataset.num_tasks=2 dataset.rotations_deg=[0,90] eval.stability_gap.eval_freq_steps=1 eval.stability_gap.window_steps=500 +ablation_key=decomposition`.

## A.2 Curriculum Conditions (§4.5)

Launched via `scripts/run_curriculum.sh` with `BLOCK=rot_mnist`. All curriculum conditions are applied on top of standard ER (1 k reservoir, no balancing). Buffer-fidelity diagnostics are *off* by default — the curriculum claims rest on gap_depth and ACC, not on per-step gradient geometry.

| Label | Schedule | Ramp N | EMA α | λ_min | Tests |
|-------|----------|--------|-------|-------|-------|
| C1 | linear | 50 | — | 0.0 | RQ1 |
| C2 | linear | 100 | — | 0.0 | RQ1 |
| C3 | linear | 200 | — | 0.0 | RQ1 |
| C4 | adaptive | — | 0.05 | 0.0 | RQ2 |
| C5 | adaptive | — | 0.05 | 0.05 | RQ4 |
| C6 | adaptive | — | 0.05 | 0.10 | RQ4 |
| C7 | adaptive | — | 0.05 | 0.20 | RQ4 |

Each row is repeated at `training.momentum=0.0` and `training.momentum=0.9` (the §4.6 momentum cross). Momentum-on labels carry a `_M` suffix.

Total: 7 base × 2 momentum × 5 seeds = **70 runs**.

The RQ3 falsifier (momentum alone, no curriculum) is the `G1_M` run from §A.1 — vanilla ER at momentum 0.9. No separate condition is needed in this block.

### A.2.1 Full Hydra Overrides

Common to every row: `method=er method.mode=standard dataset=rot_mnist dataset.num_tasks=2 dataset.rotations_deg=[0,90] eval.stability_gap.eval_freq_steps=1 eval.stability_gap.window_steps=500 +ablation_key=curriculum method.lambda_curriculum.enabled=true`.

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

Launched via `scripts/run_curriculum.sh` with `BLOCK=cifar10`. All four conditions are run at `training.momentum=0.9` only (the headline configuration carried over from the rot-MNIST sweep). No momentum cross, no decomposition variants — see Appendix C for the rationale and the deferred-experiment list.

| Label | Method | Curriculum | Backbone |
|-------|--------|------------|----------|
| D1 | `er` | off (vanilla ER) | ResNet-18 |
| D2 | `ncl` | — | ResNet-18 |
| D3 | `er` | linear, N = best from C-series (default 200) | ResNet-18 |
| D4 | `er` | adaptive, EMA α = 0.05 | ResNet-18 |

Total: 4 conditions × 1 momentum × 5 seeds = **20 runs**.

### A.3.1 Full Hydra Overrides

Common: `dataset=dom_cifar10 model=resnet18 training.momentum=0.9 eval.stability_gap.eval_freq_steps=1 eval.stability_gap.window_steps=500 +ablation_key=curriculum_cifar`.

| Label | Method-specific overrides |
|-------|---------------------------|
| D1 | `method=er method.mode=standard` |
| D2 | `method=ncl` |
| D3 | `method=er method.mode=standard method.lambda_curriculum.enabled=true method.lambda_curriculum.schedule=linear method.lambda_curriculum.ramp_steps=${BEST_N}` |
| D4 | `method=er method.mode=standard method.lambda_curriculum.enabled=true method.lambda_curriculum.schedule=adaptive method.lambda_curriculum.ema_alpha=0.05` |

`BEST_N` is set via env-var override after the rot-MNIST sweep identifies the best linear curriculum length.

## A.4 Aggregate Run Counts

| Block | Runs |
|-------|------|
| §A.1 — Decomposition (G-series, ±momentum) | 50 |
| §A.2 — Curriculum (C-series, ±momentum) | 70 |
| §A.3 — CIFAR-10 generalisation (D-series) | 20 |
| **Total** | **140** |

## A.5 Common Training Protocol

Every condition above inherits the protocol below unless explicitly overridden.

| Key | Value |
|-----|-------|
| `training.epochs_per_task` | 1 (online regime) |
| `training.batch_size` | 256 |
| `training.optimizer` | `sgd` |
| `training.lr` | 0.1 |
| `training.weight_decay` | 0.0 |
| `memory.sampling` | `reservoir` (Vitter's Algorithm R) |
| `memory.total_budget` | 1000 (overridden to 60000 for G2/G4) |
| `eval.eval_every_n_steps` | 10 |
| `eval.stability_gap.enabled` | `true` |
| `eval.stability_gap.eval_freq_steps` | 1 (dense per-step eval in the boundary window) |
| `eval.stability_gap.window_steps` | 500 (length of the dense window after each task transition) |
| `eval.stability_gap.pre_switch_steps` | 10 (dense eval also active for the final 10 steps of each task) |
| `eval.stability_gap.grad_diagnostics.enabled` | `false` by default, `true` for the G-series (`scripts/run_decomposition.sh`); when true, g_true is computed on the full past-task training set (no sub-sampling). |
