# Appendix B — Implementation Details

This appendix documents the implementation choices behind the experiments. All references are to files under `src/` in the project repository.

## B.1 Model Architectures

### B.1.1 MLP (rot-MNIST sweep)

`src/models/mlp.py`. A two-hidden-layer fully-connected network with ReLU activations and **no regularisation layers** (no dropout, no batch-norm). The absence of normalisation is deliberate: it keeps the Hessian structure clean so that the envelope-theorem regularity assumptions of §3.5 (positive-definite, well-conditioned H_λ along the path) are plausibly approximated.

| Layer | Shape | Activation |
|-------|-------|------------|
| Input | 784 (flattened 28 × 28) | — |
| Linear | 784 → 400 | ReLU |
| Linear | 400 → 400 | ReLU |
| Linear | 400 → 10 | — (raw logits) |

Total parameters: 784 · 400 + 400 + 400 · 400 + 400 + 400 · 10 + 10 = **478,410**.

### B.1.2 ResNet-18 (CIFAR-10 generalisation)

`src/models/resnet.py`. Standard ResNet-18 with two CIFAR-specific modifications:

1. The initial 7×7 conv (stride 2) is replaced by a 3×3 conv (stride 1) so that the feature map stays at 32×32 through the stem.
2. The initial max-pool is removed for the same reason.

All other details — `BasicBlock` with batch-norm, four residual stages with widths [64, 128, 256, 512] and [2, 2, 2, 2] blocks each, global average pooling, linear classifier — match the canonical ResNet-18. The output head is 10-class for `dom_cifar10`.

## B.2 Experience Replay

`src/methods/er.py`. Two execution modes are exposed via `method.mode`:

### B.2.1 `standard` mode

Two separate backward passes — one on the current-task batch, one on a replay batch sampled from the buffer. The two flat gradients are then combined as

```
g_joint = λ(t) · g_new + g_replay
```

and written into `.grad` for the optimiser step. This is mathematically equivalent to a single backward pass on `λ · L_current + L_replay` (up to float-summation order); two passes are used so the gradient norms are available before λ is resolved, which the adaptive schedule requires.

When `lambda_curriculum.enabled=false` the schedule resolver returns `1.0` and the code path reduces to plain ER: `g_joint = g_new + g_replay`.

### B.2.2 `balanced` mode

Two backward passes as above, then a three-step normalisation:

```
g_joint = λ · w_new · (g_new / ‖g_new‖) + w_replay · (g_replay / ‖g_replay‖)
g_joint ← g_joint / ‖g_joint‖
```

with `w_new = w_replay = 0.5` (the `task_weighted=false` setting used throughout this thesis). The per-component normalisation removes the magnitude asymmetry between g_new and g_replay; the final joint normalisation removes the dependence of step size on landscape steepness. The latter is precisely the property that costs balanced ER its final-accuracy margin (§3.3).

### B.2.3 Replay batch size

Two configuration knobs control the replay batch:

- `method.replay_batch_size` — when set to `null` (default), the replay batch matches the current-task batch size (256). When set to an integer, that many samples are drawn from the buffer with replacement on every step.
- `method.replay_full_buffer` — when `true`, overrides `replay_batch_size` and uses `ReservoirBuffer.sample_all()`: every stored sample is included in the replay batch exactly once per step, with no sampling. The replay loss is the cross-entropy averaged over the entire buffer.

For G2 and G4 we set `replay_full_buffer=true` against a 60 k buffer (`memory.total_budget=60000`, equal to the MNIST per-task training-set size). Because the buffer holds the full past-task training set and every sample contributes exactly once, the resulting replay gradient is the **deterministic empirical past-task gradient** at the current parameters — zero sampling noise. Vanilla ER (G1) draws 256 samples with replacement and is therefore a stochastic estimator of the same gradient; this contrast is what defines the estimator-bias share G_est in §4.4.

### B.2.4 Buffer population

`ER.end_task(task_id, train_loader)` iterates the completed task's training loader once and calls `ReservoirBuffer.add(x, y, task_id)` on every sample individually. During T₀ training the buffer is empty by construction, so ER degrades to plain SGD for the first task; the stability gap is measurable only from T₁ onward.

## B.3 Reservoir Buffer

`src/data/memory_buffer.py`. Implements **Vitter's Algorithm R** (1985): the k-th arriving sample is placed in a uniformly random buffer slot with probability `budget / k`. This guarantees that at any point each seen sample has equal probability `budget / n_seen` of currently occupying a slot, regardless of arrival order or task structure.

`buffer.sample(n)` draws n indices uniformly *with replacement* and returns the corresponding (x, y, task_id) batch. The dependence on Python's `random` (not torch) is relevant for any test that needs reproducible buffer draws — seed `random` explicitly or patch `sample` directly.

`buffer.sample_all()` returns every currently-stored sample exactly once (in insertion order, no shuffling) — used by the `replay_full_buffer=true` path of ER (§B.2.3) to compute the exact empirical past-task gradient. Because cross-entropy on a batch is a permutation-invariant average, the lack of shuffling has no effect on the resulting gradient.

## B.4 The λ-Curriculum

`ER._curriculum_lambda(task_id, g_new_norm=None, g_replay_norm=None)` in `src/methods/er.py`. Returns the curriculum weight for the *current step* of `task_id`.

Four schedules are exposed via `method.lambda_curriculum.schedule`:

| Schedule | Formula | Notes |
|----------|---------|-------|
| `linear` | λ = progress = (task_step − 1) / N | Default. Plateau at 1 after N steps. |
| `cosine` | λ = ½ (1 − cos(π · progress)) | Smooth, slower start. Same plateau. |
| `step` | λ = 0 for first N steps, then 1 | Binary delay — used as a control. |
| `adaptive` | λ = clip(EMA_α(‖g_replay‖ / ‖g_new‖), 0, 1) | Self-paced; ignores N. |

After the schedule resolves a raw λ, the result is floored at `method.lambda_curriculum.lambda_min` (default 0.0). The floor is most relevant for `adaptive`, where ‖g_replay‖ ≈ 0 at θ_0* drives EMA(r) ≈ 0 and λ would otherwise stay near zero for the very first steps.

On `task_id = 0` (no past tasks to preserve) or when `lambda_curriculum.enabled=false`, the resolver returns `1.0`. The adaptive EMA is reset to 0 at every task boundary so λ starts at 0 on the first step of every new task — matching the slow-start behaviour of the time-indexed schedules.

Unit tests for all four schedules, the λ_min floor, and the λ = 1 → vanilla ER equivalence live in `tests/test_er_curriculum.py`.

## B.5 NCL (path-finding reference)

`src/methods/ncl.py`. Implements the Natural Continual Learning algorithm of Kao et al. (2021) with K-FAC (Kronecker-Factored Approximate Curvature) for the precision matrix. Key configuration in `configs/method/ncl.yaml`:

| Field | Value | Meaning |
|-------|-------|---------|
| `fisher_samples` | 1000 | Max training examples used to estimate K-FAC factors after each task |
| `damping` | 0.2 | ε added to A and G diagonals before inversion — regularises near-zero K-FAC eigenvalues (relevant for sparse MNIST inputs) |

K-FAC factors are accumulated online — `end_task()` folds the just-finished task's factors into the evolving prior `Λ_k ≈ Λ_{k-1} + F_k`. Only `nn.Linear` layers receive K-FAC treatment; batch-norm and embeddings are updated with the raw gradient unchanged.

NCL has **no replay buffer**, so the buffer-fidelity diagnostics (`true_grad_cosine`, `true_grad_mag_ratio`) are silently no-ops when the diagnostics toggle is on.

## B.6 Datasets

### B.6.1 Rotated MNIST

`src/data/rotated_mnist.py`. Configured as a two-task sequence in `configs/dataset/rot_mnist.yaml`:

```yaml
name: rot_mnist
num_tasks: 2
rotations_deg: [0, 90]
```

Each task wraps the standard 60 000-sample MNIST training set and 10 000-sample test set with a fixed rotation applied at item-read time. Rotation is implemented via `torchvision.transforms.functional.rotate` (clockwise, fill = 0). The output of `_RotatedMNISTTask.__getitem__` is `(x, y)` with `x` of shape `(784,)` (flattened) and `y` an integer class label.

### B.6.2 Domain CIFAR-10

`src/data/domain_cifar10.py`. Three-task domain-incremental setting (`configs/dataset/dom_cifar10.yaml`):

```yaml
name: dom_cifar10
num_tasks: 3
corruption_types: [none, gaussian_noise, shot_noise]
severity: 3
```

All three tasks share the same 10 CIFAR-10 classes — only the input distribution shifts. Corruptions are applied on-the-fly via the `imagecorruptions` package; standard CIFAR-10 mean / std normalisation is applied *after* corruption.

## B.7 Stability-Gap Tracking

`src/eval/stability_gap.py`. Two trackers are instantiated at the start of every task with `task_id > 0`:

- `StabilityGapTracker` snapshots pre-task accuracy on every past task's test loader, then evaluates the model on those test loaders at every training step until `window_steps` steps have elapsed. Exposes `gap_depth`, `gap_area`, `max_drop`, and `recovery_steps` (defined in §4.2.1).
- `GradientTracker` (active only when `eval.stability_gap.grad_diagnostics.enabled=true`) records per-step `true_grad_cosine` ≔ cos(g_replay, g_true) and `true_grad_mag_ratio` ≔ ‖g_replay‖ / ‖g_true‖ over the first 500 steps. The `g_true` reference is built by `_load_full_past_task_data` in `scripts/train.py`: every past-task training sample is concatenated into a single (x, y) tensor and passed to `ER.set_diagnostic_data`. Each step then computes g_true via a forward+backward over that full set — no sub-sampling, no bootstrap, so g_true is the exact empirical past-task gradient at the current parameters. The `true_grad_*` naming flags this directly: the comparison is against the *true* past-task gradient, distinct from `grad_ratio` (which compares g_replay to the current-task gradient g_new and is independent of g_true).

## B.8 Continual-Learning Metrics

`src/eval/metrics.py`. Implements every metric from De Lange et al. (ICLR 2023):

- `ACC`, `FORG` — task-boundary metrics computed from the full accuracy matrix R[i, j].
- `min_ACC`, `WF_w`, `WP_w`, `WC_ACC` — within-training metrics from the step-level history.

All metrics return 0.0 in degenerate cases (e.g. `FORG` with N < 2). The recording API (`record_step`, `notify_task_end`) is called from `scripts/train.py` at every periodic eval and at every task-boundary eval.

## B.9 Training Entry Point

`scripts/train.py`. Hydra-based main entry point. Resolves the device (`auto` → MPS > CUDA > CPU), constructs the model / dataset / method from the resolved config, then drives the continual-learning loop:

```
for each task:
    instantiate StabilityGapTracker + GradientTracker (if task_id > 0)
    for each (x, y) in train_loader:
        method.observe(x, y, task_id)
        periodic eval / dense-window eval
    method.end_task(task_id, train_loader)
    evaluate all seen tasks → accuracy matrix entry
    save checkpoint
after all tasks:
    write accuracy matrix, metrics summary, per-task wall clock
    write run_manifest.json (git commit, config hash, seed)
    upload to W&B if tracking.backend=wandb
```

Reproducibility: the launcher refuses to run with a dirty git tree (in `scripts/run_decomposition.sh` and `scripts/run_curriculum.sh`), every run dumps its full resolved config to W&B, and `run_manifest.json` records the git SHA used.
