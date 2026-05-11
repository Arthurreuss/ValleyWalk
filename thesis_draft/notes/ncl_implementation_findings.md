# NCL Implementation: findings and lessons

**Date:** 2026-05-11
**Scope:** A focused debugging + tuning session on the NCL baseline (Kao et al., NeurIPS 2021) used as a reference method in this thesis. This note captures what was wrong, what was fixed, and what we learned about the method's behaviour vs. its theoretical claims.

---

## TL;DR

The original NCL implementation in `src/methods/ncl.py` had three bugs that together prevented it from converging at all. After fixing them, a small hyperparameter sweep over `prior_init` (α) showed that the paper-default α=1.0 is *over*-regularising on rot-MNIST: α=0.1 wins, and the gradient suggests pushing further. Even at the best α, **simple ER outperforms NCL** in our 2-task setup — which is consistent with the post-2022 literature consensus that replay dominates regularisation-only methods when the buffer is large enough. NCL also exhibits a non-trivial **stability gap**, which is *not* contradictory to its "stability" claim: NCL minimises final forgetting (FORG), not the transient dip immediately after the task switch.

---

## 1. Bugs found and fixed

All three were silent — tests passed previously because the test suite asserted shape/PSD/symmetry properties of the K-FAC factors but never exercised the *training dynamics*.

### 1.1 `training.momentum` was being silently dropped

`NCL.__init__` constructed `torch.optim.SGD(...)` without forwarding `cfg.training.momentum`. The §4.6 momentum-cross experiments for NCL would have been no-ops. Fix at `src/methods/ncl.py:170` (added `momentum=float(cfg.training.momentum)`). A regression test now pins this: `tests/test_ncl.py::test_optimizer_reads_momentum_from_training_cfg`.

### 1.2 Missing initial prior `p_w = α·I`

Algorithm 1 line 6 in Kao et al. (2021) initialises `Λ_0 = p_w`, a Gaussian prior precision. The implementation initialised `Λ_0 = 0` (empty K-FAC dicts), so on task 1 `Λ_1 = F_0` alone — and in low-curvature directions of `F_0`, `Λ_1⁻¹` had near-unbounded eigenvalues. The natural-gradient step `Λ_1⁻¹∇L_1` exploded within ~20 task-1 steps; loss went to NaN. Fix: each Linear layer's K-FAC factors are initialised to `A_l = G_l = √α·I` at construction (so `A ⊗ G = α·I` per layer). Now `Λ_k = α·I + Σ F_i`, and `Λ_k⁻¹` has eigenvalues bounded by `1/α`. Config: `method.ncl.prior_init` in `configs/method/ncl.yaml`.

### 1.3 Rubber-band on task 0 prevented learning

Once 1.2 was fixed, task 0 still couldn't learn — loss stayed at `ln(10) ≈ 2.27` (uniform prediction). Cause: with `Λ_0 = α·I` and `μ_0 = θ_init`, the Eq. (8) rubber-band term `η·(θ − θ_init)` actively dragged θ back to its random initialisation every step. With `η_eff ≈ 1.0` (lr=0.1, momentum=0.9), this dominated the gradient signal.

Fix: a `_has_prior` flag that flips to `True` only after the first `end_task` call. On task 0 (no real Fisher accumulated yet), `_apply_natural_gradient` does plain SGD pass-through. From task 1 onward, the full Eq. (8) update fires with `Λ_1 = α·I + F_0`.

> **Note on paper fidelity.** This gating *deviates* from a literal reading of Algorithm 1 in the paper, which applies Eq. (8) from task 0. We chose deviation because the literal reading didn't converge at α=1; the standard practical interpretation across continual-learning implementations (and what most reference code does) is to start regularising only once a real posterior exists. At very small α, paper-literal behaviour might also work; not re-tested.

---

## 2. Hyperparameter sweep findings

### 2.1 Iteration 1: α ∈ {0.1, 1.0, 10.0}, plus ρ=0 control

| Config | What it tested | Result |
|---|---|---|
| `baseline` (α=1.0, ρ=0.9) | Paper-like default | Reasonable but not best |
| `loose_prior` (α=0.1) | Weaker rubber-band | **Best** |
| `tight_prior` (α=10.0) | Stronger rubber-band | Worse — over-regularises |
| `no_momentum` (ρ=0.0) | Remove momentum × rubber-band interaction | (Reference data; no decisive signal) |

**Direction:** smaller α is better, monotonically, in the range tested.

### 2.2 Iteration 2 (queued): α ∈ {0.1, 0.03, 0.01, 0.003} + lr=0.05 rescue

Script `scripts/probe_ncl_hparams.sh` updated to probe further down the α axis. `alpha_0.003` is expected to hit the divergence floor; `alpha_0.03_lowlr` tests whether halving lr rescues stability if `Λ⁻¹` amplification at small α overshoots.

### 2.3 What α actually controls — the directional argument

`prior_init` is the most consequential NCL knob because of how it interacts with the Kronecker structure:

- **In directions where `F_0` is large** (task-0-important): `Λ ≈ F_0 ≫ α·I`. α barely affects the natural-gradient step here — stability is preserved regardless of α.
- **In directions where `F_0` is small** (task-0-irrelevant): `Λ ≈ α·I`, so `Λ⁻¹ ≈ (1/α)·I`. Lowering α amplifies the step *only* in directions task 0 doesn't care about — exactly where new tasks want to move.

So α is a **directionally surgical** plasticity knob: lowering it adds plasticity without (much) cost to stability — until the amplification gets large enough that K-FAC's misestimates of "low-curvature" directions start mattering, which is the divergence floor.

---

## 3. Knob reference (config: `configs/method/ncl.yaml`)

| Knob | Role | Tuning intuition |
|---|---|---|
| `fisher_samples` | # samples for K-FAC estimation at `end_task` | More = cleaner F, less noise in `Λ⁻¹` amplification. End-of-task cost only. |
| `damping` | ε added to A, G before inversion | Pure numerical safeguard now (`p_w` does the regularisation). Keep small (1e-3). |
| `prior_init` (α) | Strength of `p_w = α·I` initial prior | The main plasticity↔stability lever. Lower = more plastic. |
| `trust_radius` | Reported in `tr_scale` diagnostic | Not enforced (paper Algorithm 1 absorbs r into η). |
| `training.lr` | Step size η | Multiplies *both* terms in Eq. (8) equally — symmetric, not surgical. |
| `training.momentum` | Velocity buffer decay ρ | Passes the full Eq. (8) direction through momentum, including the rubber-band — can compound prior pull. |

---

## 4. Diagnostics

`get_step_diagnostics()` returns two scalars logged per step in each task's `task_NN_train.csv`:

### `kl_proxy` = ½ (θ − μ)ᵀ Λ (θ − μ)

The quadratic form of the rubber-band penalty itself. A proxy for KL between the posterior approximation and the prior. Reads:

- Rising smoothly during a task → expected (θ adapting to new data).
- Spikes → numerical issue. In the first failed smoke run it went 0.98 → 42.7 → 564 → 2.5e+7 over ~5 steps, in lockstep with the loss explosion. Earliest warning sign we have.
- Cross-config: higher final `kl_proxy` = configuration that wandered further from μ (more plastic).

### `tr_scale` = min(1, r / ‖step‖_Λ)

Trust-region scale from Eq. (7). We *compute* it but don't clip the update.

- Stays at 1.0 → step within the implicit trust region. Healthy.
- 0.3–1.0 → step mildly oversized. Usually fine.
- Drops below 0.3 → strong signal of imminent divergence. In the failed run it dropped 0.93 → 0.78 → 0.34 → 0.11 → 0.012 → 1.7e-9 just before NaN.

---

## 5. Why ER beats NCL on rot-MNIST 2-task

We observed simple ER outperforming NCL even at the best α. This is **expected** and not a sign the implementation is broken:

1. **Few-task regime is unfavourable to NCL.** With only 2 tasks, ER with any reasonable buffer just memorises task 0; there's no room for a regularisation method to compete. Kao et al.'s SOTA claims are from long task sequences (10+) with *constrained* ER buffers.
2. **Community consensus since ~2022.** Buzzega et al. (DER, 2020), Boschini et al. (2022), GDumb (Prabhu et al., 2020), and the "Continual Evaluation" line of work have repeatedly shown that well-tuned ER with adequate buffer beats every regularisation-only method, NCL included. NCL's modern value is "best *regularisation-only* method", not "best CL method overall".
3. **Implementation gaps vs. reference code.** Our implementation differs from the official `tachukao/ncl` repo in ways that probably cost performance:
   - **Empirical Fisher** (gradient against true labels) vs. proper Fisher (sampled labels). Important when the model is confident on old tasks — empirical Fisher underestimates curvature exactly where you want protection.
   - **End-of-task K-FAC snapshot** vs. EMA-style accumulation during training.
   - **Fixed scalar damping** vs. adaptive Tikhonov with per-factor scaling (π²·ε_A + ε_G).
   - **`_has_prior` gating** vs. paper-literal Eq. (8) from task 0.

Items 1–2 are structural; items 3 (the implementation gaps) could close some of the distance but probably not enough to flip the result in the 2-task regime.

---

## 6. The stability gap

NCL exhibits a non-trivial stability gap (transient dip in task-0 accuracy immediately after the task switch). This **is not contradictory** to its stability claim:

- NCL's "stability" = small **final** forgetting (FORG, post-task-1-training).
- The stability gap = a **transient** dip in the first ~tens of steps of task 1.

De Lange et al. (2023) showed every CL method has a stability gap, regardless of regularisation strategy. For NCL specifically the gap comes from:

1. At the task switch, θ = μ_0, so the rubber-band is zero. The first step is *pure* preconditioned gradient `−η·Λ_0⁻¹∇L_1(θ_0*)` on a freshly-bad loss.
2. K-FAC is a layer-wise Kronecker approximation, not the true Fisher. Directions K-FAC thinks are "low Fisher" can still matter for task 0, so the redirected step still hurts.
3. Lower α (our winner) widens this gap: `Λ⁻¹` has eigenvalues capped at 1/α, so smaller α = more first-step displacement in K-FAC-underestimated directions = bigger transient drop.

There's a trade-off here we bought into without noticing: our α sweep optimised ACC/FORG but probably traded some gap depth for that gain.

Knobs that target the gap specifically (none currently applied):

- Smaller `lr` (or a first-task-switch lr warm-up). Gap depth scales linearly with η.
- Reset the SGD momentum buffer at `end_task`. Currently leftover task-0 velocity compounds the first task-1 step.
- Proper Fisher + more `fisher_samples`. Reduces K-FAC's underestimation of important directions.

---

## 7. Implications for the thesis

- **Position NCL as the strongest regularisation-only baseline**, not as a competitor to replay. The thesis comparison should make this regime explicit.
- **The 2-task rot-MNIST setting is a debugging vehicle, not a showcase for NCL.** If the thesis needs to demonstrate NCL competitively, run the 10-task variants of P-MNIST or R-MNIST with constrained ER buffers — that's where Kao et al.'s SOTA claim lives.
- **The stability gap result is a thesis-worthy finding**, not a failure. NCL's transient gap vs. its FORG advantage is a clean Pareto-frontier story; the gap quantifies a cost the original paper didn't measure.
- **The `_has_prior` gating choice is a thesis-worthy implementation note.** Worth a paragraph in the methodology chapter explaining why we deviated from paper-literal Algorithm 1 and what that means for comparability.

---

## 8. Open questions / not yet tested

- Does paper-literal Eq. (8) from task 0 (without `_has_prior` gating) work at α ≪ 1? If yes, we may want to remove the gating for fidelity.
- Proper Fisher (sample labels) vs. empirical: how much of the ER gap does it close?
- Does momentum-buffer reset at `end_task` shrink gap_depth without hurting FORG?
- Sweep on long task sequences (10× P-MNIST, 10× R-MNIST) — does NCL's ranking flip relative to ER as task count grows?
