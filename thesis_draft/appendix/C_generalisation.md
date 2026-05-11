# Appendix C — CIFAR-10 Generalisation

## C.1 Aim and Scope

The rot-MNIST sweep (§4.4–§4.6) is deliberately a *clean-regime* experiment: the MLP has no batch-norm, the task shift is a single rigid rotation, and the loss landscape near a converged optimum is approximately quadratic in a low-dimensional active subspace. Under these conditions the envelope-theorem assumptions (A1–A3 in §3.1) hold to a good approximation and the O(1/N) bound on the trajectory contribution is sharply testable.

CIFAR-10 with a ResNet-18 backbone violates every one of those conditions: batch-norm couples the per-layer curvature, ReLU non-linearity over 30+ million parameters renders the Hessian poorly conditioned, and a corruption-style domain shift (Gaussian noise, shot noise) produces a non-trivial change in input statistics. The §3.5 assumption analysis predicts that the *qualitative* direction of the prediction (continuous λ-schedule reduces depth) should carry over, but the *quantitative* O(1/N) scaling is not guaranteed.

The CIFAR-10 generalisation block is therefore scoped as a **direction-of-effect gut check**, not a second full sweep. Its job is to confirm or falsify carry-over of the headline finding, not to characterise the full Pareto surface.

## C.2 Testbed

| Component | Value |
|-----------|-------|
| Dataset | `dom_cifar10` (§B.6.2) — 3 tasks: clean, gaussian_noise (sev. 3), shot_noise (sev. 3) |
| Backbone | ResNet-18 with CIFAR stem (§B.1.2) |
| Train / test per task | 50 000 / 10 000 (standard CIFAR-10) |
| Epochs per task | 1 (online regime) |
| Batch size | 256 |
| Optimiser | SGD, lr = 0.1, weight_decay = 0.0 |
| Momentum | 0.9 (fixed; no momentum cross — see §C.4) |
| Replay buffer | 1 000 reservoir samples |
| Seeds | 5 (`seed ∈ {1, 2, 3, 4, 5}`) |
| Dense per-step eval window | 500 steps after every task transition |

CIFAR-10 has fewer steps per task (~195 batches/task at batch size 256) than rot-MNIST (~235), so the 500-step `window_steps` setting effectively covers the entire new-task training; this is harmless but means `gap_area` and `recovery_steps` see the full trajectory rather than a prefix.

## C.3 Conditions

Four headline conditions only (full overrides in §A.3):

| Label | What is varied | Role |
|-------|----------------|------|
| D1 — Vanilla ER | — | Reference baseline (analogue of G1) |
| D2 — Standard NCL | path-finding instead of replay | Reference for whether second-order machinery transfers (analogue of NCL row in §A.1) |
| D3 — Best linear curriculum | λ-curriculum, ramp length = best N from rot-MNIST (default 200) | Headline landscape-shaping condition |
| D4 — Adaptive curriculum | λ-curriculum, closed-loop EMA on the gradient ratio | Hyperparameter-free landscape-shaping condition |

D3's ramp length is set via the `BEST_N` env var in `scripts/run_curriculum.sh`. The default is 200, matching the headline rot-MNIST setting; override after the rot-MNIST C-series identifies a different best N.

## C.4 What This Block Does Not Probe

The following are deliberately out of scope for the CIFAR-10 block and would each require a separate study:

- **Decomposition on CIFAR.** No D-analogue of G2/G3/G4. Even if the headline direction carries over, we make no claim about the three-contributor shares on CIFAR. The decomposition argument is anchored on rot-MNIST.
- **Momentum cross.** All D-conditions run at momentum 0.9 (the headline setting). The momentum-suppression-of-oscillation prediction and the momentum-alone falsifier are tested on rot-MNIST only, where 5 × 2 × 5 = 50 cheap MLP runs are affordable; replicating that cross on ResNet-18 + CIFAR would multiply the runtime by orders of magnitude without strengthening the headline.
- **λ_min sweep.** D4 uses `lambda_min=0.0`; the floor sweep (C5–C7) is not replicated on CIFAR.
- **Buffer-fidelity diagnostics.** Off for the D-series. The per-step g_true comparison is most useful on the decomposition conditions, which are not part of this block.
- **NCL damping sensitivity.** D2 uses the default damping (0.2). A sensitivity check on damping is left as a separate sub-study.

## C.5 What a Successful Carry-Over Looks Like

A positive result is **qualitative agreement** with the rot-MNIST direction:

- `gap_depth(D3) < gap_depth(D1)` (curriculum reduces depth relative to vanilla ER), in a paired-Wilcoxon test across the five seeds.
- `gap_depth(D4) ≈ gap_depth(D3)` within a small margin (adaptive matches tuned linear).
- `ACC(D3), ACC(D4) ≈ ACC(D1)` (curriculum preserves final accuracy; no balancing-style ACC collapse).
- `gap_depth(D3) ≤ gap_depth(D2)` (curriculum matches or beats NCL on depth).

We do *not* require the O(1/N) quantitative scaling — Chapter 6 will discuss any departure from it in terms of the assumption violations of §3.5.

## C.6 Launching the Block

```bash
# Run only the CIFAR block on the curriculum-script entry point:
BLOCK=cifar10 bash scripts/run_curriculum.sh

# Override the best-N once the rot-MNIST sweep has reported:
BLOCK=cifar10 BEST_N=100 bash scripts/run_curriculum.sh

# Use a different momentum setting for the entire D-series:
BLOCK=cifar10 MU_CIFAR=0.0 bash scripts/run_curriculum.sh
```

Total runs in this block: 4 × 5 = **20**. On a single GPU at `N_JOBS=2`, the block completes in roughly 1–2 hours depending on hardware.
