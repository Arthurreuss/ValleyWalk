# Chapter 4 — Methodology

This chapter describes the empirical testbed (§4.1), the metrics that summarise each run (§4.2), and the full sweep of experimental conditions (§4.3–§4.7) that test the theoretical predictions of Chapter 3. Each experimental section maps directly to a research question from §1.5 and to a falsifiable prediction derived in Chapter 3.

## 4.1 Testbed

### 4.1.1 Dataset: Rotated MNIST

Our primary sweep uses **Rotated MNIST (rot-MNIST)** as a **two-task** sequence:

| Task | Rotation | Train size | Test size |
|------|----------|------------|-----------|
| T₀ | 0° | 60,000 | 10,000 |
| T₁ | 90° | 60,000 | 10,000 |

The 90° rotation creates tasks that are visually distinct (well-separated input distributions) but structurally identical (same label space, same underlying digit identities). This makes the gradient dynamics at the transition maximally informative: the tasks are different enough that the new-task gradient is large at θ_0*, but similar enough that the replay buffer is directly representative of the past data distribution.

Two tasks produce exactly one transition — one instance of the stability gap. This keeps the analysis clean: there is no interference between multiple overlapping gaps.

### 4.1.2 Model: MLP

The backbone is a two-layer MLP with ReLU activations (architecture details in Appendix B):

| Component | Value |
|-----------|-------|
| Input | 784 (flattened 28×28 greyscale) |
| Hidden layers | 2 × 400 (fully-connected, ReLU) |
| Output | 10-class logits (no activation) |
| Regularisation | None (no dropout, no batchnorm) |
| Total parameters | 478,410 |

The MLP is chosen over a ResNet for the rot-MNIST sweep on deliberate scientific grounds:

- Simpler loss landscape — fewer spurious curvature effects unrelated to the curriculum dynamics.
- Faster training — the curriculum + refinement sweep requires many conditions × seeds; an MLP run completes in seconds.
- Cleaner adiabatic regime — the Hessian assumptions of §3.5 hold approximately, so the envelope-theorem prediction is sharply testable.

A ConvNet backbone is used for the CIFAR-10 generalisation experiment (§4.7); details in Appendix C.

### 4.1.3 Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Epochs per task | 1 (online regime) |
| Batch size | 256 |
| Optimiser | SGD |
| Learning rate | 0.1 |
| Momentum (default) | 0.0 (plain SGD); 0.9 for momentum conditions |
| Weight decay | 0.0 |
| Replay buffer (small) | 1,000 samples, reservoir sampling |
| Replay buffer (full-data) | 60,000 samples (all past data) — used only for the full-data conditions |
| Seeds | 5 per condition |

The **online regime** (one epoch per task) is standard for stability-gap analysis: it gives a single, clean pass through the transition where the gap is most pronounced.

### 4.1.4 Evaluation Protocol

**Dense per-step evaluation** is used during the first 250 steps of T₁ training:

- `StabilityGapTracker` snapshots the pre-task accuracy on T₀ test data before T₁ training begins.
- At every training step, the tracker evaluates the model on T₀ test data and records (step, accuracy).
- After T₁ training, the tracker computes the scalar summaries listed in §4.2.

A `GradientTracker` records, at the same per-step cadence, the directional and magnitude fidelity of the replay-buffer gradient against a fresh, large past-task sample.

Periodic evaluation (every 10 steps) is used for the general training-loss and final-accuracy curves outside the boundary window.

## 4.2 Metrics

All metrics are recorded automatically by the evaluation pipeline (`src/eval/metrics.py`, `src/eval/stability_gap.py`) and reported as **mean ± standard deviation across 5 seeds**.

### 4.2.1 Stability-Gap Metrics

Recorded by `StabilityGapTracker` from dense per-step evaluation:

| Metric | Definition | Role |
|--------|------------|------|
| **gap_depth** | max_j (acc_j^pre − min_{t} acc_j(t)) over the full record window | Primary metric — worst single point of instability |
| **gap_area** | Σ_j ∫ max(0, acc_j^pre − acc_j(t)) dt (trapezoidal) | Captures depth × duration in one scalar |
| **max_drop** | gap_depth restricted to the first 200 steps | Legacy fixed-window metric, kept for back-comparability |
| **recovery_steps** | First step at which every past task is at ≥ 90 % of its pre-task baseline, measured from the point of maximum drop | None if no recovery is observed in the record window |

### 4.2.2 Continual-Learning Metrics (De Lange et al., ICLR 2023)

Recorded by `ContinualMetrics` at task boundaries and over the full record:

| Metric | Definition | Role |
|--------|------------|------|
| **ACC** | (1/N) Σ_j A(E_j, f_{T_{N-1}}) | Average task accuracy at the final model |
| **FORG** | (1/(N−1)) Σ_{i<N-1} [A(E_i, f_{T_i}) − A(E_i, f_{T_{N-1}})] | Average accuracy drop from right-after-learning to final |
| **min-ACC** | (1/(N−1)) Σ_{i<N-1} min{ A(E_i, f_n) : n > T_i } | Worst-case retention since learning; safety-margin metric |
| **WF_w (w = 10, 100)** | Average over tasks of the max accuracy drop in any window of w consecutive evaluations | Worst-case stability over short / medium horizons |
| **WP_w (w = 10, 100)** | Symmetric counterpart of WF_w — max accuracy *gain* | Plasticity / learning velocity |
| **WC-ACC** | (1/N) · A(E_{N-1}, f_{T_{N-1}}) + (1 − 1/N) · min-ACC | Worst-case trade-off — lower bound on ACC |

The full task-boundary accuracy matrix R[i, j] = A(E_j, f_{T_i}) is logged for every run, so any downstream metric can be reconstructed.

### 4.2.3 Gradient-Fidelity Diagnostics

Recorded by `GradientTracker` during the new task's first ~500 steps. These compare the replay-buffer gradient g_replay against the *true past-task gradient* g_true, computed at every step by a separate forward+backward pass on the **entire past-task training set** (no sub-sampling — g_true is the deterministic empirical past-task gradient at the current parameters):

| Metric | Logged name | Definition | Role |
|--------|-------------|------------|------|
| **cos(g_replay, g_true)** | `true_grad_cosine` | Per-step series + mean + min over the window | Buffer-side directional fidelity (the estimator-bias contributor of §4.4) |
| **‖g_replay‖ / ‖g_true‖** | `true_grad_mag_ratio` | Per-step series + mean | Buffer-side magnitude fidelity |
| **‖g_new‖ / ‖g_replay‖** | `grad_ratio` | Per-step, from `ER.get_last_grad_ratio` | *Update-side* magnitude asymmetry — large at step 1 when ‖g_replay‖ ≈ 0 |

The first two share the `true_grad_*` prefix because they are computed against the *true* (full-data) past-task gradient g_true — they are buffer-fidelity diagnostics, measuring how faithfully a finite buffer approximates the past data the gradient is supposed to represent (Aljundi et al., 2019). The third (`grad_ratio`, no `true_` prefix) compares g_replay to the *current-task* gradient g_new and has nothing to do with g_true; its inverse, ‖g_replay‖ / ‖g_new‖, is the signal the adaptive curriculum (§4.5) uses to set λ(t).

**Scope.** Buffer-fidelity diagnostics are enabled for the **decomposition (G) conditions only**, where they provide direct per-step evidence for the estimator-bias contributor and pair with the indirect evidence from the depth differences G3 − G4. Each step adds one extra forward+backward pass over every past-task training sample (60 000 examples for rot-MNIST T₀); this is affordable on the MLP testbed but would dominate runtime on larger backbones, so the toggle stays off for the curriculum (C/D) blocks whose claims rest on gap_depth and ACC rather than on per-step gradient geometry. The toggle is available via `GRAD_DIAG=on` if a specific contrast in a later analysis needs the fidelity signal.

### 4.2.4 Statistical Comparison

Across conditions we compare metrics using the **paired Wilcoxon signed-rank test** on the seed-paired difference. Seeds are paired by index across conditions (seed = 1 of vanilla ER is paired with seed = 1 of the curriculum condition), so per-seed deltas Δ_s = m(C_a, s) − m(C_b, s) factor out the seed-induced variance that would otherwise dominate the 5-seed sample. Wilcoxon is preferred over the paired t-test because gap_depth and accuracy are bounded in [0, 1] and typically right-skewed near zero, so the normality assumption of the t-test is suspect; the Wilcoxon test exchanges that assumption for the milder one of distributional symmetry of Δ_s under the null. For equivalence-style claims ("the adaptive curriculum matches the best linear curriculum"), we report the seed-paired difference with its bootstrap 95 % CI rather than a null-hypothesis test.

## 4.3 Overview of Experiments

Each experimental block tests one or more falsifiable predictions from Chapter 3 and answers one of the research questions in §1.5. The full design is summarised below; subsequent sections describe each block.

| Block | Section | RQ tested | Falsifiable prediction (from Ch. 3) | Conditions |
|-------|---------|-----------|--------------------------------------|------------|
| Three-contributor decomposition | §4.4 | RQ5 | Magnitude + estimator + trajectory together account for the total gap; trajectory residual matches Kao et al. (§2.5) | G1–G4, plus NCL reference |
| Linear λ-curriculum | §4.5 | RQ1 | gap_depth falls monotonically with N; ACC matches vanilla ER (Eq. 9 of §3.4) | N ∈ {50, 100, 200} |
| Adaptive λ-curriculum | §4.5 | RQ2 | gap_depth falls relative to vanilla ER without an explicit tunable N | Adaptive schedule from ‖g_replay‖/‖g_current‖ |
| λ_min refinement | §4.5 | RQ4 | Adding λ_min > 0 recovers early-task velocity without inflating depth back to vanilla-ER levels (§3.7) | λ_min ∈ {0.05, 0.10, 0.20} on adaptive |
| Momentum cross-cut | §4.6 | RQ3 | Momentum tightens depth on top of the curriculum, but momentum-alone (no curriculum) does *not* reduce depth (Eq. 11 of §3.6) | Every condition above repeated at momentum 0.9 |
| CIFAR-10 generalisation | §4.7 | All | Headline contrasts carry to a stronger task shift and a deeper backbone | ER, NCL, best linear, best adaptive |

All conditions use 5 seeds and the training protocol of §4.1.3 unless noted otherwise.

## 4.4 Three-Contributor Decomposition

**Aim.** Empirically isolate the three contributors named in §3.1 — magnitude bias, estimator bias, trajectory effect — and quantify each one's share of the total gap. This is the diagnostic question of RQ5 and the empirical analogue of the discontinuity framing of Chapter 3.

**Rationale.** Two of the three contributors are predicted to be downstream symptoms of the discontinuity at the task boundary; the third (estimator bias) is a property of finite replay buffers and persists at any schedule. Removing each contributor *by construction* and observing the residual gap depth identifies which contributors drive the observed dip. The path G1 → G2 → G3 → G4 each removes one source.

**Conditions.**

| Condition | Replay buffer | Gradient handling | Removed by construction | Residual gap |
|-----------|---------------|-------------------|--------------------------|--------------|
| G1 — Vanilla ER | 1 k reservoir | Unbalanced | (nothing) | G_mag + G_est + G_traj |
| G2 — Full-data ER | 60 k (all past data) | Unbalanced | Estimator noise (exact gradient, see below) | G_mag + G_traj |
| G3 — Balanced ER | 1 k reservoir | Unit-norm balanced | Magnitude asymmetry (at the cost of SGD's step adaptation) | G_est + G_traj (+ balancing cost) |
| G4 — Full-data + balanced | 60 k | Unit-norm balanced | Magnitude + estimator | G_traj |
| NCL (reference) | None (no replay buffer) | K-FAC precision-matrix preconditioning | — | Reference path-finding method (Kao et al., 2021) |

**On "full-data" in G2/G4.** The 60 k condition holds the entire past-task training set in the buffer (`memory.total_budget=60000`, which equals the MNIST training-set size) and, at every step, computes the replay gradient on **every stored sample exactly once** (`method.replay_full_buffer=true`). The resulting replay gradient is the deterministic empirical past-task gradient at the current parameters — no sampling, no bootstrap variance. Vanilla ER (G1) draws 256 samples with replacement, so its replay gradient is a stochastic estimator of this same quantity. G_est in §3.1 is therefore *exactly* the gap-depth difference between an estimator-driven step and an exact-gradient step; G_traj in G4 is the residual once both the estimator and the magnitude asymmetry are removed.

**Decomposition shares.** Differences along the path G1 → G3 → G4 identify each contributor:

- G_mag := gap_depth(G1) − gap_depth(G3) — share removed by balancing alone.
- G_est := gap_depth(G3) − gap_depth(G4) — additional share removed by full data on top of balancing.
- G_traj := gap_depth(G4) — residual after both magnitude and estimator are removed; the trajectory contribution of Chapter 3.

The G2 condition is a sanity check: gap_depth(G1) − gap_depth(G2) should also recover G_est along an orthogonal path through the design.

**NCL reference.** NCL is included as a representative *path-finding* method: it accepts the discontinuity but preconditions each SGD step by the past-task precision matrix. Its gap depth is the natural reference for "how much depth reduction is achievable by acting in parameter space rather than in schedule space" and gives us a fair point of comparison for the λ-curriculum in §4.5.

**What we expect.** Gap depth reduces along the path G1 → G3 → G4. The trajectory residual G4 is small but non-zero (Kao et al.'s prediction, §2.5), and is the contributor that the λ-curriculum should additionally suppress in §4.5.

## 4.5 The λ-Curriculum Sweep

**Aim.** Test the envelope-theorem prediction of §3.4 — that a continuous λ-schedule eliminates the trajectory contribution to the gap — and the λ_min trade-off prediction of §3.7. Answers RQ1, RQ2, RQ4.

**Conditions.** All curriculum conditions are applied on top of standard ER (1 k reservoir, no balancing, no NCL) so that the curriculum's effect is isolated.

| Condition | λ(t) schedule | Hyperparameter | Tests |
|-----------|---------------|----------------|-------|
| Vanilla ER (baseline) | λ ≡ 1 from step 1 (discontinuous) | — | Reference |
| Linear N = 50 | clip(t / 50, 0, 1) | N = 50 | RQ1 |
| Linear N = 100 | clip(t / 100, 0, 1) | N = 100 | RQ1 |
| Linear N = 200 | clip(t / 200, 0, 1) | N = 200 | RQ1 |
| Adaptive | λ = clip(EMA(‖g_replay‖ / ‖g_new‖), 0, 1) | EMA α | RQ2 |
| Adaptive + λ_min = 0.05 | max(λ_min, clip(EMA(‖g_replay‖/‖g_new‖), 0, 1)) | λ_min = 0.05 | RQ4 |
| Adaptive + λ_min = 0.10 | max(λ_min, clip(EMA(‖g_replay‖/‖g_new‖), 0, 1)) | λ_min = 0.10 | RQ4 |
| Adaptive + λ_min = 0.20 | max(λ_min, clip(EMA(‖g_replay‖/‖g_new‖), 0, 1)) | λ_min = 0.20 | RQ4 |

The adaptive schedule is *self-paced*: at θ_0* the replay gradient is ≈ 0 (stationarity), so r := ‖g_replay‖ / ‖g_new‖ ≈ 0 and λ starts near 0. As the iterate leaves θ_0*, ‖g_replay‖ grows while ‖g_new‖ shrinks (the model fits T₁), so r climbs and λ → 1 without any explicit ramp length. The schedule ignores N entirely. A non-zero λ_min floor is mainly relevant for the adaptive variant: it prevents the schedule from sitting at λ ≈ 0 in the very first steps when the EMA estimate of r is still close to 0.

**What we expect.**

- *Linear sweep (RQ1).* gap_depth should fall monotonically as N grows from 50 → 200, consistent with the O(1/N) bound of Eq. (9). ACC should remain close to vanilla ER, because the curriculum preserves SGD's natural step-size adaptation (unlike balancing in G3).
- *Adaptive (RQ2).* The adaptive schedule should achieve a depth reduction comparable to a tuned linear curriculum, removing the need to fix N as a hyperparameter.
- *λ_min (RQ4).* Increasing λ_min from 0 should raise the gap *floor* slightly (the trajectory cannot dip below L_replay(Θ*(λ_min))) while accelerating T₁ learning. We expect a small λ_min ∈ [0.05, 0.15] to yield a better Pareto position on (depth, ACC) than λ_min = 0.

The contrast G3 (balanced ER) vs. linear N = 200 isolates the difference between *acting on the magnitude symptom alone* and *acting on the schedule discontinuity*: both should suppress depth, but only the curriculum should preserve ACC.

## 4.6 Momentum and the Curriculum × Momentum Cross

**Aim.** Test the two predictions of §3.6 — (i) momentum on top of the curriculum suppresses the residual stochastic oscillation around the optimum path, and (ii) momentum *alone* (no curriculum) does not reduce gap depth because the first-step direction is unchanged. Answers RQ3.

**Design.** Every condition in §4.4 and §4.5 is repeated with momentum 0.9 (Nesterov off), giving a 2 × (decomposition + curriculum) cross:

| Curriculum / decomp. condition | Momentum 0.0 | Momentum 0.9 |
|--------------------------------|--------------|--------------|
| Vanilla ER (G1) | ✓ | ✓ |
| Balanced ER (G3) | ✓ | ✓ |
| Full-data ER (G2) | ✓ | ✓ |
| Full-data + balanced (G4) | ✓ | ✓ |
| NCL | ✓ | ✓ |
| Linear N = 50, 100, 200 | ✓ | ✓ |
| Adaptive | ✓ | ✓ |
| Adaptive + λ_min ∈ {0.05, 0.10, 0.20} | ✓ | ✓ |

**What we expect.**

- *On top of the curriculum* — momentum should reduce the seed-to-seed standard deviation in gap_depth and the per-step oscillation amplitude on the T₀ accuracy curve, without changing the depth mean by much. This is the (1 − β) factor in the second term of Eq. (11).
- *Without the curriculum* — momentum should *not* reduce gap_depth: depth(G1, momentum 0.9) ≈ depth(G1, momentum 0.0). A null result here is the falsifiable prediction.

The single contrast (linear N = 200, momentum on) vs. (linear N = 200, momentum off) measures the residual that momentum cleans up; the contrast (vanilla ER, momentum on) vs. (vanilla ER, momentum off) tests the falsification.

## 4.7 Generalisation: CIFAR-10

**Aim.** Confirm the direction of the rot-MNIST findings under (i) stronger task shift and (ii) a deeper, harder-to-analyse architecture. This probes the regime in which the assumptions (A1)–(A3) of §3.1 weaken (§3.5).

**Conditions.** A small set of headline conditions only — not the full sweep.

| Condition | Backbone | Role |
|-----------|----------|------|
| Vanilla ER | ConvNet (Appendix C) | Reference |
| NCL | ConvNet | Reference path-finding |
| Best linear curriculum (from §4.5) | ConvNet | Headline landscape-shaping |
| Best adaptive curriculum (from §4.5) | ConvNet | Hyperparameter-free landscape-shaping |

**What we expect.** Qualitative carry-over of the rot-MNIST result: the curriculum should reduce gap depth relative to vanilla ER and at least match NCL on depth while preserving ACC better than balanced ER would. The quantitative O(1/N) scaling of the bound is not expected to hold tightly here; deviations are reported and discussed in §6.
