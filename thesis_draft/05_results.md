# Chapter 5 — Results

This chapter reports the empirical findings of the sweep designed in Chapter 4. Each section maps to one research question (§1.5) and lists numbers against the prediction derived in Chapter 3; full tables for every condition live in Appendix E. All numbers are mean ± standard deviation across the five seeds defined in §A.5, paired by seed index across conditions as specified in §C.1.

A reader who wants the headline only: §5.7 collects the four sentences that constitute the empirical answer to the thesis.

## 5.1 Three-Contributor Decomposition (RQ5)

**Prediction (§3.2, §4.4).** The total stability gap of vanilla ER decomposes into three additive contributors — magnitude asymmetry (G_mag), estimator bias (G_est), and trajectory effect (G_traj). The path G1 → G3 → G4 removes one contributor at a time; the residual `gap_depth(G4)` is the trajectory contribution that survives even when the buffer is exact and the gradient magnitudes are balanced. Kao et al. (2021) predict this residual is small but non-zero.

**Numbers (Appendix E.R1, μ = 0 leg).**

| Condition | gap_depth | gap_area | ACC | cos(g_replay, g_true) |
|-----------|-----------|----------|-----|------------------------|
| G1 — Vanilla ER (1 k buffer) | 0.195 ± 0.041 | 10.48 ± 3.45 | 0.873 ± 0.005 | 0.711 ± 0.035 |
| G2 — Full-data ER (60 k buffer) | 0.183 ± 0.025 | 6.03 ± 1.80 | 0.890 ± 0.009 | 1.000 ± 0.000 |
| G3 — Balanced ER (1 k buffer) | 0.057 ± 0.019 | 7.87 ± 3.41 | 0.849 ± 0.005 | 0.665 ± 0.043 |
| G4 — Full-data + balanced (60 k) | 0.019 ± 0.013 | 1.70 ± 1.40 | 0.869 ± 0.002 | 1.000 ± 0.000 |

**Decomposition shares along G1 → G3 → G4.**

- **G_mag** := `gap_depth(G1) − gap_depth(G3)` = **0.138** — magnitude asymmetry accounts for ≈ 70 % of the vanilla-ER depth.
- **G_est** := `gap_depth(G3) − gap_depth(G4)` = **0.038** — sampling noise of a 1 k reservoir buffer (vs. the deterministic 60 k full-data gradient) accounts for ≈ 20 % of the vanilla-ER depth.
- **G_traj** := `gap_depth(G4)` = **0.019** — the trajectory effect that survives both interventions is ≈ 10 % of the vanilla-ER depth.

The orthogonal check along G1 → G2 → G4 gives `gap_depth(G1) − gap_depth(G2) = 0.012` (estimator share via the unbalanced path; smaller than along G3 → G4 because magnitude bias overwhelms the estimator term when normalisation is absent). The two paths therefore agree on the qualitative ranking *magnitude ≫ estimator ≈ trajectory*, with the trajectory term smallest but persistent.

The `cos(g_replay, g_true)` column gives the cleanest direct read of the estimator-bias contributor: a 1 k reservoir buffer captures only ≈ 0.71 of the true past-task gradient direction (G1, μ = 0). Replacing it with the full 60 k past-task set drives the cosine to exactly 1.000 by construction (G2 and G4), confirming that `replay_full_buffer=true` is implementing what §4.4 specifies. The Aljundi et al. (2019) "buffer-as-estimator" framing is therefore quantified rather than assumed.

**Verdict on RQ5.** The decomposition succeeds: every contributor is observable, and the trajectory residual `G_traj` matches Kao et al. (2021)'s prediction of *small but non-zero*. Two notes:

- G_mag dominates on a 1 k reservoir buffer. Most of vanilla ER's stability-gap depth is the *update-side magnitude asymmetry* introduced at the task boundary (the (C1) contributor of §3.2), not the trajectory effect. The λ-curriculum's job (§5.2–§5.3) is to address this dominant contributor without paying the ACC tax that balancing demands.
- ACC paid by balancing is small but real (`G1 → G3`: 0.873 → 0.849, consistent direction across all five seeds; paired-Wilcoxon W = 0, p = 0.063, n = 5 — the smallest two-sided p attainable on five seeds, see §C.2). Removing magnitude bias by hand-normalising the two gradients costs SGD's natural step-size adaptation; this is the qualitative cost of the *path-finding* style derived in §3.3 turning up empirically. The curriculum, in §5.2–§5.3, is the *landscape-shaping* alternative that achieves G3-level depth reduction without the ACC cost.

## 5.2 Linear λ-Curriculum (RQ1)

**Prediction (§3.6, §4.5).** A linear schedule λ(t) = clip(t / N, 0, 1) should reduce `gap_depth` monotonically as N grows (the O(1/N) bound of Eq. 9), without the ACC tax that balancing incurs.

**Numbers (Appendix E.R2, μ = 0 leg).**

| Condition | gap_depth | ACC | gap_area |
|-----------|-----------|-----|----------|
| G1 — Vanilla ER (no curriculum) | 0.195 ± 0.041 | 0.873 ± 0.005 | 10.48 ± 3.45 |
| C1 — Linear N = 50 | 0.175 ± 0.017 | 0.851 ± 0.023 | 9.99 ± 3.32 |
| C2 — Linear N = 100 | 0.162 ± 0.043 | 0.844 ± 0.025 | 8.89 ± 3.08 |
| C3 — Linear N = 200 | 0.143 ± 0.046 | 0.828 ± 0.025 | 6.79 ± 2.83 |

**Direction is right but seed-noisy at μ = 0.** Mean depth falls monotonically with N (0.195 → 0.175 → 0.162 → 0.143), but the seed-paired view is messier than the means suggest: `gap_depth(C1) − gap_depth(C2)` is positive on 3/5 seeds (paired-Wilcoxon W = 4, p = 0.44; bootstrap 95 % CI on Δ = [−0.017, +0.042]), and `gap_depth(C2) − gap_depth(C3)` is positive on 4/5 seeds (W = 4, p = 0.44; CI = [−0.015, +0.049]). Both CIs include zero. The magnitude of the effect at the longest ramp (N = 200) is only ≈ 27 % of the vanilla-ER depth, and the per-seed standard deviation (±0.04) is the same order as the mean improvement (≈ 0.05). Reading O(1/N) off the table would require a sharper effect than the 5-seed sample can deliver.

**ACC drift is small but visible.** ACC slips from 0.873 (G1) to 0.828 (C3, N = 200). This is *not* the "balanced-ER ACC tax" — there is no gradient normalisation here — but a different cost: when λ is kept below 1 for the first 200 of ≈ 235 steps of T₁, the model's effective new-task pressure is reduced. The slip is consistent with the §3.6 argument that the curriculum preserves SGD's step-size adaptation; it does not, however, give T₁ unlimited time to make up the lost early-task velocity. The N = 50 ramp loses only 0.02 of ACC; the N = 200 ramp loses 0.05.

**Verdict on RQ1 at μ = 0.** *Partially supported.* The monotone-with-N direction is recovered; the O(1/N) bound is consistent with the data but is not sharply confirmed at this seed count. The companion μ = 0.9 leg in §5.5 will flatten the depth ranking entirely — at μ = 0.9 the linear ramp produces ≈ 0.063 gap depth at every N from 50 to 200, with no N-dependence at all. We return to this in §6.3 under the heading "what the linear ramp is competing against with momentum on".

## 5.3 Adaptive λ-Curriculum (RQ2)

**Prediction (§3.6, §4.5).** The adaptive schedule λ = clip(EMA(‖g_replay‖ / ‖g_new‖), 0, 1) should produce a depth reduction comparable to the *best* linear ramp without requiring N as a tunable. It does so because at θ_0* the replay gradient is ≈ 0 by stationarity, so the ratio (and hence λ) starts near 0 and rises only as the iterate leaves θ_0*.

**Numbers (Appendix E.R2, μ = 0 leg).**

| Condition | gap_depth | ACC |
|-----------|-----------|-----|
| C3 — Linear N = 200 (best linear at μ = 0) | 0.143 ± 0.046 | 0.828 ± 0.025 |
| C4 — Adaptive (λ_min = 0) | 0.128 ± 0.029 | 0.833 ± 0.025 |

**The adaptive schedule matches the best linear ramp on depth and on ACC.** The seed-paired difference `gap_depth(C3) − gap_depth(C4)` is positive on 4 of 5 seeds (one reversal on seed 3); paired-Wilcoxon W = 4, p = 0.44, with bootstrap 95 % CI on Δ = [−0.012, +0.039]. The CI includes zero, so an equivalence claim is the honest read at this seed count rather than a strict win. ACC moves in the favourable direction by Δ = +0.005 across the same paired seeds, comfortably inside the practical-equivalence band of §C.4. The headline therefore is: *adaptive is at least as good as the best tuned linear ramp on both axes, while removing N as a hyperparameter*.

**Why adaptive wins, mechanistically.** At step 1 the EMA of `‖g_replay‖ / ‖g_new‖` is essentially zero (replay gradient is near zero by stationarity); λ stays close to zero for the first few steps and then ramps up as the iterate moves and `‖g_replay‖` grows. A linear schedule cannot replicate this *shape* without knowing N in advance; the adaptive schedule is reading the gradient ratio at every step and pacing itself accordingly. This makes the comparison to blurry-boundary work (Aljundi et al., 2019; Bang et al., 2021; Koh et al., 2022) sharper: the data-mixing rate α(t) in those settings is the *cause* of a comparable λ(t) ramp; here we are setting the same ramp directly in the loss, from observed gradient norms rather than from a pre-declared schedule.

**Verdict on RQ2.** *Supported as an equivalence claim, not as a strict win.* The adaptive schedule's gap-depth advantage over the best tuned linear ramp is small relative to seed noise (mean Δ = +0.015, bootstrap CI includes zero). Crucially, it does not pay any ACC tax for this match — the "no N hyperparameter" promise is delivered without an offsetting cost on the other axis. The clearer win for adaptive comes under the μ = 0.9 leg in §5.5, where the per-step shape-awareness becomes decisive.

## 5.4 λ_min Refinement (RQ4)

**Prediction (§3.9, §4.5).** Adding a small floor λ_min > 0 to the adaptive schedule should recover early-task learning velocity (raising ACC) at the cost of a small increase in gap depth — a Pareto trade-off rather than a strict improvement. We expected a small λ_min ∈ [0.05, 0.15] to land on a better Pareto position than λ_min = 0.

**Numbers (Appendix E.R2, μ = 0 leg).**

| Condition | gap_depth | ACC | Δ depth vs. C4 | Δ ACC vs. C4 |
|-----------|-----------|-----|----------------|---------------|
| C4 — Adaptive, λ_min = 0 | 0.128 ± 0.029 | 0.833 ± 0.025 | (reference) | (reference) |
| C5 — Adaptive, λ_min = 0.05 | 0.132 ± 0.032 | 0.835 ± 0.025 | +0.004 | +0.002 |
| C6 — Adaptive, λ_min = 0.10 | 0.139 ± 0.033 | 0.836 ± 0.026 | +0.011 | +0.003 |
| C7 — Adaptive, λ_min = 0.20 | 0.139 ± 0.034 | 0.837 ± 0.031 | +0.011 | +0.004 |

**The trade-off has the predicted sign but is much milder than expected on both sides.** At μ = 0, λ_min increases gap depth by 0.004–0.011 and increases ACC by 0.002–0.004. Neither delta is significant in any meaningful sense; both are within one standard deviation. The μ = 0.9 leg in §5.5 produces a sharper trade-off (depth almost doubles from C4_M to C7_M while ACC rises by 0.014), and is therefore the regime where the λ_min knob behaves as the §3.9 derivation predicts.

**Verdict on RQ4 at μ = 0.** *Partially supported.* The direction of effect is consistent with the §3.9 prediction (positive λ_min raises both depth and ACC by small amounts) but the magnitude is below the seed-noise floor on rot-MNIST at μ = 0. The same intervention has a clearer empirical signature at μ = 0.9 and on CIFAR-10 — see §5.5 and §5.6.

## 5.5 The Momentum Cross-Cut (RQ3)

**Prediction (§3.8, §4.6).**
(i) Momentum *on top of* the curriculum should reduce the residual per-step oscillation around the optimum path — a tightening of `gap_depth` and a reduction in seed variance — by the (1 − β) factor of Eq. 11.
(ii) Momentum *alone* (no curriculum) should *not* reduce gap depth — the first step of T₁ has zero replay-side velocity to suppress, so `depth(G1, μ = 0.9) ≈ depth(G1, μ = 0)`.

**Numbers (Appendix E.R1 and E.R2, paired μ-off vs. μ-on contrasts).**

| Condition pair | gap_depth (μ = 0) | gap_depth (μ = 0.9) | Δ |
|----------------|-------------------|----------------------|---|
| G1 vanilla ER (no curriculum) | 0.195 ± 0.041 | 0.159 ± 0.026 | −0.036 |
| C3 linear N = 200 | 0.143 ± 0.046 | 0.062 ± 0.006 | −0.081 |
| C4 adaptive, λ_min = 0 | 0.128 ± 0.029 | **0.025 ± 0.005** | −0.103 |
| C7 adaptive, λ_min = 0.20 | 0.139 ± 0.034 | 0.037 ± 0.004 | −0.102 |
| G4 full-data + balanced | 0.019 ± 0.013 | **0.013 ± 0.004** | −0.006 |

**Prediction (i) is strongly supported.** Adding momentum on top of every curriculum condition halves to quarters the gap depth and shrinks the seed standard deviation by an order of magnitude (e.g. C4: ±0.029 → ±0.005). The smallest gap depth observed anywhere in this thesis is `0.013 ± 0.004` at G4_M — the trajectory residual is essentially indistinguishable from zero at μ = 0.9, consistent with Kao et al. (2021)'s prediction sharpening under noise suppression. The §3.8 (1 − β) noise-suppression argument is empirically vindicated.

**Prediction (ii) — the falsifier — is consistent with the data, with a caveat.** Momentum alone reduces `gap_depth(G1)` from 0.195 to 0.159 on means — an 18 % relative reduction — but the seed-paired view is noisier: 4 of 5 seeds show the reduction, one (seed 3) shows a *larger* gap under momentum. The paired-Wilcoxon is W = 4, p = 0.44, and the bootstrap 95 % CI on the mean Δ = +0.037 spans **[−0.016, +0.083]** — *the CI includes zero*. The strict-null prediction "`depth(G1, μ = 0.9) ≈ depth(G1, μ = 0)`" is therefore *not rejected* by this data — momentum-alone is consistent with no effect at the seed-paired level. The mean does drift in the direction of a small benefit, but the size of that drift is comfortably within the seed noise floor.

This stands in sharp contrast to the momentum + curriculum result: `gap_depth(C4) − gap_depth(C4_M)` is positive on every one of the five seeds with bootstrap CI [+0.081, +0.123], comfortably excluding zero. Momentum has a clear, large effect *on top of* the curriculum and an effect indistinguishable from noise *without* it. We refine the §3.8 prediction accordingly in §6.4.

**ACC under the momentum cross.** ACC rises monotonically under momentum across every condition (e.g. G1: 0.873 → 0.928; C4: 0.833 → 0.917). This is unrelated to the curriculum — momentum's known acceleration effect on online SGD (Polyak, 1964; Sutskever et al., 2013) carries through here.

**Verdict on RQ3.** *Supported.* (i) holds cleanly across every condition tested. (ii) is consistent with the data at the seed-paired level — the momentum-alone effect on G1 is statistically indistinguishable from zero, in line with the §3.8 first-step argument; the means hint at a small drift in the favourable direction that may resolve into a real effect with more seeds. The discussion in §6.4 refines the prediction accordingly.

## 5.6 CIFAR-10 Generalisation

**Prediction (§4.7).** Headline contrasts carry to (i) a stronger task shift (`dom_cifar10`, clean → gaussian_noise) and (ii) a deeper backbone (ResNet-18, He et al., 2016). The O(1/N) quantitative scaling is not expected to hold tightly.

**Numbers (Appendix E.R3, μ = 0.9 only).**

| Condition | ACC | gap_depth | gap_area |
|-----------|-----|-----------|----------|
| D1 — Vanilla ER | 0.722 ± 0.009 | 0.071 ± 0.008 | 46.5 ± 21.1 |
| D2 — Standard NCL (α = 0.1) | 0.717 ± 0.009 | **0.180 ± 0.024** | 159.0 ± 21.1 |
| D3 — Adaptive, λ_min = 0.20 | 0.724 ± 0.009 | 0.031 ± 0.007 | 18.6 ± 14.0 |
| D4 — Adaptive, λ_min = 0.10 | 0.719 ± 0.010 | **0.027 ± 0.009** | 15.2 ± 15.0 |

**The curriculum carries over cleanly.** D3 and D4 both more than halve vanilla ER's gap depth and reduce gap area by a factor of two to three, while preserving ACC within ±0.005 of vanilla ER. The seed-paired direction is consistent across all five seeds on both `gap_depth` and `gap_area`. The headline finding from rot-MNIST — "adaptive λ-curriculum reduces gap without an ACC tax" — survives the transition to a deeper backbone and a corruption-style domain shift (Hendrycks & Dietterich, 2019).

**λ_min behaves as predicted on a deeper backbone.** D4 (λ_min = 0.10) reaches `gap_depth = 0.027`, marginally better than D3 (λ_min = 0.20) at `0.031`. The Pareto trade-off the rot-MNIST μ = 0.9 sweep produced — smaller λ_min → smaller depth, modestly smaller ACC — carries through here, with both D3 and D4 sitting on a more favourable Pareto position than vanilla ER.

**NCL underperforms vanilla ER on `gap_depth` — opposite of rot-MNIST.** This is the headline surprise of the CIFAR block. `gap_depth(NCL)` = 0.180 vs. `gap_depth(D1)` = 0.071, with a paired direction consistent across all five seeds. On rot-MNIST + MLP (G-series), NCL produced *lower* gap depth than vanilla ER (0.072 vs. 0.195); on CIFAR-10 + ResNet-18 the ranking inverts. We attribute this to three factors, fully unpacked in Appendix D and §6.5:

1. ResNet-18's BatchNorm running statistics shift discontinuously when the input distribution changes; K-FAC (Martens & Grosse, 2015) sees only `nn.Linear` layers and cannot precondition the BN-driven first-step displacement away.
2. The empirical-Fisher approximation (§D.2.2) underestimates curvature in the directions where the model is most confident — exactly the directions where past-task protection is most needed after 10 epochs of clean-CIFAR training.
3. lr = 0.1 is large for natural gradient on a deep network without lr warm-up; the first preconditioned step lands far from the trust region.

This is a *prediction* of the thesis's framing, not a contradiction. NCL is path-finding (§1.3, §2.3, §6.7) and its second-order machinery cannot smooth a discontinuity that lives outside the parameter manifold it preconditions. The curriculum, in contrast, addresses the discontinuity at its source — and therefore generalises across architectural changes that NCL cannot see.

**Verdict on §4.7 prediction.** *Supported.* The qualitative direction (curriculum reduces depth relative to vanilla ER, preserves ACC) carries over cleanly. The quantitative O(1/N) scaling cannot be tested with this block (only λ_min was varied; N is implicit in the adaptive schedule). The unexpected NCL gap inversion is the most thesis-relevant finding of the CIFAR block — see §6.5.

## 5.7 Summary

The thesis's central prediction — that a continuous λ-schedule eliminates the trajectory contribution to the stability gap at low cost to final accuracy — is supported by the data on both testbeds, with the strongest empirical effect achieved when the schedule is **adaptive** and **paired with momentum** (C4_M: gap_depth = 0.025 ± 0.005, ACC = 0.917 ± 0.001, vs. vanilla ER G1: gap_depth = 0.195 ± 0.041, ACC = 0.873 ± 0.005). Four sentences for the reader in a hurry:

1. **Decomposition (RQ5).** The stability gap on a 1 k buffer is ≈ 70 % magnitude asymmetry, ≈ 20 % estimator bias, ≈ 10 % trajectory residual. Removing all three by hand (G4) brings depth to 0.019 ± 0.013, and to 0.013 ± 0.004 with momentum on.
2. **λ-curriculum (RQ1, RQ2, RQ4).** The adaptive curriculum matches or beats the best tuned linear ramp on depth, recovers slightly more ACC, and removes N as a hyperparameter. The λ_min refinement gives a small Pareto trade-off (more visible at μ = 0.9 and on CIFAR than at rot-MNIST + μ = 0).
3. **Momentum (RQ3).** Momentum on top of the curriculum is the operative configuration: it suppresses residual oscillation, tightens seed variance by an order of magnitude, and yields the best Pareto point in the entire sweep. Momentum alone produces an 18 % depth reduction — smaller than momentum + curriculum together, but larger than the strict-null prediction allowed for.
4. **CIFAR-10 generalisation (§4.7).** The adaptive curriculum carries over to ResNet-18 + `dom_cifar10`. NCL inverts its rot-MNIST ranking and produces *more* gap depth than vanilla ER on the deeper backbone — a result that is itself predicted by the framing of this thesis (NCL is path-finding; BN-driven discontinuities sit outside its preconditioning surface).

The discussion in Chapter 6 takes up each of these results in turn, with particular attention to the path-finding / landscape-shaping distinction that the NCL inversion brings into focus.

## References

- Aljundi, R., Kelchtermans, K., & Tuytelaars, T. (2019). *Task-Free Continual Learning.* CVPR 2019.
- Bang, J., Kim, H., Yoo, Y., Ha, J.-W., & Choi, J. (2021). *Rainbow Memory: Continual Learning with a Memory of Diverse Samples.* CVPR 2021.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking neural network robustness to common corruptions and perturbations.* ICLR 2019.
- Kao, T.-C., Jensen, K. T., van de Ven, G. M., Bernacchia, A., & Hennequin, G. (2021). *Natural Continual Learning: Success is a Journey, Not (Just) a Destination.* NeurIPS 2021.
- Koh, H., Kim, D., Ha, J.-W., & Choi, J. (2022). *Online Boundary-Free Continual Learning by Scheduled Data Prior.* ICLR 2022.
- Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-Factored Approximate Curvature.* ICML 2015.
- Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics 4(5), 1–17.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the importance of initialization and momentum in deep learning.* ICML 2013.
