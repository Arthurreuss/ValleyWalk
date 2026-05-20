# Chapter 6 — Discussion

Chapter 5 reported what the sweep produced. This chapter takes up *what the numbers mean* against the predictions of Chapter 3. We work through each finding in turn, then step back in §6.7 to revisit the path-finding / landscape-shaping distinction in light of the CIFAR NCL inversion, and close with limitations (§6.8) and future work (§6.9).

## 6.1 Summary of Findings

The four-sentence summary at the end of §5.7 is the operational answer to the thesis. Restated as a single claim: **on both testbeds, the smallest stability gap among methods that preserve final accuracy is achieved by an adaptive λ-curriculum + momentum**, not by any path-finding method we tested. Two quantitative anchors for the rest of this chapter:

- The strongest single Pareto point we observed is C4_M on rot-MNIST: `gap_depth = 0.025 ± 0.005`, `ACC = 0.917 ± 0.001`. The reduction in seed variance — one order of magnitude relative to the same configuration without momentum (C4: ±0.029) — matters as much as the reduction in mean.
- The unexpected result is D2 on CIFAR-10: NCL produces `gap_depth = 0.180 ± 0.024`, ≈ 2.5× the depth of vanilla ER on the same architecture. The same NCL implementation that beat vanilla ER on rot-MNIST + MLP fails to beat it on ResNet-18.

Sections §6.2–§6.5 take up these results one by one. §6.6 inventories the predictions that landed cleanly versus those that needed refinement.

## 6.2 What the Decomposition Shares Tell Us

The G-series shares (§5.1) place magnitude asymmetry at ≈ 70 % of vanilla ER's depth, estimator bias at ≈ 20 %, and the trajectory effect at ≈ 10 %. This re-orders the priorities a typical reading of the stability-gap literature would suggest. Two implications.

**Magnitude bias is operationally larger than the literature treats it.** De Lange et al. (2023) characterise the gap descriptively (depth, area, recovery time) but do not decompose it. Hess et al. (2023) name the "trajectory bend" as the mechanistic core. Both framings are correct as far as they go, but they leave the *operational* picture incomplete: on a 1 k reservoir buffer, the trajectory residual is one-seventh the size of the magnitude effect at μ = 0. A practitioner aiming to reduce the gap who only addresses the trajectory effect (e.g., by acting on the SGD trajectory in parameter space) is targeting the smallest of the three contributors.

The λ-curriculum, by virtue of attacking the discontinuity at its source, addresses *all three* contributors simultaneously: continuity in λ smooths magnitude (g_replay is present at full strength from step 1), reduces estimator overshoot (the curriculum gives the buffer's noisy estimate less weight when the iterate is far from θ_0*), and bounds the trajectory effect (the Eq. 9 envelope-theorem bound). The decomposition is therefore consistent with — and arguably explains — why a single intervention with no curvature machinery reaches the same depth as NCL on rot-MNIST.

**The 10 % trajectory residual is the bound G4 reports.** G4 (full-data + balanced) is the cleanest empirical isolation of the trajectory contributor we can construct: estimator noise is zero by construction (`replay_full_buffer=true`), magnitude asymmetry is zero by construction (`grad_balance.normalize_components=true`). The remaining `gap_depth = 0.019 ± 0.013` at μ = 0 is the geometric trajectory effect of Kao et al. (2021), measured at the depth of the network and the regime of the experiment. Two notes:

- This residual is not zero. The Eq. 9 bound says it should *vanish* in the η → 0 continuous-time limit, but our experiments run at lr = 0.1 with discrete updates; a non-zero residual is exactly what the discretisation argument of §3.6 predicts.
- The residual *does* approach zero under momentum: G4_M reports `gap_depth = 0.013 ± 0.004` and `gap_area = 0.14 ± 0.06` — two orders of magnitude smaller `gap_area` than G3_M (8.18 ± 0.95) and the smallest gap we observed anywhere in this thesis. Kao et al. (2021)'s trajectory prediction therefore survives the noise floor only at μ = 0.9.

## 6.3 The Linear Ramp at μ = 0.9 Reveals a Floor, Not a Failure

The RQ1 prediction was monotonic depth reduction with N. At μ = 0 the prediction is recovered weakly (§5.2: 0.195 → 0.143 across N = 50 → 200). At μ = 0.9 it disappears: C1_M, C2_M, C3_M produce gap depths 0.064 / 0.064 / 0.062 — flat in N. Naively this looks like a failure of the O(1/N) bound.

A more careful reading: the O(1/N) bound applies to the *trajectory contribution* alone. At μ = 0.9, three things change simultaneously. (i) The first-step magnitude asymmetry is suppressed because the velocity buffer carries past replay-side signal forward into the first new-task step (smaller G_mag). (ii) The per-step variance is suppressed by the (1 − β) factor of §3.8 (smaller noise floor). (iii) The trajectory residual itself shrinks under noise suppression (G4_M result above). The remaining ≈ 0.06 depth at μ = 0.9 is *not* the trajectory contributor that the curriculum addresses; it is a floor set by the discrete-time + finite-buffer composition of effects the linear schedule does not touch. The linear-ramp result at μ = 0.9 is therefore consistent with the §3.6 bound being already saturated by the time the ramp begins.

The adaptive schedule's behaviour at μ = 0.9 (§5.5: C4_M = 0.025) drives the point home. Adaptive λ achieves less than half of the linear-ramp floor because it *reads the gradient ratio at every step* — it can keep λ small even after step N would have plateaued it at 1, when the ratio shows the iterate is still far from local equilibrium. The linear schedule is shape-blind; adaptive schedule is shape-aware. This is the empirical content of the §4.5 mechanistic story.

## 6.4 Momentum is Essential, Not a Refinement

§3.8 framed momentum as a "refinement" of the curriculum that suppresses residual oscillation. The data invert that framing: momentum is the *primary* axis of improvement, and the curriculum compounds with it rather than the other way around. Three observations.

**The C4 → C4_M transition is the largest depth reduction in the sweep.** Adding momentum to the adaptive curriculum reduces depth from 0.128 to 0.025 — a factor of five — and reduces seed standard deviation from 0.029 to 0.005. By comparison the C3 → C4 transition (best linear → adaptive, both at μ = 0) is 0.143 → 0.128 — a factor of 1.1.

**The G1 → G1_M result is consistent with the strict-null falsifier.** §3.8 predicted `depth(G1, μ = 0.9) ≈ depth(G1, μ = 0)` as a falsifier of "momentum alone is enough". The means do drift in the favourable direction (0.195 → 0.159, an 18 % relative reduction on the mean), but the seed-paired view shows one of five seeds reversing, paired-Wilcoxon W = 4, p = 0.44, and a bootstrap 95 % CI on Δ = +0.037 spanning [−0.016, +0.083]. The CI includes zero, so the strict-null is *not rejected*. The §3.8 falsifier therefore stands; the qualitative ranking — momentum-alone helps less than momentum + curriculum — is reinforced by the C4 → C4_M contrast, where the paired CI [+0.081, +0.123] cleanly excludes zero.

The cleanest explanation of the small mean drift in G1 → G1_M is the §3.8 first-step argument. The velocity buffer is zero at the moment T₁ begins (the optimiser has been resting at θ_0*); the very first step is therefore identical to plain SGD. From the second step onward, the velocity buffer accumulates the *replay-side* gradient as well, so the per-step new-task drift is damped by a factor (1 − β). Step 1 produces the same dip as μ = 0; step 2+ produces a smaller dip. The mean drift we observe is consistent with this two-regime account; whether it sharpens into a statistically robust effect at higher seed counts is open.

**The Pareto picture: momentum is best deployed *with* the curriculum, not as a substitute.** ACC rises cleanly under momentum across every condition tested (G1: 0.873 → 0.928; C4: 0.833 → 0.917) — that effect is well-documented in the deep-learning literature (Polyak, 1964; Sutskever et al., 2013) and survives every seed-paired contrast. Gap-depth reduction under momentum, by contrast, is only large and statistically robust *in the presence of the curriculum*. The curriculum is the operative depth lever; momentum acts as a noise filter on top of it. Adopt the discussion of §6.7 (momentum is the velocity-buffer reading of the *temporal axis* the curriculum also acts on), and the picture becomes: the temporal axis admits two complementary interventions that compose without conflict, but momentum alone has no leverage on a discontinuity that the curriculum has not first smoothed.

## 6.5 The CIFAR-10 NCL Inversion is Predicted by the Framing

The §5.6 surprise — NCL produces more gap depth than vanilla ER on CIFAR-10 — looks at first glance like a contradiction of the thesis's "NCL as path-finding reference" claim. It is in fact *predicted* by the path-finding / landscape-shaping distinction of §1.3, once one looks at what NCL preconditions and what it does not.

**NCL preconditions Linear weights. BatchNorm does not have Linear weights.** ResNet-18 (Appendix B.1.2) uses BatchNorm after every convolution. The running statistics `μ_bn`, `σ_bn²` of BatchNorm layers are not trainable parameters; they are exponential moving averages of the activation statistics. When the input distribution switches from clean CIFAR-10 to `gaussian_noise`-corrupted CIFAR-10 (severity 3 in the Hendrycks & Dietterich, 2019 corruption suite), these running statistics shift discontinuously over the first few mini-batches of T₁, *outside* the parameter manifold that K-FAC builds factors over. NCL has nothing to say about that displacement; vanilla ER, by replaying clean-CIFAR examples, drags the running statistics back toward the clean-CIFAR fixed point.

This is a falsification opportunity that the rot-MNIST testbed (no BN, no input distribution shift in the noise sense) could not have produced. On rot-MNIST, NCL has *all* the structure it needs to precondition; on CIFAR, it has only a fraction of it. The thesis's framing — NCL accepts the discontinuity and navigates it via preconditioning — predicts that NCL's effectiveness degrades when the discontinuity has components outside its preconditioning surface. The CIFAR result is exactly that prediction's empirical signature.

**The curriculum sees no such surface-dependence.** The λ-curriculum is implemented as a scalar weight in the loss; it acts on the gradient regardless of which parameter the gradient targets, and it acts on the BN running-statistic update through the same backward pass. The CIFAR D3 / D4 results (depth 0.027 / 0.031, ACC matched to vanilla ER) confirm that the curriculum's effectiveness is invariant to the architectural surface. This is the operational content of *landscape-shaping* in the sense of §1.3.

A reader who wants to see this as a contradiction of the thesis is reading §1.3 too narrowly. Path-finding is *defined* by what it preconditions, and is therefore brittle to architectural changes that move discontinuities off the preconditioning surface. Landscape-shaping, by acting at the source of the discontinuity, is insensitive to this brittleness. Appendix D unpacks two further contributing factors (empirical-Fisher underestimation in high-confidence directions; lr = 0.1 being aggressive for NCL on ResNet-18), each of which is consistent with the framing.

## 6.6 Predictions That Landed Cleanly vs. Predictions That Needed Refinement

| Prediction | Source | Verdict | Refinement (if any) |
|------------|--------|---------|---------------------|
| Three-contributor decomposition is observable | §3.2, §4.4 | **Clean.** All three contributors quantified; shares 70/20/10 | None |
| `G_traj` is small but non-zero (Kao et al., 2021) | §2.5, §3.2 | **Clean.** G4 = 0.019 ± 0.013; G4_M = 0.013 ± 0.004 | None |
| Linear ramp: depth ↓ monotonically with N | §3.6, RQ1 | **Partial at μ = 0; null at μ = 0.9** | Add §6.3: μ = 0.9 saturates the bound floor; linear shape-blindness limits the effect |
| Adaptive ≈ best linear (RQ2) | §3.6, RQ2 | **Clean.** Adaptive ≥ best linear on both depth and ACC | None |
| λ_min: Pareto trade-off (RQ4) | §3.9 | **Partial at μ = 0; clean at μ = 0.9 and CIFAR** | Effect at μ = 0 is below seed-noise floor; needs the noise-suppressed regime to be visible |
| Momentum + curriculum tightens depth | §3.8, RQ3-i | **Clean.** C4 → C4_M halves depth and shrinks variance 10× | None |
| Momentum alone ≈ no momentum (RQ3-ii) | §3.8, RQ3-ii | **Consistent with the data at the seed-paired level (CI on Δ includes zero).** Means hint at a small favourable drift that may sharpen with more seeds. | §6.4: "first step identical, subsequent steps damped by (1 − β)" remains the predicted mechanism for the mean drift |
| Curriculum carries over to CIFAR + ResNet-18 | §4.7 | **Clean.** D3/D4 halve vanilla ER depth at unchanged ACC | None |
| NCL = path-finding reference | §3.3 | **Clean direction; surprising magnitude inversion on CIFAR** | §6.5: predicted brittleness when discontinuities leave the preconditioning surface |

Six clean, three needing refinement, zero strict-null falsifiers rejected at the seed-paired level (the §3.8 momentum-alone falsifier survives as a non-rejection, with a small mean drift open for confirmation at higher seed counts). The two predictions that needed refinement (linear-ramp behaviour at μ = 0.9, λ_min trade-off at μ = 0) both relate to the same underlying issue: the §3.6 / §3.9 derivations were continuous-time arguments, and the data show that the corresponding effects only become cleanly visible once discrete-time noise is suppressed (i.e., at μ = 0.9 and on the deeper CIFAR backbone). None of those corrections changes the qualitative ranking of methods or the sign of any reported effect.

## 6.7 Two Axes of Softening: NCL as the Spatial Dual of the Curriculum

Chapter 1 (§1.3) and Chapter 2 (§2.3) together contrasted *path-finding* (navigate the discontinuous landscape) with *landscape-shaping* (remove the discontinuity at its source) on the basis of each method's **local** operation, placing CACL, NCL, EWC, GEM, and A-GEM in the path-finding camp. A complementary, **global** reading of NCL specifically — developed here — shows that NCL and the λ-curriculum are two implementations of the same intervention along orthogonal axes.

**The discontinuity has two coordinates.** The combined objective near the boundary is L_replay + λ(t) · L_current with effective Hessian H_replay + λ(t) · H_current. Two ways to make this object behave continuously across t = t_0 are: (i) **smooth the schedule** — let λ(t) ramp continuously from 0 to 1 so that the second term arrives gradually, with the geometry of each step left untouched; (ii) **smooth the geometry** — leave the schedule abrupt, but precondition each step by an old-task curvature metric (loosely H_replay^{-1}) so the gradient field the optimiser experiences is continuous through t_0 even though the underlying loss jumps. The first is the λ-curriculum; the second is NCL. Both produce a continuously-evolving effective optimisation regime; they differ only in which coordinate of the (θ, t) plane they act on.

**The trust-region view makes the duality concrete.** A penalty λ ‖θ − θ_0*‖²_F induces an effective trust region around θ_0* with stiffness λ and shape F. NCL holds λ fixed and reshapes F to be anisotropic — narrow along high-old-curvature directions, wide along directions the old task does not constrain (Kao et al., 2021, §3). The curriculum holds F = I and modulates λ — a spherical trust region whose stiffness ramps from low to high. NCL is *anisotropic and constant in time*; the curriculum is *isotropic and varying in time*. They populate orthogonal entries of the same {shape × strength} parameterisation of how strongly the old solution constrains motion at the boundary.

**Each method is explicit about exactly the axis the other treats implicitly.** NCL is explicit about geometry (a metric computed and applied at every step) and implicit about time (fixed λ across the boundary). The curriculum is explicit about time (a deliberate schedule of λ across a small window) and implicit about geometry (identity metric, plain SGD). Under this reading, the empirical claim that ER + curriculum reaches NCL's depth on rot-MNIST and *beats* NCL's depth on CIFAR becomes a consequence of axis selection, not of MLP-on-MNIST tuning: the temporal axis is one-dimensional, free of curvature estimation, active only near t_0, and additive on top of any optimiser. The spatial axis is high-dimensional, requires a Fisher or curvature estimate, and is active at every step. To first order both axes can carry the same intervention; the temporal axis is simply cheaper to operate on and more robust to which surface the discontinuity sits on (the §6.5 argument).

**The Hessian-along-trajectory is the shared mechanism.** What an SGD optimiser experiences across the boundary is the Hessian along its path through (θ, t). NCL keeps that Hessian-along-trajectory continuous through preconditioning — the new-task gradient component along high-old-curvature directions is damped at every step, so the curvature seen does not jump even when H_current does. The curriculum keeps that Hessian-along-trajectory continuous through scheduling — λ(t) · H_current arrives in instalments rather than all at once, so the curvature evolves continuously even though H_current changes abruptly underneath. Same end-state, two routes.

**A unified intervention is the obvious composition.** Anisotropic preconditioning combined with time-varying λ would compose without conflict: the metric handles *which* directions are dangerous, the schedule handles *when* the constraint is binding. We do not pursue this here — on rot-MNIST the curriculum already absorbs the depth that NCL also targets — but on harder benchmarks where G_traj is plausibly larger (§6.8) or where the discontinuity moves off the K-FAC surface (the CIFAR BatchNorm case), the composition is the natural design and is flagged as future work (§6.9).

## 6.8 Limitations and Threats to Validity

**(L1) Task count.** Both testbeds are two-task. The decomposition argument and the curriculum claims are clean in this regime, but the long-task-sequence behaviour of NCL — where Kao et al. (2021)'s original SOTA claims live — is not probed here. The discussion of §6.5 (NCL on CIFAR) is therefore explicitly a *two-task* result; whether NCL recovers its rot-MNIST advantage on 10-task R-MNIST is open.

**(L2) Seed count.** Five seeds is sufficient for the consistent-direction criterion of §C.2 but insufficient for the conventional `α = 0.05` Wilcoxon threshold (smallest attainable two-sided p is 0.0625). We report this transparently throughout Chapter 5; readers wanting stricter significance would need additional seeds. The headline differences (e.g. C4 vs. G1) are large enough that 20 seeds would not change the rankings, but the smaller deltas (e.g. C5 vs. C4 at μ = 0) might dissolve under further sampling.

**(L3) NCL implementation gap vs. reference code.** Three deliberate deviations from Kao et al. (2021)'s Algorithm 1 are documented in Appendix D (`_has_prior` gating on task 0, empirical Fisher rather than sampled Fisher, end-of-task K-FAC snapshot rather than EMA accumulation). The α sweep that landed on `prior_init = 0.1` was performed against this implementation; whether the closer-to-paper variant would shift the NCL gap_depth on either testbed is not tested. The §6.5 argument depends on NCL's preconditioning surface (Linear weights only) rather than on these implementation choices, so the qualitative comparison should be robust; the specific quantitative numbers for D2 might move.

**(L4) Adiabatic-regime assumption.** §3.7's assumption analysis acknowledges that the bound is sharply testable only on rot-MNIST + MLP — the regime where the Hessian along the optimum path is approximately quadratic. On ResNet-18 + CIFAR the assumption is known to weaken; the §5.6 results are therefore a *direction-of-effect* check, not a quantitative confirmation of the O(1/N) bound. We do not over-claim from the CIFAR numbers.

**(L5) Buffer-fidelity diagnostics off for the C-block.** The curriculum block reports `gap_depth` and ACC but not the per-step `cos(g_replay, g_true)` series that the G-block provides. The interpretation of the adaptive schedule in §5.3 ("λ stays low while the iterate is far from local equilibrium") is consistent with the G-block's mean cosine of 0.71 but is *inferred*, not directly measured, on the curriculum runs themselves. Enabling the diagnostic on a focused sub-block of the C-series is a cheap experiment that would close this loop and is flagged in §6.9.

**(L6) `recovery_steps` censoring.** Many G-block conditions report "none recovered" within the 500-step dense window (Appendix E.R1). This is not a recovery failure in the practical sense — final ACC is high — but a 90 %-of-pre-switch threshold that the model is slow to cross when pre-switch accuracy is already high. We report it transparently in §C.5 but note that the metric is more informative on shallower-depth regimes (e.g. CIFAR) than on the deepest-depth G1 / G2 conditions.

## 6.9 Future Work

The findings above suggest five concrete lines of follow-up, ordered from cheapest to deepest.

**(F1) Buffer-fidelity diagnostics on the C-block.** Toggle `GRAD_DIAG=on` in `scripts/run_curriculum.sh` for C4 / C7 (the two adaptive variants) on rot-MNIST. This produces the per-step λ(t) curve alongside `cos(g_replay, g_true)` and `‖g_r‖/‖g_t‖` and would directly verify the mechanistic claim in §5.3. Roughly +50 % runtime per condition; trivial in design.

**(F2) Decomposition on CIFAR-10.** The G2-G4 path was not replicated on CIFAR-10 because of the 60 k full-data buffer requirement (50 k for CIFAR-10 trainset). Running an analogous D2-D4 with `replay_full_buffer=true` and `memory.total_budget=50000` would quantify which contributors carry over to ResNet-18 and which do not. The expectation, from the §6.5 argument, is that the BatchNorm displacement contributes a *new* contributor outside the §4.4 three-contributor frame.

**(F3) NCL + lr warm-up.** The Appendix D §D.5 prediction is that a small first-task-switch lr warm-up would close some of the CIFAR NCL gap-depth gap. The cleanest experiment is D2_warmup with a 200-step lr warm-up on each task-1+ — a single new condition on the existing CIFAR block.

**(F4) Composition of curriculum + NCL.** §6.7 closes by flagging the natural composition: anisotropic preconditioning + time-varying λ. This would be implemented as `method=ncl method.lambda_curriculum.enabled=true` once a curriculum hook is added to the NCL trainer — a small implementation change followed by a single sweep on both testbeds.

**(F5) Long task sequences.** The two-task results say nothing about 10-task P-MNIST or R-MNIST. Kao et al. (2021)'s NCL advantage and the various continual-learning rankings (Buzzega et al., 2020; Prabhu et al., 2020; Boschini et al., 2022) live in that regime. Whether the adaptive curriculum maintains or sharpens its lead over NCL as task count grows is the most important untested question raised by this thesis.

## 6.10 References

- Aljundi, R., Kelchtermans, K., & Tuytelaars, T. (2019). *Task-Free Continual Learning.* CVPR 2019.
- Bang, J., Kim, H., Yoo, Y., Ha, J.-W., & Choi, J. (2021). *Rainbow Memory: Continual Learning with a Memory of Diverse Samples.* CVPR 2021.
- Boschini, M., Buzzega, P., Bonicelli, L., Porrello, A., & Calderara, S. (2022). *Class-Incremental Continual Learning into the eXtended DER-verse.* IEEE TPAMI.
- Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). *Dark Experience for General Continual Learning: a Strong, Simple Baseline.* NeurIPS 2020.
- De Lange, M., van de Ven, G., & Tuytelaars, T. (2023). *Continual evaluation for lifelong learning: Identifying the stability gap.* ICLR 2023.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking neural network robustness to common corruptions and perturbations.* ICLR 2019.
- Hess, T., Mundt, M., Pliushch, I., & Ramesh, V. (2023). *A second perspective on continual learning.* (preprint).
- Kao, T.-C., Jensen, K. T., van de Ven, G. M., Bernacchia, A., & Hennequin, G. (2021). *Natural Continual Learning: Success is a Journey, Not (Just) a Destination.* NeurIPS 2021.
- Koh, H., Kim, D., Ha, J.-W., & Choi, J. (2022). *Online Boundary-Free Continual Learning by Scheduled Data Prior.* ICLR 2022.
- Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-Factored Approximate Curvature.* ICML 2015.
- Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics 4(5), 1–17.
- Prabhu, A., Torr, P. H. S., & Dokania, P. K. (2020). *GDumb: A Simple Approach that Questions Our Progress in Continual Learning.* ECCV 2020.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the importance of initialization and momentum in deep learning.* ICML 2013.
