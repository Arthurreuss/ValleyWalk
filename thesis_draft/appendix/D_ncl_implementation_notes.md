# Appendix D — NCL Implementation Notes

NCL (Kao et al., 2021) appears in this thesis as a *reference path-finding method* — the strongest second-order baseline the curriculum is compared against. The behaviour of our NCL implementation has a direct bearing on the interpretation of the G-series decomposition (§4.4) and on the surprising CIFAR-10 result that NCL produces a *larger* gap than vanilla ER (§4.7, Appendix E.R3). This appendix documents the implementation choices behind NCL, the small hyperparameter sweep that settled the `prior_init` default, and the structural reasons why ER outperforms NCL on the two-task rot-MNIST testbed.

The full session notes are preserved under `thesis_draft/notes/ncl_implementation_findings.md`.

## D.1 Algorithm and Source

Our NCL closely follows Algorithm 1 of Kao et al. (2021), "Natural Continual Learning: Success is a Journey, Not (Just) a Destination" (NeurIPS 2021). The natural-gradient step is realised through K-FAC (Kronecker-Factored Approximate Curvature; Martens & Grosse, 2015) factors `A_l ⊗ G_l` per `nn.Linear` layer, accumulated end-of-task into a running prior

```
Λ_k  =  α · I  +  Σ_{i < k} F_i
```

with `F_i` the task-i K-FAC Fisher block and α the prior strength `method.ncl.prior_init`. The update at parameter θ is the trust-region form of Kao et al. (2021) Eq. (8):

```
θ_{t+1}  =  θ_t  −  η · Λ_k⁻¹ · ( ∇L_current(θ_t)  +  Λ_{k-1} · (θ_t − μ_{k-1}) )
```

where `μ_{k-1}` is the mean of the past-task Laplace posterior (i.e., the parameter at the end of task k − 1). Only `nn.Linear` layers carry K-FAC factors; batch-norm and embedding parameters are updated with the raw gradient unchanged. Implementation in `src/methods/ncl.py`.

## D.2 Deviations from the Paper

Three implementation choices deviate from a literal reading of the paper's pseudocode.

**(D2.1) Prior gating on task 0.** A paper-literal Eq. (8) at task 0 produces an active "rubber-band" term `Λ_0 · (θ − μ_0) = α · (θ − θ_init)` that drags θ back to its random initialisation every step. With η_eff ≈ 1 (lr 0.1, momentum 0.9) this dominates the gradient signal and prevents task 0 from learning. We therefore introduce a `_has_prior` flag that is `False` on task 0 (plain SGD pass-through) and flips to `True` once `end_task` has produced a real Fisher estimate. This is consistent with how most reference implementations behave in practice but is not literally what Algorithm 1 prescribes.

**(D2.2) Empirical Fisher (gradient against true labels) rather than proper Fisher.** The reference implementation `tachukao/ncl` samples labels from the model's softmax to construct the Fisher; we use the empirical Fisher with the true labels. The empirical Fisher underestimates curvature in directions where the model is highly confident — precisely the directions where past-task protection matters most — and is therefore expected to weaken NCL on later tasks. We retain it for two reasons: it is the standard approximation in continual-learning K-FAC implementations and it removes one degree of stochasticity that complicates per-step diagnostics. The cost of the approximation is acknowledged in §6.

**(D2.3) End-of-task K-FAC snapshot rather than EMA accumulation.** `F_k` is computed once at `end_task` over up to `fisher_samples = 1000` examples, rather than maintained as a moving average during task k. This avoids interaction between the Fisher estimate and the optimiser's velocity buffer, at the cost of using a fewer-samples snapshot.

All three deviations are flagged in Chapter 6's limitations discussion and would each support a follow-up study; none of them changes the qualitative direction of the comparisons made in Chapter 5.

## D.3 Tuning `prior_init` (α)

Algorithm 1's Gaussian prior `p_w = α · I` is the precision of the very-first Laplace approximation. Its value matters because it bounds `Λ_k⁻¹`: with `Λ_k = α · I + Σ F_i`, the natural gradient cannot amplify the raw gradient beyond `1/α` in any direction. The paper-literal default `α = 1.0` corresponds to a unit-variance Gaussian prior on the weights.

We ran a small one-dimensional sweep over α on rot-MNIST + MLP (the testbed of §4.1):

| Configuration | ACC | FORG | Interpretation |
|---------------|-----|------|----------------|
| α = 10.0 | low | high | strong rubber-band → over-regularises |
| α = 1.0 (paper default) | mid | mid | reasonable but not best |
| α = 0.1 | **best** | **best** | weak rubber-band → enough plasticity |
| α ≤ 0.03 | diverges at lr = 0.1 | — | `1/α` amplifies K-FAC-underestimated directions until SGD blows up |

The qualitative picture is that α is a **directionally surgical** plasticity knob:

- In directions where `F_0` is large (task-0-important), `Λ ≈ F_0 ≫ α · I`. α barely affects the natural-gradient step here — stability is preserved regardless of α.
- In directions where `F_0` is small (task-0-irrelevant), `Λ ≈ α · I`, so `Λ⁻¹ ≈ (1/α) · I`. Lowering α amplifies the step *only* in directions task 0 does not care about — exactly where new tasks want to move.

So lowering α adds plasticity without much stability cost — until the amplification gets large enough that K-FAC's misestimates of "low-curvature" directions begin to matter, which is the divergence floor observed at α ≤ 0.03 with lr = 0.1.

**Default in the thesis: α = 0.1.** Pinned both in `configs/method/ncl.yaml` and explicitly on the NCL row of every sweep that uses it (G-series in §A.1, D2 in §A.3). This default is *tuned for rot-MNIST*; the analogous sweep on CIFAR-10 + ResNet-18 has not been performed.

## D.4 Why ER Outperforms NCL on Two-Task rot-MNIST

The G-series decomposition (Appendix E.R1) reports `gap_depth(NCL) = 0.072` at μ = 0 against `gap_depth(G1) = 0.195` — NCL wins on the gap measure. But on the headline `ACC` it is the other way around: `ACC(NCL) = 0.770` vs. `ACC(G1) = 0.873`. Three structural reasons explain the ACC ranking, and they are independent of any of the implementation choices in §D.2.

**(i) Two tasks is the wrong regime for NCL.** With only one task to remember, ER with a 1 000-sample reservoir buffer effectively memorises task 0. There is no room for a regularisation-only method to add value. NCL's published SOTA claims (Kao et al., 2021, Table 1) are on long task sequences (≥ 10 tasks) with *constrained* ER buffers; that is the regime where preconditioning beats simple replay.

**(ii) The community consensus since ~2022 favours replay.** Subsequent work in continual learning — DER (Buzzega et al., 2020), DER++ (Boschini et al., 2022), GDumb (Prabhu et al., 2020) — has repeatedly shown that a well-tuned replay method with an adequate buffer beats every regularisation-only method, NCL included. NCL's modern value is as the best *regularisation-only* baseline, not as a candidate for "best CL method overall".

**(iii) Our two-task setup uses an MLP without batch-norm.** The Hessian structure that K-FAC approximates is unusually clean here — but it is also unusually small. A bigger network with batch-norm (ResNet-18 on CIFAR, §A.3) breaks the per-layer Kronecker assumption that K-FAC relies on, which is one component of the gap-depth inversion on CIFAR documented next.

## D.5 The CIFAR-10 NCL Gap Inversion

On CIFAR-10 (Appendix E.R3), NCL produces `gap_depth = 0.180` against vanilla ER's `0.071` — the opposite direction of the rot-MNIST result. Three contributing factors, all consistent with the analysis above:

1. **The `_has_prior` gating + ResNet-18's BatchNorm.** Even after `end_task`, the BatchNorm running statistics shift abruptly when the input distribution switches from `none` to `gaussian_noise`. The K-FAC Fisher only sees Linear layers, so the BN-driven first-step displacement is *not* preconditioned away. NCL has nothing to say about a discontinuity that lives outside Linear weights.
2. **Empirical-Fisher underestimation in confident regions.** A ResNet-18 trained for 10 epochs on clean CIFAR-10 is confident on most training samples; the empirical Fisher in those regions is small, so the corresponding directions of `Λ⁻¹` have large eigenvalues. The first new-task step lands far from the trust region, deepening the transient dip.
3. **lr = 0.1 is too aggressive for NCL on ResNet-18.** Gap depth scales linearly with η under the first-step argument of §2.5; ER tolerates this lr through replay, NCL does not. Smaller lr or a first-task-switch lr warm-up would shrink the gap but is out of scope for this thesis's CIFAR block.

These observations re-frame the CIFAR-10 NCL result as a *prediction* of the framing of this thesis (the discontinuity-survives-preconditioning argument of Chapter 3, applied to a setting where K-FAC's assumptions are weaker), not as a contradiction of it. Chapter 6 picks up this thread.

## D.6 Diagnostics Logged Per Step

`NCL.get_step_diagnostics()` exposes two scalars logged per step in each task's `task_NN_train.csv`:

- `kl_proxy` = ½ (θ − μ)ᵀ Λ (θ − μ) — the quadratic form of the rubber-band penalty itself. A proxy for KL between the current posterior approximation and the prior. Rises smoothly during a task (θ adapting to new data); a spike is the earliest warning of numerical trouble.
- `tr_scale` = min(1, r / ‖step‖_Λ) — the trust-region scale of Kao et al. (2021) Eq. (7). We compute it but do not clip the update. Stays at 1.0 in healthy runs; dropping below 0.3 signals imminent divergence.

Both scalars are useful for debugging an NCL run but are not used in any of the headline metrics of Chapters 5–6.

## D.7 References for this Appendix

- Kao, T.-C., Jensen, K. T., van de Ven, G. M., Bernacchia, A., & Hennequin, G. (2021). *Natural Continual Learning: Success is a Journey, Not (Just) a Destination.* NeurIPS 2021.
- Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-Factored Approximate Curvature.* ICML 2015.
- Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). *Dark Experience for General Continual Learning: a Strong, Simple Baseline.* NeurIPS 2020.
- Boschini, M., Buzzega, P., Bonicelli, L., Porrello, A., & Calderara, S. (2022). *Class-Incremental Continual Learning into the eXtended DER-verse.* IEEE TPAMI.
- Prabhu, A., Torr, P. H. S., & Dokania, P. K. (2020). *GDumb: A Simple Approach that Questions Our Progress in Continual Learning.* ECCV 2020.
- De Lange, M., van de Ven, G., & Tuytelaars, T. (2023). *Continual evaluation for lifelong learning: Identifying the stability gap.* ICLR 2023.
