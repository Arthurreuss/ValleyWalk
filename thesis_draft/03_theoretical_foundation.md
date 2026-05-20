# Chapter 3 — Theoretical Foundation

This chapter develops the mathematical foundation for the λ-curriculum. §3.1 introduces the homotopy family that the curriculum implements and states the assumptions used throughout. §3.2 decomposes the stability gap into three additive contributors — magnitude asymmetry, estimator bias, and trajectory effect — and identifies which of them the curriculum is designed to address. §3.3 formalises the *path-finding vs. landscape-shaping* distinction that places the curriculum in a different family of interventions from NCL, EWC, and GEM. §3.4 proves that the replay loss is monotone non-decreasing along the optimum path of the homotopy (the envelope-theorem inequality, Part A). §3.5 proves that the SGD trajectory tracks the optimum path with error O(1/N) in the adiabatic limit (Part B). §3.6 combines the two parts into an O(1/N) bound on the trajectory contribution to the stability gap. §3.7 discusses the regime in which the bound is expected to be tight. §3.8 and §3.9 derive the momentum and λ_min refinements within the same framework.

The chapter is theory only; the empirical testing of every prediction made here is in Chapter 4 (design) and Chapter 5 (results).

## 3.1 Setup: The Homotopy Family L_λ

Standard online continual learning trains the model on a sequence of objectives that *change discontinuously*. During T₀, the optimiser descends L_replay = L_T₀ alone. At the moment T₁ begins, the objective switches in a single step from L_replay to L_replay + L_current = L_T₀ + L_T₁. The parameter θ does not move during the switch, but the loss surface the optimiser is following — and the location of its minimum — does. We refer to this as **landscape teleportation**.

The λ-curriculum replaces this discontinuous switch with a one-parameter family of losses

```
L_λ(θ) = L_replay(θ) + λ · L_current(θ),    λ ∈ [0, 1]                  (Def. L_λ)
```

with λ = 0 recovering pure replay (optimum at θ_0* := argmin L_replay) and λ = 1 recovering the joint loss (optimum at θ_joint* := argmin L_replay + L_current). The curriculum ramps λ along a continuous schedule

```
λ(t) = clip(t / N, 0, 1)                                                (Linear schedule)
```

over the first N steps of the new task; for t > N the schedule plateaus at 1 and training proceeds as standard ER.

Two trajectories through parameter space play a role in what follows:

- **Optimum path** Θ*(λ) := argmin_θ L_λ(θ). Under regularity assumptions (smoothness of L_replay and L_current, positive-definite Hessian along the path), Θ*(λ) is a continuously differentiable curve in λ ∈ [0, 1] connecting θ_0* (at λ = 0) to θ_joint* (at λ = 1).
- **SGD trajectory** θ(t). The actual sequence of iterates produced by gradient descent on L_λ(t)(θ) with step size η. In the continuous limit η → 0, this becomes the ODE dθ/dt = −∇L_λ(t)(θ).

**Assumptions used throughout the chapter.**

(A1) *Smoothness*: L_replay and L_current are C² with locally Lipschitz Hessians on a neighbourhood of the optimum path.
(A2) *Strict minima*: H_λ(Θ*(λ)) ≻ 0 for every λ ∈ [0, 1].
(A3) *Slow homotopy*: τ_drive := N is large compared to the local equilibration time τ_eq(λ) := 1 / μ_min(H_λ(Θ*(λ))).

§3.7 discusses how well these assumptions hold on rot-MNIST + MLP, the regime in which the bound is sharply testable.

## 3.2 Three-Contributor Decomposition of the Stability Gap

The remainder of the chapter develops a bound on the *trajectory* contribution to the stability gap and shows it vanishes under the curriculum. Before the bound, we name what it is bounding and what it is not. This section decomposes the gap of vanilla ER into three additive contributors and identifies which one the envelope-theorem argument addresses.

**Setup at the task boundary.** Consider an SGD step under vanilla ER at the first iteration of T₁. The parameter θ_0* is a stationary point of L_replay alone (it has been trained to convergence on T₀), so ∇L_replay(θ_0*) ≈ 0. The vanilla-ER update is

```
θ₁ = θ_0* − η · (ĝ_replay(θ_0*) + g_new(θ_0*))                          (12)
```

where g_new(θ) := ∇L_current(θ) is the deterministic new-task gradient at θ, and ĝ_replay(θ) is the *finite-buffer estimator* of ∇L_replay(θ) — the gradient computed on a 256-sample mini-batch drawn from the 1 k reservoir buffer. The "true" past-task gradient at θ — the gradient on the *full* T₀ training set — we denote g_replay(θ); by stationarity g_replay(θ_0*) ≈ 0. Decompose the buffer estimate around the true gradient:

```
ĝ_replay(θ_0*) = g_replay(θ_0*) + ε(θ_0*) ≈ ε(θ_0*)                     (13)
```

where ε is a zero-mean (in expectation over the buffer) noise term capturing the finite-buffer estimator gap. Substituting (13) into (12):

```
θ₁ ≈ θ_0* − η · (ε(θ_0*) + g_new(θ_0*))                                 (14)
```

so the first step under vanilla ER has *three* features, each contributing a distinct mechanism to the subsequent T₀ accuracy dip.

**(C1) Magnitude asymmetry.** At θ_0*, g_new(θ_0*) is generically O(1) — the new-task gradient has not been zeroed by training on T₁ — while ε(θ_0*) is mean-zero and O(1/√k) where k is the buffer size: it is bounded in expectation but does not have an O(1) "restoring direction" toward θ_0*. The two terms in (14) are therefore *asymmetric in magnitude*: in expectation, the new-task term dominates the first step by a factor of order √k. The displacement at step 1 is essentially −η · g_new(θ_0*), with the replay term acting as a small zero-mean perturbation.

In the framing of the second-order Taylor expansion of L_replay around θ_0*,

```
L_replay(θ₁) − L_replay(θ_0*) ≈ ½ · η² · g_new(θ_0*)ᵀ · H_replay(θ_0*) · g_new(θ_0*) > 0   (15)
```

whenever H_replay has any positive curvature in the direction of g_new — which it generically does at a strict local minimum. This is the magnitude-asymmetry contributor; we write its share of the observed gap as **G_mag**.

**(C2) Estimator bias.** Even if ε is mean-zero, on any individual run ε ≠ 0 and contributes a stochastic kick to θ₁. Aljundi et al. (2019) frame this as a *buffer-as-estimator* gap: a 1 k reservoir buffer is a finite-sample estimator of the past data distribution, and the per-step error ε produces a depth contribution that does not vanish under any schedule because it is independent of the loss composition. We write its share of the observed gap as **G_est**.

The cleanest empirical isolation of G_est is the contrast between vanilla ER (using ĝ_replay on a 256-sample buffer mini-batch) and a *full-data* ER condition that uses the entire past-task training set at every step (`replay_full_buffer=true`, `memory.total_budget=60000` in §4.4): in the full-data condition ĝ_replay = g_replay exactly, so ε ≡ 0 by construction. Any residual difference in gap depth between the two conditions is G_est.

**(C3) Trajectory effect.** Even if ε ≡ 0 and the magnitude asymmetry is removed (e.g. by gradient balancing), an additional residual remains. This is the geometric observation of Kao et al. (2021) reproduced in §2.5: with the *exact* joint-loss gradient and an instantaneous switch from λ = 0 to λ = 1, the SGD trajectory still bends off the steepest-descent path of L_joint because of the discrete step size. The first joint-loss step lands at a point that is not on Θ*(1), and L_replay along that off-path trajectory is strictly higher than L_replay(Θ*(1)). We write its share as **G_traj**.

The envelope-theorem argument of §3.4–§3.6 quantifies G_traj specifically: it shows that a *continuous* λ-schedule (rather than the discontinuous jump from 0 to 1) eliminates G_traj in the continuous-time limit, with the residual scaling as O(1/N) at finite N.

**Additivity (approximate).** The three mechanisms above are not independent in general — for instance, magnitude asymmetry at step 1 displaces the iterate, which then changes the effective per-step estimator error on step 2. To first order in η and in the buffer size, however, each contributor acts at its own characteristic step:

- G_mag is set at *step 1* by the first-step Taylor argument (15).
- G_est is set across *all steps* by the buffer-sampling noise.
- G_traj is set across the *first ≈ τ_eq / η steps* by the off-path SGD geometry of §3.6.

The empirical decomposition of §4.4 (the G1 → G2 → G3 → G4 ablation path) uses exactly this structure: G1 contains all three; G3 removes G_mag by normalisation; G2 removes G_est by replacing the buffer estimator with the full past-task gradient; G4 removes both, leaving G_traj as the residual. The shares reported in §5.1 (≈ 70 % magnitude, ≈ 20 % estimator, ≈ 10 % trajectory at μ = 0) are obtained from this design.

**Scope of the chapter's bound.** From §3.4 onward, the derivation addresses G_traj. The bound does not say anything about G_mag (other than that the curriculum's continuity removes the discrete-time first-step displacement of (15) — see §3.6) or about G_est (which is an estimator property of the buffer, independent of any schedule). Chapter 5 evaluates whether the curriculum-plus-momentum combination is effective on all three contributors empirically; the formal claim of this chapter is restricted to the trajectory term.

## 3.3 Path-Finding vs. Landscape-Shaping

A second framing organises continual-learning methods by *what they do with the discontinuity* of L_λ across the task boundary, rather than by what curvature information they use. Two families result.

**Path-finding.** Accept the discontinuity. The objective the optimiser sees still jumps from L_replay to L_replay + L_current at t = t₀; the intervention is to make the SGD step robust to that jump by reshaping *how* the optimiser moves in parameter space. Formally, path-finding methods modify the update rule from plain SGD to a preconditioned form

```
θ_{t+1} = θ_t − η · M(θ_t) · ĝ_t                                        (P)
```

where M(θ) is some second-order quantity built from past-task data — the inverse Fisher (EWC; Kirkpatrick et al., 2017), the precision matrix Λ_k = α·I + Σ F_i (NCL; Kao et al., 2021), or a projection onto a feasible set (GEM; Lopez-Paz & Ranzato, 2017) — and ĝ_t is the SGD gradient on the current objective. The schedule of when L_current "switches on" is unchanged; only the geometry of the step is altered.

Gradient balancing (the "balanced" mode of ER used in §4.4) is path-finding in the same sense: it leaves the schedule discontinuous but rescales the gradient so the first step is not dominated by g_new. It does this with M = (per-component unit-norm projection) instead of a Hessian-derived metric, so it sits at the cheap end of the family.

**Landscape-shaping.** Remove the discontinuity. The intervention modifies the *objective itself* in a small neighbourhood of t₀ so the surface the optimiser is descending is continuous through the task boundary. The λ-curriculum is the canonical example:

```
θ_{t+1} = θ_t − η · ĝ_t,    where ĝ_t = ∇(L_replay + λ(t) · L_current)(θ_t)   (S)
```

with λ(t) a continuous schedule from 0 to 1 over t ∈ [t₀, t₀ + N]. The geometry of the step is left to SGD's natural step-size adaptation; the change is purely in the *target* the gradient points at. (S) and (P) act on *different coordinates* of the (objective, geometry) plane.

**Why this distinction matters operationally.** Three differences fall out directly from the (P) vs. (S) split:

1. *Surface dependence.* Path-finding methods depend on the surface their preconditioner M is built over. NCL's K-FAC factors are computed on `nn.Linear` weights only; gradient balancing acts on the flattened gradient of every parameter. Landscape-shaping has no such surface — λ(t) multiplies *the loss*, and the modification flows through the backward pass to every parameter without discriminating between BatchNorm running statistics, Linear weights, and convolutional kernels. §6.5 takes this up as the structural explanation of the CIFAR NCL inversion: K-FAC cannot see the BN-driven discontinuity, the curriculum can.

2. *Step-size adaptation.* Plain SGD on a smooth loss has a natural step-size adaptation: large gradient → large step, small gradient → small step. Any preconditioning M that normalises the gradient (the per-component unit-norm projection of balanced ER; the inverse-Hessian rescaling of NCL when M is large in low-curvature directions) breaks this adaptation. Landscape-shaping leaves it intact. This is the operational source of the "balanced-ER ACC tax" observed in §5.1 (G1 → G3: ACC drops by ≈ 0.024 across all five seeds).

3. *Cost per step.* (P) requires a per-step matrix-vector product M·ĝ, and the construction of M typically requires an end-of-task or per-step Fisher estimate. (S) adds one scalar multiply per backward pass. The temporal axis is therefore much cheaper than the spatial axis in implementation as well as in conceptual machinery.

**A complementary spatial/temporal reading of NCL.** §6.7 develops a finer point that is worth previewing here: although path-finding and landscape-shaping look orthogonal as defined above, a global reading of NCL specifically can also be cast as a *landscape-shaping* intervention along the spatial axis — NCL's preconditioner keeps the Hessian-along-trajectory continuous through the task boundary, in the same way the curriculum keeps the loss-along-trajectory continuous. NCL is then "landscape-shaping in parameter space"; the curriculum is "landscape-shaping in time". The two are complementary entries of a {shape × strength} parameterisation of how strongly the old solution constrains motion at the boundary. Chapter 6 develops this in detail; the §3.3 distinction is sufficient for everything that follows in this chapter.

**Predictive consequence used downstream.** The path-finding / landscape-shaping split implies an empirical ranking: on a setting where the discontinuity is confined to the preconditioning surface (e.g. rot-MNIST + MLP without BatchNorm), path-finding and landscape-shaping should produce comparable gap depths; on a setting where the discontinuity has components outside the preconditioning surface (e.g. CIFAR + ResNet-18 with BatchNorm), landscape-shaping should outperform path-finding. §5.6 confirms this directly.

## 3.4 Part A — The Optimum-Path Inequality

**Claim.** d/dλ L_replay(Θ*(λ)) ≥ 0 for all λ ∈ [0, 1].

At the optimum Θ*(λ), the first-order condition is

```
∇_θ L_λ(Θ*(λ)) = ∇L_replay(Θ*(λ)) + λ · ∇L_current(Θ*(λ)) = 0           (FOC)
```

By the chain rule,

```
d/dλ L_replay(Θ*(λ)) = ∇L_replay(Θ*(λ))ᵀ · dΘ*/dλ                       (1)
```

Differentiating (FOC) with respect to λ implicitly:

```
H_λ(Θ*(λ)) · dΘ*/dλ + ∇L_current(Θ*(λ)) = 0                             (2)
```

where H_λ := ∇²_θ L_λ = H_replay + λ · H_current. Solving (2):

```
dΘ*/dλ = −H_λ(Θ*(λ))⁻¹ · ∇L_current(Θ*(λ))                              (3)
```

The inverse exists by (A2).

Substituting (3) into (1) and using ∇L_replay = −λ · ∇L_current from (FOC):

```
d/dλ L_replay(Θ*(λ)) = λ · ∇L_current(Θ*(λ))ᵀ · H_λ⁻¹ · ∇L_current(Θ*(λ))   (4)
```

The right-hand side of (4) is non-negative for every λ ∈ [0, 1]: λ ≥ 0, H_λ⁻¹ ≻ 0, so the quadratic form is non-negative.

**Conclusion of Part A.** L_replay is monotone non-decreasing along the optimum path Θ*(λ). Consequently L_replay(Θ*(λ)) ≤ L_replay(θ_joint*) for all λ ∈ [0, 1] — *no overshoot is possible on the optimum path*.

## 3.5 Part B — Adiabatic Tracking

The optimum-path inequality of Part A is about Θ*(λ), not the SGD iterates θ(t). To connect them we estimate how closely θ(t) tracks Θ*(λ(t)) in the adiabatic limit (A3).

Consider gradient flow

```
dθ/dt = −∇_θ L_λ(t)(θ(t))                                               (5)
```

with λ(t) increasing from 0 to 1 over [0, N] (in continuous time). Write θ(t) = Θ*(λ(t)) + δ(t) and Taylor-expand:

```
∇L_λ(t)(θ(t)) = ∇L_λ(t)(Θ*(λ(t))) + H_λ · δ + O(‖δ‖²)
              = 0 + H_λ · δ + O(‖δ‖²)                                   (6)
```

(the first term vanishes by FOC). The dynamics of δ become

```
dδ/dt = −H_λ · δ − dΘ*/dt + O(‖δ‖²)
      = −H_λ · δ − (dΘ*/dλ) · (1/N) + O(‖δ‖²)                           (7)
```

This is a linear ODE driven by a forcing term proportional to 1/N. The steady-state amplitude of δ is therefore O(τ_eq / τ_drive) = O(1/(N · μ_min)). For any fixed task (μ_min > 0 fixed by A2):

```
‖θ(t) − Θ*(λ(t))‖ = O(1/N)                                              (8)
```

so the SGD trajectory converges uniformly to the optimum path as N → ∞.

## 3.6 Combined Bound: The Stability-Gap as O(1/N)

The trajectory contribution to the gap depth is bounded by the maximum of L_replay along the SGD trajectory minus its baseline. Taylor-expanding L_replay around Θ*(λ(t)) and substituting (8):

```
sup_t L_replay(θ(t)) ≤ L_replay(θ_joint*) + O(1/N)                      (9)
```

The first term is the *unavoidable* steady-state increase from joint training: the model trades a small amount of T₀ fit for T₁ capacity at the joint optimum. The O(1/N) term is the *trajectory* component (G_traj of §3.2) that the curriculum eliminates as N grows.

**Conclusion.** In the continuous-limit, perfect-gradient regime, the trajectory contribution to the stability gap vanishes as N → ∞. The remaining gap is the genuine plasticity–stability trade-off at the joint optimum, not a transient artefact of trajectory geometry.

This is the prediction tested in Chapter 5: under the linear λ-curriculum on rot-MNIST, gap depth should fall monotonically with N at fixed step budget, while accuracy on the joint problem should remain close to that of vanilla ER (since the bound preserves the natural step-size adaptation of SGD — the §3.3 landscape-shaping argument).

## 3.7 Regime of Applicability

The bound (9) is derived under (A1)–(A3). These hold to a good approximation on rot-MNIST + small MLP:

- The MLP has ~10⁵ parameters; near a converged optimum the loss is approximately quadratic in a low-dimensional active subspace and H_λ is well-conditioned in that subspace.
- T₀ (0°) and T₁ (90°) MNIST are in the same input space and label set; the joint optimum is connected to θ_0* by a smooth path with no architectural mismatch.
- The displacement ‖θ_joint* − θ_0*‖ is small relative to the curvature scale (the two tasks share most features), so δ stays in the regime where the linearisation (6) is accurate.

Under these conditions the bound (9) is tight enough that even modest N (50–200 steps, on a ≈235-step task budget) should produce a measurable depth reduction. As N approaches the budget, the O(1/N) term shrinks visibly while the unavoidable joint-optimum term remains constant — the predicted monotone-decreasing depth curve.

On harder benchmarks (deeper networks, larger task shifts) the assumptions weaken: H_λ is poorly conditioned in deep ReLU networks, dΘ*/dλ may pass through ill-conditioned regions, and the linearisation breaks down for large δ. The *qualitative* direction of the prediction is preserved (a continuous schedule should reduce trajectory error), but the *quantitative* O(1/N) scaling is not guaranteed. The CIFAR-10 generalisation experiment (§4.7) probes this regime.

## 3.8 Momentum: Suppressing Discrete-Time Stochastic Fluctuation

The envelope-theorem bound (9) is for *continuous-time gradient flow* on the *exact* loss. Two real-world deviations remain in practice:

- **Discretisation** — finite step size η, contributing O(η) per-step error.
- **Stochasticity** — the gradient at each step is computed on a mini-batch, so the actual update is dθ_t = −η · (∇L_λ(t) + ξ_t) where ξ_t is mean-zero gradient noise. This is exactly the buffer-estimator noise ε of §3.2 (G_est), in its per-step form.

The continuous-limit derivation discards both. In practice, adding the curriculum still leaves a residual per-step oscillation around the optimum path and a non-zero seed-to-seed spread in measured gap depth. This section derives why momentum suppresses precisely this residual.

**The argument.** SGD with momentum coefficient β maintains a running average of the gradient,

```
v_t = β · v_{t-1} + (1 − β) · ∇L_λ(t)(θ_t)
θ_{t+1} = θ_t − η · v_t
```

so the *effective* gradient seen by the parameter update is a low-pass filter on the per-step gradient: high-frequency noise components ξ_t are averaged out, while the slow drift driven by the schedule (the dΘ*/dλ term) passes through nearly unaffected.

Concretely, decompose the per-step gradient at iterate θ_t = Θ*(λ_t) + δ_t into

```
∇L_λ_t(θ_t) ≈ H_λ · δ_t + drift_t + ξ_t                                 (10)
```

where drift_t encodes the schedule-driven motion of Θ*(λ_t) (slow, of order 1/N) and ξ_t is per-step stochastic noise (zero-mean, of order O(1) per step, uncorrelated across steps). Without momentum, the δ_t dynamics are forced by both drift and ξ — and ξ produces oscillation around Θ*(λ_t) of amplitude O(η · ‖ξ‖). With momentum, ξ is averaged across the most recent ≈ 1 / (1 − β) steps, so its variance is suppressed by a factor of (1 − β) — and oscillations of amplitude O(η · ‖ξ‖ · (1 − β)) instead.

**Theoretical statement.** Under (A1)–(A3) plus mean-zero, finite-variance gradient noise, the steady-state amplitude of δ along the curriculum satisfies

```
‖δ‖² ≤ O(τ_eq² / N²)  +  O(η² · ‖ξ‖² · (1 − β))                         (11)
```

The deterministic adiabatic error (the first term) and the noise-driven error (the second term) decompose additively. Increasing N suppresses the first; increasing β suppresses the second. The two refinements address *different* sources of trajectory error.

**Why momentum alone cannot fix the gap.** This is the key falsifiable prediction. Without the curriculum, the schedule is still discontinuous: λ jumps from 0 to 1 at step 1, the unopposed-first-step pathology of (15) fires, and the model is shoved off Θ*(λ) by an O(η) deterministic step (the G_mag contributor) that has nothing to do with stochastic noise. Momentum has no leverage on this — it filters the gradient signal but does not change *where* the gradient points, and the first-step direction is still −∇L_current(θ_0*). The prediction is therefore depth(no curriculum, momentum) ≈ depth(no curriculum, no momentum), and momentum's role is specifically to clean up the residual G_est-driven oscillation that survives the curriculum, not to substitute for the curriculum.

Chapter 4 (§4.6) defines the curriculum × momentum cross that tests both halves of this prediction.

## 3.9 λ_min: Recovering Early-Task Velocity Without Re-introducing Depth

The basic linear curriculum starts at λ(0) = 0. By construction, this means at step 1 the network sees zero current-task signal: the update is purely ∇L_replay. The trajectory stays exactly on Θ*(0) = θ_0* and the model does not begin learning T₁ until λ has ramped some way up. The per-step T₁ accuracy curve reflects this: it stays flat (or even briefly drops, due to noise) for the first ~10–20 % of N before rising. For an N comparable to the task budget, this can cost meaningful T₁ progress.

**The λ_min refinement.** Floor λ at a small positive value:

```
λ(t) = max(λ_min, t / N)        for t ≤ N
λ(t) = 1                         for t > N                              (λ_min schedule)
```

with λ_min > 0.

**What this changes.** The optimum path is unchanged in form, but the *starting point* on the path is now Θ*(λ_min) ≠ θ_0*. The network begins the homotopy at an interior point of the path where ∇L_current is non-zero in the update — so it makes T₁ progress from step 1. The schedule remains continuous: λ(t) is still a continuous (piecewise-linear) function of t, with no jumps. So Part A of §3.4 and the adiabatic tracking of §3.5 both apply, simply over the sub-interval [λ_min, 1] of the homotopy parameter.

**The trade-off.** Setting λ_min > 0 raises the initial replay loss slightly: L_replay(Θ*(λ_min)) > L_replay(θ_0*) by the integral of (4) over [0, λ_min]. So the *floor* of the gap rises (the trajectory cannot go below this value). At the same time, the model's T₁ progress per unit time is faster — by a factor proportional to λ_min over the early phase.

**Theoretical statement.** λ_min controls a one-parameter trade between the gap-floor tightness (smaller λ_min → smaller gap floor) and the early-task velocity (larger λ_min → faster T₁ progress).

**Combined refinement.** Momentum and λ_min act on different parts of the trajectory error — stochastic oscillation (eq. 11, second term) vs. early-time velocity — and are therefore additive. Chapter 4 (§4.6) tests their combined Pareto position against either alone.
