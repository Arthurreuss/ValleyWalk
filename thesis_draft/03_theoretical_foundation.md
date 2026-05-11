# Chapter 3 — Theoretical Foundation

This chapter develops the mathematical foundation for the λ-curriculum. §3.1 introduces the homotopy family that the curriculum implements and states the assumptions used throughout. §3.2 proves that the replay loss is monotone non-decreasing along the optimum path of the homotopy (the envelope-theorem inequality, Part A). §3.3 proves that the SGD trajectory tracks the optimum path with error O(1/N) in the adiabatic limit (Part B). §3.4 combines the two parts into an O(1/N) bound on the trajectory contribution to the stability gap. §3.5 discusses the regime in which the bound is expected to be tight. §3.6 and §3.7 derive the momentum and λ_min refinements within the same framework.

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

§3.5 discusses how well these assumptions hold on rot-MNIST + MLP, the regime in which the bound is sharply testable.

## 3.2 Part A — The Optimum-Path Inequality

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

## 3.3 Part B — Adiabatic Tracking

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

## 3.4 Combined Bound: The Stability-Gap as O(1/N)

The trajectory contribution to the gap depth is bounded by the maximum of L_replay along the SGD trajectory minus its baseline. Taylor-expanding L_replay around Θ*(λ(t)) and substituting (8):

```
sup_t L_replay(θ(t)) ≤ L_replay(θ_joint*) + O(1/N)                      (9)
```

The first term is the *unavoidable* steady-state increase from joint training: the model trades a small amount of T₀ fit for T₁ capacity at the joint optimum. The O(1/N) term is the *trajectory* component that the curriculum eliminates as N grows.

**Conclusion.** In the continuous-limit, perfect-gradient regime, the trajectory contribution to the stability gap vanishes as N → ∞. The remaining gap is the genuine plasticity–stability trade-off at the joint optimum, not a transient artefact of trajectory geometry.

This is the prediction tested in Chapter 5: under the linear λ-curriculum on rot-MNIST, gap depth should fall monotonically with N at fixed step budget, while accuracy on the joint problem should remain close to that of vanilla ER (since the bound preserves the natural step-size adaptation of SGD).

## 3.5 Regime of Applicability

The bound (9) is derived under (A1)–(A3). These hold to a good approximation on rot-MNIST + small MLP:

- The MLP has ~10⁵ parameters; near a converged optimum the loss is approximately quadratic in a low-dimensional active subspace and H_λ is well-conditioned in that subspace.
- T₀ (0°) and T₁ (90°) MNIST are in the same input space and label set; the joint optimum is connected to θ_0* by a smooth path with no architectural mismatch.
- The displacement ‖θ_joint* − θ_0*‖ is small relative to the curvature scale (the two tasks share most features), so δ stays in the regime where the linearisation (6) is accurate.

Under these conditions the bound (9) is tight enough that even modest N (50–200 steps, on a ≈235-step task budget) should produce a measurable depth reduction. As N approaches the budget, the O(1/N) term shrinks visibly while the unavoidable joint-optimum term remains constant — the predicted monotone-decreasing depth curve.

On harder benchmarks (deeper networks, larger task shifts) the assumptions weaken: H_λ is poorly conditioned in deep ReLU networks, dΘ*/dλ may pass through ill-conditioned regions, and the linearisation breaks down for large δ. The *qualitative* direction of the prediction is preserved (a continuous schedule should reduce trajectory error), but the *quantitative* O(1/N) scaling is not guaranteed. The CIFAR-10 generalisation experiment (§4.7) probes this regime.

## 3.6 Momentum: Suppressing Discrete-Time Stochastic Fluctuation

The envelope-theorem bound (9) is for *continuous-time gradient flow* on the *exact* loss. Two real-world deviations remain in practice:

- **Discretisation** — finite step size η, contributing O(η) per-step error.
- **Stochasticity** — the gradient at each step is computed on a mini-batch, so the actual update is dθ_t = −η · (∇L_λ(t) + ξ_t) where ξ_t is mean-zero gradient noise.

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

**Why momentum alone cannot fix the gap.** This is the key falsifiable prediction. Without the curriculum, the schedule is still discontinuous: λ jumps from 0 to 1 at step 1, the unopposed-first-step pathology fires, and the model is shoved off Θ*(λ) by an O(η) deterministic step that has nothing to do with stochastic noise. Momentum has no leverage on this — it filters the gradient signal but does not change *where* the gradient points, and the first-step direction is still −∇L_current(θ_0*). The prediction is therefore depth(no curriculum, momentum) ≈ depth(no curriculum, no momentum), and momentum's role is specifically to clean up the residual oscillation that survives the curriculum, not to substitute for it.

Chapter 4 (§4.6) defines the curriculum × momentum cross that tests both halves of this prediction.

## 3.7 λ_min: Recovering Early-Task Velocity Without Re-introducing Depth

The basic linear curriculum starts at λ(0) = 0. By construction, this means at step 1 the network sees zero current-task signal: the update is purely ∇L_replay. The trajectory stays exactly on Θ*(0) = θ_0* and the model does not begin learning T₁ until λ has ramped some way up. The per-step T₁ accuracy curve reflects this: it stays flat (or even briefly drops, due to noise) for the first ~10–20 % of N before rising. For an N comparable to the task budget, this can cost meaningful T₁ progress.

**The λ_min refinement.** Floor λ at a small positive value:

```
λ(t) = max(λ_min, t / N)        for t ≤ N
λ(t) = 1                         for t > N                              (λ_min schedule)
```

with λ_min > 0.

**What this changes.** The optimum path is unchanged in form, but the *starting point* on the path is now Θ*(λ_min) ≠ θ_0*. The network begins the homotopy at an interior point of the path where ∇L_current is non-zero in the update — so it makes T₁ progress from step 1. The schedule remains continuous: λ(t) is still a continuous (piecewise-linear) function of t, with no jumps. So Part A of §3.2 and the adiabatic tracking of §3.3 both apply, simply over the sub-interval [λ_min, 1] of the homotopy parameter.

**The trade-off.** Setting λ_min > 0 raises the initial replay loss slightly: L_replay(Θ*(λ_min)) > L_replay(θ_0*) by the integral of (4) over [0, λ_min]. So the *floor* of the gap rises (the trajectory cannot go below this value). At the same time, the model's T₁ progress per unit time is faster — by a factor proportional to λ_min over the early phase.

**Theoretical statement.** λ_min controls a one-parameter trade between the gap-floor tightness (smaller λ_min → smaller gap floor) and the early-task velocity (larger λ_min → faster T₁ progress).

**Combined refinement.** Momentum and λ_min act on different parts of the trajectory error — stochastic oscillation (eq. 11, second term) vs. early-time velocity — and are therefore additive. Chapter 4 (§4.6) tests their combined Pareto position against either alone.
