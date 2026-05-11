# Chapter 2 — Background

## 2.1 Continual Learning

Continual (or lifelong) learning describes the setting where a model is trained on a non-stationary stream of tasks T₀, T₁, T₂, …, without joint access to all tasks simultaneously. The fundamental difficulty is the **plasticity–stability dilemma**: a model must remain plastic enough to learn new tasks while remaining stable enough to retain old ones. Neural networks trained with standard SGD exhibit catastrophic forgetting (McCloskey & Cohen, 1989; Ratcliff, 1990) — gradient updates for a new task overwrite the weights that stored past knowledge, because the objectives are not simultaneously minimised.

Three families of approaches address forgetting in deep neural networks:

- **Regularisation-based** methods (e.g., EWC, Kirkpatrick et al., 2017) add a penalty term that slows updates to parameters important for past tasks, as measured by a diagonal Hessian approximation (Fisher information).
- **Architecture-based** methods (e.g., PackNet, Progressive Neural Networks) allocate separate capacity to each task, preventing parameter interference at the cost of growing model size.
- **Replay-based** methods maintain a memory buffer of past-task examples and replay them during new-task training. This is the family studied in this thesis.

## 2.2 Experience Replay

Experience Replay (Lopez-Paz & Ranzato, 2017; Ratcliff, 1990) is the simplest replay baseline. At each training step the model performs two separate forward+backward passes — one on the current-task mini-batch and one on a replay mini-batch sampled from the buffer — and sums the two gradients before the optimiser step:

```
θ ← θ − η · (∇_θ L(x_current, y_current; θ) + ∇_θ L(x_replay, y_replay; θ))
```

This is the "standard" mode used throughout this thesis; a "balanced" variant that normalises each gradient before summing is introduced in §3.3 and used as one of the decomposition conditions in §4.4.

The replay buffer is populated by **reservoir sampling** (Vitter's Algorithm R, 1985): each sample seen so far has equal probability `budget / n_seen` of currently occupying a buffer slot, regardless of arrival order. In our implementation the buffer is populated *after* each task ends by iterating once over the completed task's training data and calling the per-sample reservoir insertion; samples may be accepted or rejected stochastically depending on how many samples have already been seen. During T₀ training the buffer is empty by construction, so ER degrades to plain SGD for the first task — the stability gap is measurable only from T₁ onward.

ER's simplicity makes it a strong, well-understood baseline. Its primary limitation for stability-gap analysis is precisely that it does *nothing* to balance gradient magnitudes or directions — making it the ideal condition in which each contributor's individual impact can be measured.

**Buffer-as-estimator caveat.** A finite replay buffer is an *imperfect estimator* of the true past-task data distribution. Aljundi et al. (2019) formalise this by framing buffer selection as a constraint-reduction problem in which the chosen samples approximate the feasibility region defined by the *full* past-task data; reservoir sampling is a specific (and not optimal) policy within that frame. The gradient ∇L_replay(θ) computed on a 1 k-sample buffer differs from the gradient ∇L_true(θ) computed on all past data, both in direction and magnitude. This estimator gap is present at every step and does not disappear under any schedule. It is reported in this thesis as a separate, persistent contributor to the stability gap (§3.2) and is *not* the contributor the λ-curriculum is designed to fix.

## 2.3 Path-Finding Methods: NCL, GEM, A-GEM, EWC

A second class of replay-based and replay-adjacent methods attempts to navigate the discontinuous landscape at the task boundary by constraining the SGD update with second-order information about the past task(s).

**GEM** (Lopez-Paz & Ranzato, 2017) projects the new-task gradient to the closest direction that does not increase the loss on any past task in the memory buffer:

```
min_g  ‖g − g_new‖²    s.t.   g · g_k ≥ 0   for all k in buffer
```

This is a quadratic program that ensures the update does not increase replay loss on any single past task. **A-GEM** (Chaudhry et al., 2018) approximates the constraint with a single average past gradient, making it computationally cheap. Both address directional bias by constraining the update direction at the current step but do not address the trajectory effect — the constraint is only enforced step-wise, not across the trajectory.

**EWC** (Kirkpatrick et al., 2017) adds a Fisher-weighted L2 penalty around θ_0*, slowing updates in directions that the old task constrained. **NCL** (Kao et al., 2021) preconditions the SGD update by the precision matrix of the past-task loss, producing a natural-gradient-like step that respects the curvature of L_replay.

In the framing of this thesis (Chapter 3), all of these are *path-finding* methods: they accept the discontinuity of the loss at the task boundary and invest second-order machinery into picking a safe trajectory across the teleported terrain.

## 2.4 The Stability Gap

De Lange et al. (2023) formally characterise the stability gap as the transient accuracy dip on previously seen tasks at the start of new-task training, in replay-based methods. Their key empirical observations:

1. The gap is largest in the first few hundred steps of each new task.
2. Methods with larger replay buffers show smaller gaps (consistent with the estimator caveat in §2.2 — a larger buffer is a better proxy for the full past data).
3. The gap partially recovers as training continues, but the minimum accuracy during the gap is systematically below the pre-task baseline.
4. The gap exists even with reservoir sampling and even at moderate learning rates.

Hess et al. (2023) provide a complementary mechanistic analysis. Their "second perspective on continual learning" introduces the concept of the *trajectory bend*: at a task optimum, the gradient of the old-task loss is approximately zero, so the first step of new-task training is one-sided. They propose looking at the geometry of the joint-loss landscape at the transition point as a predictor of gap severity.

## 2.5 Kao et al. (2021): The Gap Survives the Joint-Loss Oracle

A theoretical pillar of this thesis comes from Kao et al. (2021), "Natural Continual Learning: Success is a Journey, Not (Just) a Destination" (NeurIPS 2021), the NCL paper. As background motivation for their NCL method, they isolate a remarkably clean fact about gradient descent through a task transition.

**The setting.** Suppose we remove every replay-related confound: no buffer, no sampling noise, no magnitude imbalance from a finite mini-batch. Instead, the optimiser has perfect access to the *true joint loss* L_joint(θ) = L_T₀(θ) + L_T₁(θ), and runs SGD on it from θ_0* (a stationary point of L_T₀). This is the strongest possible replay regime — equivalent to multi-task training with perfect knowledge of past data.

**The observation.** Even in this regime, accuracy on T₀ dips before recovering. The dip is small but non-zero, and it is *not* attributable to bad gradient estimates: the gradient is exact. It is attributable to the *trajectory* the optimiser takes through parameter space.

**Why this happens, from first principles.** At θ_0*, ∇L_T₀(θ_0*) ≈ 0 by stationarity; ∇L_T₁(θ_0*) is generically non-zero. So at step 1, ∇L_joint(θ_0*) ≈ ∇L_T₁(θ_0*), and SGD moves in that direction by a finite step size η. After this step, θ₁ = θ_0* − η · ∇L_T₁(θ_0*). Because the iterate has left θ_0*, the second-order Taylor expansion gives

```
L_T₀(θ₁) − L_T₀(θ_0*) ≈ ½ · η² · ∇L_T₁(θ_0*)ᵀ · H_T₀(θ_0*) · ∇L_T₁(θ_0*) > 0
```

whenever H_T₀ has any positive curvature in the direction of ∇L_T₁ — which it generically does. The Hessian of L_T₀ at its own minimum is positive semi-definite; any displacement raises L_T₀ quadratically. The replay-restoring force ∇L_T₀(θ₁) only switches on *after* this rise, and so the joint trajectory is a curved arc that overshoots and recovers.

In our framing (§3.1), this is the cleanest possible isolation of the **trajectory contributor**: the discontinuity has been suppressed (the loss is the joint loss from step 1) and the buffer has been eliminated, yet the geometric effect of an unopposed first step survives. Chapter 3 shows that smoothing the schedule — replacing the instant switch with a continuous ramp — is sufficient to eliminate this contributor in the continuous-time limit.

## 2.6 Blurry Task Boundaries and Their Relationship to the λ-Curriculum

A parallel line of continual-learning research relaxes the assumption of *abrupt* task boundaries. In **blurry-boundary** or **boundary-free** settings (Aljundi et al., 2019, "Task-Free Continual Learning"; Bang et al., 2021, "Rainbow Memory"; Koh et al., 2022, "Online Boundary-Free Continual Learning by Scheduled Data Prior"), the data stream does not switch from T₀ to T₁ at a single time-step. Instead, T₁ samples are introduced gradually: at time t, the mini-batch is drawn from a mixture (1−α(t)) · P_{T₀} + α(t) · P_{T₁} with α(t) increasing from 0 to 1 over a transition window.

**The gradient connection (first principles).** Take an SGD update on a mini-batch that is α-fraction T₁, (1−α)-fraction T₀:

```
E[∇L_minibatch(θ)] = α · ∇L_{T₁}(θ) + (1 − α) · ∇L_{T₀}(θ)
```

This is exactly a weighted sum of the two task gradients. If we substitute λ = α / (1 − α), this is also the gradient of the loss L_{T₀} + λ · L_{T₁} (up to an overall scale 1 − α). Up to that scale factor, the mini-batch from a blurry boundary produces the *same* update direction as the λ-curriculum on the joint loss. Both interventions implement the same first-principles fix to the unopposed-first-step problem: they ensure that ∇L_replay (or its data-side equivalent ∇L_{T₀}) is present at full strength from step 1 of the transition, while ∇L_current (or ∇L_{T₁}) is introduced gradually.

**Why this matters for our decomposition.** Blurry boundaries blur three things at once: the data distribution, the loss composition, and (implicitly) the magnitude balance. The λ-curriculum surgically isolates one of these — the loss composition — while keeping the data-side and magnitude-side controls separate (the latter via the balanced ER normalisation). Reading our intervention through the blurry-boundary lens makes its scope precise: we are not proposing a new continual-learning paradigm; we are showing that the *single* knob shared between blurry boundaries and replay-with-curriculum (gradual ramp of the new-task gradient weight) is the operative element for stability-gap depth.

**Differences worth noting.** Blurry boundaries do not require a replay buffer: the gradient mixing is achieved through the data stream. In the limit α(t) = 1, the T₀ signal disappears entirely from new mini-batches, and the only retention pressure is the network's inertia. The λ-curriculum, by contrast, is built on top of replay and so retains an explicit ∇L_replay signal at λ = 1 (the standard ER regime). The two interventions therefore answer different operational questions: blurry boundaries ask *can we soften the transition without storing data?*; the λ-curriculum asks *given that we are storing data, how should we weight the loss components during the transition?*

In our framing, the curriculum is the cleaner controlled experiment for the trajectory effect: it modifies one weight in the loss while everything else (data sampling, replay magnitude, sample distribution) is held fixed by construction. The blurry-boundary literature is consistent with our findings — gradual transitions reduce forgetting — but does not isolate which ingredient of the gradual transition is doing the work. The λ-curriculum, combined with the falsification experiments in §3 and §5, identifies that ingredient as the *asymmetric ramp of the new-task component*, not the slowing or the delay.
