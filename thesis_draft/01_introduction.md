# Chapter 1 — Introduction

## 1.1 Motivation

Continual learning asks a neural network to acquire knowledge from a stream of tasks without forgetting earlier ones. The field's central challenge — catastrophic forgetting — describes a model that, after training on a new task, performs dramatically worse on all previous tasks. Replay-based methods address this by storing a small buffer of past examples and replaying them during new-task training. For this class of methods, the standard benchmark question is: *what is the final accuracy on all tasks after training is complete?*

This thesis addresses a different question: *what happens during training, not just after?* The **stability gap** (De Lange et al., 2023) is the transient accuracy dip that replay-based methods exhibit at the *start* of each new task's training, before recovery is complete. Even a well-tuned Experience Replay (ER) model that finishes task T₁ with high accuracy on task T₀ will pass through a minimum — sometimes a deep one — during the first hundred steps of T₁ training.

This matters for three reasons. First, deployed continual-learning systems are queried throughout training, not only at checkpoints: a model mid-transition makes errors that its final checkpoint never commits. Second, the transience of the gap is itself a scientific puzzle — the model clearly retains the knowledge needed to recover, so what is the mechanism that temporarily disrupts it? Relearning something you knew before is wasted compute. Third, understanding the gap mechanistically is a prerequisite for fixing it efficiently.

## 1.2 Position: The Gap is a Symptom of a Discontinuous Loss

This thesis argues that the stability gap is, in its dominant components, the downstream symptom of a single structural feature of continual learning with abrupt task boundaries: at the moment T₁ begins, the loss the optimiser is following changes discontinuously, from L_replay alone to L_replay + L_current. We refer to this abrupt switch — metaphorically, since the parameter θ does not move — as **landscape teleportation**: from the optimiser's point of view, the surface it has been descending suddenly disappears and is replaced by a different one whose minimum is at θ_joint*, not θ_0*.

The chapter on theory (§3.2) decomposes the gap into three contributors. Two of them — the **magnitude asymmetry** at the first step of the new task and the **trajectory effect** by which SGD leaves the optimum path of the joint loss — are direct downstream consequences of the discontinuity and are eliminated when continuity is restored. The third, **estimator bias** from the finite replay buffer not being a perfect proxy for the true past-task data (Aljundi et al., 2019), is independent of the discontinuity and persists at any schedule. The first two are the focus of the thesis; the third is reported but treated as a baseline-level residual that any replay method shares.

The theoretical reference point for the trajectory contributor is Kao et al. (2021)'s NCL paper, which shows that the gap *survives the strongest possible replay control* — even when SGD optimises the exact joint loss L_T₀ + L_T₁, a transient dip on T₀ still appears, because a finite-step SGD trajectory bends off the steepest-descent path of the joint loss. We rederive their geometric observation as a special case of our discontinuity framing in §3.2.

## 1.3 Intervention: The λ-Curriculum

Given that the gap is caused by a discontinuity, two paradigms address it.

**Path-finding** accepts the discontinuity and navigates the broken landscape carefully — NCL, GEM/A-GEM, and EWC compute or approximate the local Hessian / Fisher information at θ_0* and constrain the SGD update so that motion stays away from directions in which L_replay rises sharply. These methods treat the symptoms of the discontinuity with curvature-aware machinery; they do nothing about the discontinuity itself.

**Landscape-shaping** removes the discontinuity at its source. The **λ-curriculum** is a homotopy that ramps the new-task loss weight from 0 to 1 over the first N steps of each task transition,

```
L_λ(t) = L_replay + λ(t) · L_current,    λ(t) = clip(t / N, 0, 1)
```

so the optimum Θ*(λ) is a continuous curve in parameter space joining θ_0* (at λ = 0) to θ_joint* (at λ = 1). The model "rides the moving valley" along this curve rather than being teleported across an abyss. The envelope theorem (§3.6) guarantees that the replay loss is monotone non-decreasing along Θ*(λ), so in the continuous limit no overshoot is possible. Crucially, the curriculum preserves the natural magnitudes of both gradients — it only modulates the *weight* on the new-task loss, leaving SGD's step-size adaptation intact.

This is the headline intervention of the thesis. Path-finding methods address downstream symptoms with curvature machinery; gradient balancing addresses the magnitude symptom alone but at the cost of SGD's step-size adaptation; the curriculum addresses the root cause (the discontinuity) and leaves the optimiser's natural behaviour alone.

## 1.4 Two Refinements: λ_min and Momentum

The basic linear curriculum has two operational shortcomings, each addressed by a small refinement (derived from first principles in §3.9 and §3.8).

First, a linear ramp introduces a tunable length N. We test an **adaptive λ-schedule** that sets λ(t) from the live ratio of gradient magnitudes ‖g_replay‖ / ‖g_current‖, removing the explicit knob. Adaptive scaling can however leave λ near zero for the earliest steps of the new task and stall plasticity on T₁. We introduce a **λ_min floor**: λ(t) = max(λ_min, schedule(t)), so the current-task gradient is present in the update from step 1 without re-introducing the depth pathology that the curriculum was designed to remove.

Second, even with the curriculum the per-step accuracy curve on T_0 retains a visible oscillation — the discrete-time, stochastic part of SGD that the continuous-limit derivation in §3.6 abstracts away. **Momentum** acts as a low-pass filter on this noise: it averages the per-step gradient over recent steps with an exponential weighting, suppressing high-frequency mini-batch noise while preserving the slow drift along Θ*(λ). The theoretical statement (§3.8) is that momentum tightens the adiabatic-tracking bound toward the deterministic envelope.

## 1.5 Research Questions

**RQ1**: On rot-MNIST, does the linear λ-curriculum applied to standard ER monotonically reduce stability-gap depth as the ramp length N grows, and does it do so without the final-accuracy cost incurred by gradient balancing?

**RQ2**: On rot-MNIST, does the adaptive λ-curriculum applied to standard ER reduce stability-gap depth, and does it do so without the final-accuracy cost incurred by gradient balancing?

**RQ3**: Does adding momentum on top of the λ-curriculum reduce the residual depth fluctuation predicted by the discrete-time stochastic argument? Does momentum on its own — without the curriculum — produce any comparable benefit?

**RQ4**: Does adding a small floor λ_min > 0 on the adaptive curriculum schedule recover the early-task learning rate without re-introducing the depth pathology that the curriculum was designed to remove?

**RQ5 (diagnostic)**: How does the gap depth decompose into magnitude, estimator, and trajectory contributions, and what does each contribution reveal about *why* a curriculum-based intervention is the right shape of fix?

## 1.6 Thesis Outline

Chapter 2 reviews the continual-learning setting, Experience Replay, the established path-finding methods that motivate our comparison, the prior stability-gap literature (De Lange et al., 2023; Hess et al., 2023), Kao et al. (2021)'s trajectory analysis, and the parallel literature on blurry task boundaries. Chapter 3 develops the theory: the discontinuity framing (§3.1), the three-contributor decomposition (§3.2), the path-finding vs. landscape-shaping distinction (§3.3), and the envelope-theorem derivation that bounds the trajectory contribution under the λ-curriculum (§3.4–§3.6), followed by the momentum (§3.8) and λ_min (§3.9) refinements. Chapter 4 describes the experimental testbed, metrics, and the full sweep of conditions. Chapters 5–7 report results, discuss the spatial/temporal duality with NCL, and conclude.
