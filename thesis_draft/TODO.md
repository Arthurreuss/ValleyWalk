# Thesis TODO

Last reviewed: 2026-05-20. Items are grouped by what they unblock: **experiments** (new data), **writing** (still-empty or thin sections), **coherence** (cross-reference / accuracy fixes already flagged in the audit), and **polish** (figures, formatting, bibliography).

Items marked **(must)** are blocking a "complete" thesis; **(should)** lifts it from "complete" to "very good"; **(nice)** are stretch goals.

---

## 1. Experiments

### 1.1 (must) CIFAR-10 adaptive curriculum with no λ_min floor (D5)

Currently the CIFAR block has D1 (vanilla ER), D2 (NCL), D3 (adaptive λ_min = 0.20), D4 (adaptive λ_min = 0.10). The pure-adaptive case (λ_min = 0) is missing.

**Why it matters.** The rot-MNIST C4 → C7 trend at μ = 0.9 (depth 0.025 → 0.037 as λ_min grows) suggests λ_min = 0 should give the smallest CIFAR gap_depth too. Right now §5.6 reports D4 as the headline ("D4 is marginally better than D3") but can't claim "smaller λ_min monotonically reduces depth on CIFAR" without the λ_min = 0 anchor.

**Plan.** Add a `D5` block to `scripts/run_curriculum.sh` (analogous to D3/D4 but with `method.lambda_curriculum.lambda_min=0.0`), 5 seeds. Launch:

```bash
CONDITIONS="D5" BLOCK=cifar10 bash scripts/run_curriculum.sh
```

**Downstream writing changes:** add D5 row to Appendix A.3 and Appendix E.R3; extend §5.6 with one paragraph; extend §6.5 if the depth ranking is sharper than expected.

### 1.2 (must) 3+-task rot-MNIST sweep

Currently both testbeds are two-task. Limitation L1 in §6.8 explicitly flags this; future-work item F5 in §6.9 lists it as the most important untested question.

**Why it matters.** Two-task is a clean diagnostic regime but it cannot answer two questions the thesis poses: (i) does the curriculum carry over to multiple consecutive transitions (i.e. does it scale with task count, or does the cumulative trajectory residual grow)?; (ii) does NCL recover its rot-MNIST advantage at 5–10 tasks where Kao et al. (2021)'s SOTA claims live?

**Plan.** Add a new `scripts/run_long_sequence.sh` (or extend `run_curriculum.sh` with a `BLOCK=rot_mnist_5task` block). Suggested conditions:

| Label | Method | Curriculum | Task count |
|-------|--------|------------|------------|
| L1 | vanilla ER | off | 5 |
| L2 | NCL (α = 0.1) | — | 5 |
| L3 | adaptive curriculum, λ_min = 0 | adaptive | 5 |
| L4 | adaptive curriculum, λ_min = 0.10 | adaptive | 5 |

Recommended dataset overrides:

```
dataset=rot_mnist dataset.num_tasks=5 dataset.rotations_deg=[0,30,60,90,120]
```

5 conditions × 2 momentum × 5 seeds = 50 runs. Affordable on the MLP.

**Downstream writing changes:** new §5.7 (renaming current §5.7 to §5.8) "Long-Sequence Behaviour"; update §6.8 (L1 becomes "partially addressed"); rewrite §6.9 F5 to reflect what the new data shows.

### 1.3 (should) 3-task CIFAR-10 sweep — addresses cumulative shift

The default `dom_cifar10` config already lists three corruption types (`[none, gaussian_noise, shot_noise]`). Run one CIFAR D-block at `num_tasks=3` to see whether D3 / D4's edge over vanilla ER widens or narrows on a second transition.

**Plan.** Single override `dataset.num_tasks=3` on the existing D-block; one new condition set. Compute-cost is ≈ 1.5× the current CIFAR block.

### 1.4 (should) Buffer-fidelity diagnostics on C4 / C7

The C-block currently has `GRAD_DIAG=off` because it's expensive. Enable it for two adaptive variants (C4 and C7) to directly verify the §5.3 mechanistic claim ("λ stays low while the iterate is far from local equilibrium"). Single env-var change:

```bash
CONDITIONS="C4 C7" GRAD_DIAG=on bash scripts/run_curriculum.sh
```

5 conditions × 2 momentum × 5 seeds = 20 runs (only the C4 / C7 subset). Adds the per-step λ(t) curve and `cos(g_replay, g_true)` for the two key adaptive conditions.

### 1.5 (nice) Decomposition on CIFAR-10

The G2-G4 path was not replicated on CIFAR (60 k full-data buffer = the CIFAR-10 trainset, would cost ≈ 1 hour per seed). Adding D2-D4-CIFAR-analogue would quantify which contributors carry to ResNet-18 — particularly the BatchNorm-driven contributor §6.5 hypothesises but does not measure.

---

## 2. Writing — still-empty / thin sections

### 2.1 (must) Chapter 7 — Conclusion (currently empty)

Suggested structure (5–7 short paragraphs, ≈ 2 pages):

1. **Restatement of the headline.** The thesis identified the stability gap as a downstream consequence of landscape teleportation (Chapter 3), decomposed it into three contributors (§5.1), and showed that a continuous λ-schedule + momentum reaches the smallest gap among methods that preserve final accuracy on both testbeds (§5.7).
2. **What carried over and what did not.** Carried: the adaptive curriculum on a deeper backbone + corruption-style shift (§5.6). Did not: NCL's rot-MNIST advantage (CIFAR inversion in §5.6/§6.5).
3. **The path-finding / landscape-shaping refinement.** Empirically, the temporal axis (the curriculum's domain) is cheaper to operate on and more robust to architectural changes (§6.5, §6.7) — a finding not previously articulated in the stability-gap literature.
4. **Two-task vs. long-sequence.** Honest scope limit: the headline result is established at two tasks. The 3+-task experiment (Item 1.2) is the next test.
5. **A one-sentence operational recommendation** for a practitioner: "switch on SGD momentum + an adaptive λ-curriculum with λ_min ∈ [0, 0.1]; expect a ≥ 50 % reduction in stability-gap depth at unchanged final accuracy."
6. **Open questions.** Pointer to §6.9.

**No new citations needed.** Pull from chapters 1, 5, 6.

### 2.2 (must) Abstract (currently missing)

A bachelor thesis at most institutions requires a 200–300 word abstract on the title page. Suggested skeleton:

- Sentence 1: the problem (stability gap).
- Sentence 2: the gap in prior work (gap is described as a transient phenomenon but not decomposed into causes).
- Sentences 3–4: contributions (three-contributor decomposition + the λ-curriculum + momentum-cross + CIFAR carry-over).
- Sentence 5: headline result with numbers.
- Sentence 6: implications (landscape-shaping > path-finding when assumptions are honest).

### 2.3 (should) Consolidated bibliography

Each chapter currently carries its own reference list. For final submission, consolidate into `thesis_draft/references.md` (or a `.bib` file if the institution prefers LaTeX) and have each chapter cite by tag rather than by inline reference. Mechanical, but slow if done by hand at the end — easier to do incrementally.

---

## 3. Coherence — issues uncovered in the 2026-05-20 audit

### 3.1 (must) Numerical claims in Chapter 5 corrected on 2026-05-20

Mean values in Chapter 5 tables are correct, but several claims about *seed-paired consistency* and *Wilcoxon p-values* were stated more strongly than the data supports. Fixed in this revision; flagged here for future revisions because the same claim is restated in §6.4 / §6.6:

- **§5.3 (C3 vs. C4, adaptive vs. best linear)**: corrected from "positive on every one of the five seeds" → "positive on 4 of 5 seeds, CI includes zero". Verdict on RQ2 now reads "supported as equivalence, not a strict win".
- **§5.5 (G1 vs. G1_M, momentum-alone falsifier)**: corrected from "rejected, CI excludes zero" → "*not* rejected, CI includes zero". Verdict on RQ3-ii: the strict-null prediction of §3.6 is supported as a non-falsification at this seed count.
- **§5.2 (linear ramp monotonicity)**: corrected from "non-positive across all five seeds" → "3–4 of 5 seeds, CIs span zero".
- **§6.4 and §6.6 in the discussion**: corresponding revisions to the verdict table and the surrounding narrative.

The actual numbers are now computed by `scripts/aggregate_results.py` against the deduped manifests — see Appendix C.7. Re-run the script after any new sweep (Items 1.1–1.5) to keep these in sync.

### 3.2 (done 2026-05-25) Regenerate metrics summaries after all CIFAR runs finish

Discovered 2026-05-22 while inspecting the new `run_cifar.sh` headline block: the diagonal cells `R[i, i]` of every saved `accuracy_matrix.npy` for `i < N − 1` were wrong, and consequently `FORG` in `results/metrics_summary.json` was wrong on the headline + generalization runs.

**Root cause.** `metrics.record_step` was called twice at the same `global_step` for each intermediate task — once by the boundary eval after `notify_task_end`, once by the first eval in the next task's inner loop after one gradient step on the new task. `_acc_at_boundary` picked the later (post-gradient-step, mid-dip) record. Patched in `src/eval/metrics.py` on 2026-05-22 by short-circuiting on the first match at `step == end`.

**Resolution (2026-05-25).** `scripts/patch_cifar_metrics.py` walks every `outputs/**` run whose `run_manifest.json::ablation_key` is `cifar_headline` or `cifar_generalization` (48 runs total), reads the last row of each `task_NN_eval.csv` for `NN < N-1` as the corrected diagonal, rewrites `accuracy_matrix.npy` in place, recomputes `FORG` from the patched matrix, and overwrites `FORG` in both `metrics_summary.json` and `run_manifest.json::final_metrics`. Idempotent. Scope-verified: only `FORG` depended on the bugged diagonal — `ACC`/`WC_ACC` use only `R[N-1, :]` (correct in both code paths), and `min_ACC`/`WF*`/`WP*`/`stability_gap_*` are computed from per-step history sinks that already excluded the duplicate `step==end` record.

**Outcome.** GN headline runs now show `FORG ≈ 0` instead of the spurious −0.23 backward-transfer (D4 seed 1: `−0.247 → −0.007`; D5 seed 1: `−0.201 → +0.028`). BN runs shift by ≤ 0.02 as predicted. Re-run `scripts/aggregate_results.py` before regenerating any downstream Chapter 5 / Appendix E table.

**Caveat for §6 narrative.** The recipe (last in-loop eval row) is a ~10-step approximation of the true boundary value — mean abs error 0.0075, worst 0.024 against the 27 runs where the on-disk metrics.py fix had already produced an exact boundary value. The patch overwrites those exact values with the approximation for consistency across all 48 runs; if exact diagonals are needed, re-run the affected headline conditions with the committed fix.

### 3.3 (done 2026-05-20) Chapter 3 expanded with two new sections

Chapter 3 now contains an explicit §3.2 "Three-Contributor Decomposition" deriving G_mag, G_est, G_traj from the first-step Taylor expansion at θ_0*, and an explicit §3.3 "Path-Finding vs. Landscape-Shaping" formalising the (P) preconditioning / (S) loss-shaping split that earlier chapters had only referenced. Downstream sections §3.4–§3.9 are renumbered from the old §3.2–§3.7. Every §3.X cross-reference in chapters 1, 2, 4, 5, 6 and the appendix was updated to point at the new section numbers. The §3.2.1 reference (which never existed) was removed.

This subsumes the earlier (must) coherence todo on missing decomposition / path-finding sections in Chapter 3 — both predictions and supporting derivations now live where the other chapters claim they do.

---

## 4. Polish

### 4.1 (should) Figures

The repo has `scripts/generate_figures.py` but no figures are inlined in the thesis. The most impactful figures, in priority order:

1. **Per-step accuracy curves** for G1 / G3 / G4 at μ = 0 (all five seeds, mean ± std band). Shows the shape of the gap — what "depth 0.195" actually looks like. Place in §5.1.
2. **Per-step accuracy curves** for C4 vs. C4_M (adaptive ± momentum). Shows the variance-shrinking effect of momentum that the numbers in §5.5 hint at. Place in §5.5.
3. **Gap-depth scatter** for D1 / D2 / D3 / D4 on CIFAR — five dots per condition, paired across seeds. Visually demonstrates the NCL inversion. Place in §5.6.
4. **λ(t) trajectory** for C4 (a single seed, μ = 0 vs. μ = 0.9). Visually demonstrates the §5.3 claim that the adaptive schedule paces itself. Requires Item 1.4 (gradient diagnostics on the C-block).

### 4.2 (nice) Title page, acknowledgements, table of contents

Standard front-matter. Most institutions provide a LaTeX template; mechanical work that's easier to do once everything else is fixed.

### 4.3 (nice) Spell-pass + math-symbol sanity check

After all writing is locked in, do one final pass with a spell-checker. The thesis uses Unicode math symbols (Θ, λ, μ, ε) — make sure none of them got mangled during copy-paste.

### 4.4 (nice) Update `scripts/aggregate_results.py` default seeds

The script's `--seeds` default is `42,123,456,789,1337` (legacy from an earlier seed convention). The actual sweep uses `1,2,3,4,5`. Cosmetic but confusing for a future reader who runs it without the flag.

---

## 5. Quick reference: what the audit found NOT to be a problem

These were checked and look correct:

- Every condition reported in Appendix E has exactly 5 seeds after dedup (no censored seeds in the headline tables).
- D-series cross-references between §4.7 and Appendix A.3 / Appendix E.R3 match.
- The grad-diagnostics on/off policy is consistently documented (G-series on, C-series off, D-series off).
- The NCL `prior_init = 0.1` value is consistent across config / Appendix B.5 / Appendix D / §A.1 / §A.3.
- The 140-runs total reconciles across §A.4 and §R4.

---

## 6. Suggested order of operations

If aiming for a "very good" thesis by some deadline, the lowest-risk order is:

1. **Run Item 1.1 (D5 sweep)** in the background — small block, finishes overnight, unblocks the §5.6 narrative.
2. **Run Item 1.2 (3+-task rot-MNIST)** — biggest single contribution; runs in a couple of hours on the MLP.
3. **Write Chapter 7 (conclusion)** — pure-prose, no new data needed.
4. **Write abstract** — once the conclusion is settled.
5. **Re-run `scripts/aggregate_results.py`** to regenerate Appendix E with the new D5 + long-sequence data.
6. **Update §5.6 / §5.7 / §6.5 / §6.8 / §6.9** to reflect Items 1.1 and 1.2.
7. **Item 4.1 (figures)** — cosmetic but visually load-bearing; do once the data is locked.
8. **Item 2.3 (consolidated bibliography)** — last mechanical step before submission.
