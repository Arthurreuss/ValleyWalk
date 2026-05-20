# Appendix C — Statistical Methodology

This appendix specifies how the per-condition numbers in Appendix E and the comparative claims in Chapters 5–6 are turned into statistical statements. The full sweep produces five paired observations per condition (one per seed); every test below is built around that pairing.

## C.1 Seed Pairing

For every pair of conditions (A, B) compared in Chapter 5, the five per-seed values are aligned by seed index: seed = 1 of condition A is paired with seed = 1 of condition B, and so on. This is the standard paired-design construction (e.g., Bonett & Wright, 2000): per-seed deltas Δ_s = m(A, s) − m(B, s) factor out the seed-induced variance that would otherwise dominate the 5-seed sample. Every condition listed in Appendix A reuses the same `seed=1,2,3,4,5` and the same data-loader ordering, so pairing is exact rather than an approximation.

The unit of analysis is therefore the seed-paired Δ, not the individual run.

## C.2 Significance Test: Paired Wilcoxon Signed-Rank

For all significance claims in Chapter 5 we report the **paired Wilcoxon signed-rank test** (Wilcoxon, 1945) applied to the five seed-paired deltas Δ_s. We prefer it over the paired t-test for two reasons:

1. The primary metrics (`ACC`, `gap_depth`, `min-ACC`, `WC-ACC`) are bounded in [0, 1] and the gap-depth distribution is typically right-skewed near zero. The normality assumption of the t-test is suspect on a five-seed sample under right-skew.
2. Wilcoxon exchanges normality for the weaker assumption that the Δ_s are exchangeable under the null and (for the location form) distributionally symmetric. With paired data on a deterministic seed grid, exchangeability is built into the design.

We use the implementation in `scipy.stats.wilcoxon` with `zero_method="wilcox"` and `alternative="two-sided"` unless explicitly stated otherwise. Reported as

> `paired-Wilcoxon W = w, p = p, n = 5`

with `n` reporting the number of non-zero Δ_s (Wilcoxon discards ties at the null). With only five seeds the smallest attainable two-sided p-value is `2 / 2^5 = 0.0625`, so the test cannot reach the conventional `α = 0.05` threshold even under perfect rank separation. We therefore use Wilcoxon as a *non-falsification* check (consistent direction across all five seeds) rather than as a conventional significance test, and pair it with effect-size and bootstrap-CI reporting below.

## C.3 Effect Size: Mean ± Standard Deviation

Every cell in Appendix E reports mean ± standard deviation across seeds, formatted as `μ ± σ` with three decimal places (two for `gap_area`, integer for `recovery_steps`). The standard deviation uses Bessel's `ddof = 1` (unbiased sample-variance estimator). Cells with only a single valid observation report the bare value. This is the headline summary readers should consult first; the Wilcoxon p-value above adds rank-stability evidence but does not replace inspection of the spread.

## C.4 Equivalence Claims: Bootstrap CI on the Paired Difference

For *equivalence-style* claims (e.g., "the adaptive curriculum matches the best linear curriculum on ACC", or "momentum-alone does not reduce gap depth"), a non-significant Wilcoxon is not by itself evidence of equivalence — absence of evidence is not evidence of absence. We report instead the **bootstrap 95 % confidence interval on the seed-paired mean Δ**, computed by resampling the five paired Δ_s with replacement 10 000 times and taking the 2.5- and 97.5-percentile of the bootstrap means (Efron & Tibshirani, 1993). Reported as

> `Δ̂ = m, 95 % bootstrap CI = [lo, hi], n = 5`

An equivalence claim is considered supported when the entire CI is contained within a pre-declared *region of practical equivalence* — for `gap_depth` this is ±0.01 (1 % accuracy), and for `ACC` it is ±0.005. These tolerances are pre-declared (not data-dependent) so they cannot be tuned to the observed numbers.

## C.5 Censoring of `recovery_steps`

`stability_gap_recovery_steps` is right-censored by construction: when a past task does not recover to 90 % of its pre-switch accuracy within the dense window (`window_steps = 500` on rot-MNIST, `250` on CIFAR), the tracker returns `None`. Treating `None` as missing-at-random would bias the mean toward conditions that recover quickly; treating it as `window_steps` would bias the comparison the other way. We therefore report `recovery_steps` as `μ ± σ (n = k/5 recovered)`, where the mean and standard deviation are computed over the `k` seeds that recovered and the censoring fraction `(5 − k)/5` is reported alongside. When all five seeds fail to recover within the window we report "none recovered" rather than a mean.

For comparative claims that involve `recovery_steps`, we report the metric only when at least four of five paired seeds recover on both conditions, and switch to `gap_area` (which is well-defined for every run) when the censoring fraction is heavier.

## C.6 Multiplicity

Chapter 5 makes multiple comparisons against the same vanilla-ER baseline (G1 / D1) and against the same momentum-off / momentum-on contrast. We do *not* apply a multiplicity correction (Holm, Bonferroni, FDR) at the test level. The reasoning is that with `n = 5` Wilcoxon never reaches an unadjusted `α = 0.05` in the first place (§C.2); the falsification step is therefore "is every paired Δ in the predicted direction?" rather than a thresholded p-value. We list every comparison transparently in the per-RQ subsections of Chapter 5 so that a reader can apply their preferred correction post-hoc to the reported p-values; with five seeds and the consistency criterion of §C.2 the rankings are robust to any reasonable correction.

## C.7 Code

The aggregation and statistical-summary code lives in `scripts/aggregate_results.py` and is run against `outputs/` to produce Appendix E. Per-paragraph numerical claims in Chapters 5–6 are recomputed from the same `outputs/` tree; the manifests are versioned by git SHA and the dedup policy (most recent `started_at` per `(ablation_key, ablation_value, seed)`) is applied in the aggregation step.

## C.8 References for this Appendix

- Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods.* Biometrics Bulletin **1**(6), 80–83.
- Bonett, D. G., & Wright, T. A. (2000). *Sample size requirements for estimating Pearson, Kendall and Spearman correlations.* Psychometrika **65**(1), 23–28.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC.
- De Lange, M., van de Ven, G., & Tuytelaars, T. (2023). *Continual evaluation for lifelong learning: Identifying the stability gap.* ICLR 2023. (Source of the `WF_w` / `WP_w` / `WC-ACC` metric definitions adopted in §4.2.2.)
