# Appendix E — Aggregated Sweep Results

_Auto-generated on 2026-05-20 from `outputs/` (latest commit `eaa8b257`)._

All cells report **mean ± standard deviation across 5 seeds** (seeds 1, 2, 3, 4, 5), formatted per Appendix C. Where a duplicate manifest existed for a given (condition, seed), only the most recent one is retained — earlier aborted runs are excluded. Right-censored `recovery_steps` (where no past task hits 90 % of pre-switch accuracy within the dense window) follow the policy of §C.5.

Conditions correspond directly to Appendix A: §A.1 (G-series), §A.2 (C-series), §A.3 (D-series). The per-condition table E.R4 lists the run-directory path for every seed so individual training curves can be retrieved from `outputs/`.

## R1 — Three-Contributor Decomposition (rot-MNIST, §4.4)

Two-task rot-MNIST (0° → 90°), MLP 2 × 400, SGD lr = 0.1, 1 epoch per task, batch 256. Buffer-fidelity diagnostics (g_true) enabled — every step computes the deterministic empirical past-task gradient on the full 60 000-sample T₀ training set. Cosine and magnitude-ratio columns are empty for NCL (no replay buffer → no g_replay).

### R1.1 Momentum off (µ = 0.0)

| Condition | ACC | FORG | min-ACC | WC-ACC | gap_depth | gap_area | max_drop | recovery_steps | cos(g_r,g_t) mean | cos min | ‖g_r‖/‖g_t‖ mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 — Vanilla ER (1 k reservoir) | 0.873 ± 0.005 | 0.019 ± 0.020 | 0.780 ± 0.020 | 0.832 ± 0.007 | 0.195 ± 0.041 | 10.48 ± 3.45 | 0.195 ± 0.041 | 4 ± 1 | 0.711 ± 0.035 | 0.086 ± 0.103 | 1.04 ± 0.04 |
| G2 — Full-data ER (60 k buffer) | 0.890 ± 0.009 | -0.016 ± 0.009 | 0.797 ± 0.012 | 0.839 ± 0.010 | 0.183 ± 0.025 | 6.03 ± 1.80 | 0.183 ± 0.025 | 3 ± 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| G3 — Balanced ER (1 k reservoir) | 0.849 ± 0.005 | 0.031 ± 0.018 | 0.832 ± 0.003 | 0.840 ± 0.002 | 0.057 ± 0.019 | 7.87 ± 3.41 | 0.057 ± 0.019 | none recovered | 0.665 ± 0.043 | 0.025 ± 0.140 | 1.14 ± 0.06 |
| G4 — Full-data + balanced (60 k buffer) | 0.869 ± 0.002 | -0.014 ± 0.013 | 0.866 ± 0.003 | 0.853 ± 0.003 | 0.019 ± 0.013 | 1.70 ± 1.40 | 0.019 ± 0.013 | none recovered | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| NCL — Standard NCL | 0.770 ± 0.012 | 0.057 ± 0.013 | 0.816 ± 0.020 | 0.766 ± 0.010 | 0.072 ± 0.010 | 12.97 ± 1.97 | 0.071 ± 0.009 | 125 (n=1/5) | — | — | — |

### R1.2 Momentum on (µ = 0.9)

| Condition | ACC | FORG | min-ACC | WC-ACC | gap_depth | gap_area | max_drop | recovery_steps | cos(g_r,g_t) mean | cos min | ‖g_r‖/‖g_t‖ mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 — Vanilla ER (1 k reservoir) | 0.928 ± 0.003 | 0.058 ± 0.007 | 0.817 ± 0.027 | 0.888 ± 0.014 | 0.159 ± 0.026 | 13.61 ± 0.95 | 0.159 ± 0.026 | 13 ± 2 | 0.551 ± 0.033 | 0.135 ± 0.069 | 0.42 ± 0.01 |
| G2 — Full-data ER (60 k buffer) | 0.964 ± 0.002 | -0.010 ± 0.003 | 0.827 ± 0.020 | 0.895 ± 0.010 | 0.149 ± 0.009 | 3.03 ± 0.29 | 0.149 ± 0.009 | 11 ± 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| G3 — Balanced ER (1 k reservoir) | 0.932 ± 0.005 | 0.038 ± 0.003 | 0.901 ± 0.007 | 0.924 ± 0.004 | 0.059 ± 0.006 | 8.18 ± 0.95 | 0.059 ± 0.006 | none recovered | 0.441 ± 0.039 | -0.052 ± 0.117 | 0.26 ± 0.02 |
| G4 — Full-data + balanced (60 k buffer) | 0.967 ± 0.001 | -0.022 ± 0.003 | 0.945 ± 0.005 | 0.951 ± 0.002 | 0.013 ± 0.004 | 0.14 ± 0.06 | 0.013 ± 0.004 | none recovered | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 |
| NCL — Standard NCL | 0.810 ± 0.019 | 0.021 ± 0.004 | 0.877 ± 0.010 | 0.782 ± 0.021 | 0.085 ± 0.011 | 4.89 ± 0.58 | 0.085 ± 0.011 | 5 (n=1/5) | — | — | — |

## R2 — λ-Curriculum Sweep (rot-MNIST, §4.5)

Curriculum applied on top of standard ER (1 k reservoir, no balancing). Same MLP / optimiser / batch-size setup as R1. Buffer-fidelity diagnostics not enabled for the C-block — curriculum claims rest on gap_depth and ACC rather than per-step gradient geometry, and the extra forward+backward per step would dominate runtime.

### R2.1 Momentum off (µ = 0.0)

| Condition | ACC | FORG | min-ACC | WC-ACC | gap_depth | gap_area | max_drop | recovery_steps |
|---|---|---|---|---|---|---|---|---|
| C1 — Linear, N = 50 | 0.851 ± 0.023 | 0.042 ± 0.021 | 0.775 ± 0.040 | 0.818 ± 0.026 | 0.175 ± 0.017 | 9.99 ± 3.32 | 0.175 ± 0.017 | 36 ± 6 |
| C2 — Linear, N = 100 | 0.844 ± 0.025 | 0.046 ± 0.023 | 0.785 ± 0.013 | 0.819 ± 0.015 | 0.162 ± 0.043 | 8.89 ± 3.08 | 0.162 ± 0.043 | 64 ± 9 |
| C3 — Linear, N = 200 | 0.828 ± 0.025 | 0.054 ± 0.024 | 0.775 ± 0.038 | 0.802 ± 0.029 | 0.143 ± 0.046 | 6.79 ± 2.83 | 0.111 ± 0.024 | 133 ± 43 |
| C4 — Adaptive (λ_min = 0) | 0.833 ± 0.025 | 0.050 ± 0.019 | 0.792 ± 0.013 | 0.813 ± 0.016 | 0.128 ± 0.029 | 7.17 ± 2.88 | 0.128 ± 0.029 | 89 ± 6 |
| C5 — Adaptive + λ_min = 0.05 | 0.835 ± 0.025 | 0.048 ± 0.019 | 0.792 ± 0.015 | 0.814 ± 0.017 | 0.132 ± 0.032 | 7.25 ± 2.91 | 0.132 ± 0.032 | 89 ± 6 |
| C6 — Adaptive + λ_min = 0.10 | 0.836 ± 0.026 | 0.048 ± 0.021 | 0.792 ± 0.014 | 0.815 ± 0.018 | 0.139 ± 0.033 | 7.43 ± 2.96 | 0.139 ± 0.033 | 88 ± 11 |
| C7 — Adaptive + λ_min = 0.20 | 0.837 ± 0.031 | 0.049 ± 0.024 | 0.792 ± 0.017 | 0.817 ± 0.024 | 0.139 ± 0.034 | 7.90 ± 3.18 | 0.139 ± 0.034 | 65 ± 8 |

### R2.2 Momentum on (µ = 0.9)

| Condition | ACC | FORG | min-ACC | WC-ACC | gap_depth | gap_area | max_drop | recovery_steps |
|---|---|---|---|---|---|---|---|---|
| C1 — Linear, N = 50 | 0.932 ± 0.003 | 0.053 ± 0.005 | 0.896 ± 0.004 | 0.929 ± 0.003 | 0.064 ± 0.005 | 10.56 ± 1.01 | 0.063 ± 0.005 | none recovered |
| C2 — Linear, N = 100 | 0.932 ± 0.003 | 0.051 ± 0.008 | 0.896 ± 0.004 | 0.928 ± 0.002 | 0.064 ± 0.006 | 9.60 ± 1.02 | 0.064 ± 0.006 | none recovered |
| C3 — Linear, N = 200 | 0.925 ± 0.007 | 0.051 ± 0.006 | 0.900 ± 0.007 | 0.923 ± 0.007 | 0.062 ± 0.006 | 8.04 ± 0.98 | 0.060 ± 0.007 | none recovered |
| C4 — Adaptive (λ_min = 0) | 0.917 ± 0.001 | 0.019 ± 0.004 | 0.932 ± 0.004 | 0.915 ± 0.001 | 0.025 ± 0.005 | 4.22 ± 0.97 | 0.025 ± 0.005 | none recovered |
| C5 — Adaptive + λ_min = 0.05 | 0.922 ± 0.001 | 0.023 ± 0.004 | 0.931 ± 0.004 | 0.922 ± 0.001 | 0.026 ± 0.005 | 4.66 ± 0.94 | 0.026 ± 0.005 | none recovered |
| C6 — Adaptive + λ_min = 0.10 | 0.927 ± 0.002 | 0.028 ± 0.005 | 0.926 ± 0.004 | 0.927 ± 0.002 | 0.030 ± 0.004 | 5.51 ± 0.96 | 0.029 ± 0.005 | none recovered |
| C7 — Adaptive + λ_min = 0.20 | 0.931 ± 0.002 | 0.035 ± 0.005 | 0.919 ± 0.003 | 0.930 ± 0.001 | 0.037 ± 0.004 | 6.81 ± 0.96 | 0.036 ± 0.004 | none recovered |

## R3 — CIFAR-10 Generalisation (§4.7)

Two-task `dom_cifar10` (clean → gaussian_noise, severity 3), ResNet-18 (CIFAR-adapted: 3×3 stem, no max-pool stem), 10 epochs per task, batch 256, SGD lr = 0.1, momentum 0.9 only (no momentum cross). Dense per-step eval coarsened to every 50 steps over a 250-step window.

| Condition | ACC | FORG | min-ACC | WC-ACC | gap_depth | gap_area | max_drop | recovery_steps |
|---|---|---|---|---|---|---|---|---|
| D1 — Vanilla ER | 0.722 ± 0.009 | 0.023 ± 0.014 | 0.631 ± 0.014 | 0.663 ± 0.012 | 0.071 ± 0.008 | 46.48 ± 21.07 | 0.070 ± 0.006 | 450 (n=1/5) |
| D2 — Standard NCL | 0.717 ± 0.009 | 0.057 ± 0.013 | 0.478 ± 0.056 | 0.602 ± 0.027 | 0.180 ± 0.024 | 158.96 ± 21.11 | 0.179 ± 0.025 | 434 ± 136 |
| D3 — Adaptive, λ_min = 0.20 | 0.724 ± 0.009 | 0.011 ± 0.009 | 0.706 ± 0.023 | 0.700 ± 0.014 | 0.031 ± 0.007 | 18.60 ± 14.03 | 0.029 ± 0.008 | none recovered |
| D4 — Adaptive, λ_min = 0.10 | 0.719 ± 0.010 | 0.007 ± 0.010 | 0.706 ± 0.026 | 0.694 ± 0.015 | 0.027 ± 0.009 | 15.16 ± 14.99 | 0.027 ± 0.009 | none recovered |

## R4 — Provenance

Total seed-completed runs after dedup:

- **Decomposition (G-series):** 50 runs
- **Curriculum (C-series):** 70 runs
- **CIFAR (D-series):** 20 runs
- **Total:** 140 runs

### Run paths (one row per seed)

Use these paths to dig into per-step accuracy / gradient curves under `<run_dir>/results/`.

#### Decomposition µ=0

| Condition | Seed | Run dir |
|---|---|---|
| G1 — Vanilla ER (1 k reservoir) | 1 | `er/rot_mnist/2026-05-11_21-37-36_958642` |
| G1 — Vanilla ER (1 k reservoir) | 2 | `er/rot_mnist/2026-05-11_21-37-36_958901` |
| G1 — Vanilla ER (1 k reservoir) | 3 | `er/rot_mnist/2026-05-11_21-37-36_957395` |
| G1 — Vanilla ER (1 k reservoir) | 4 | `er/rot_mnist/2026-05-11_21-37-36_958138` |
| G1 — Vanilla ER (1 k reservoir) | 5 | `er/rot_mnist/2026-05-11_21-37-36_959781` |
| G2 — Full-data ER (60 k buffer) | 1 | `er/rot_mnist/2026-05-11_21-42-36_159975` |
| G2 — Full-data ER (60 k buffer) | 2 | `er/rot_mnist/2026-05-11_21-42-36_160748` |
| G2 — Full-data ER (60 k buffer) | 3 | `er/rot_mnist/2026-05-11_21-42-36_160219` |
| G2 — Full-data ER (60 k buffer) | 4 | `er/rot_mnist/2026-05-11_21-42-36_159557` |
| G2 — Full-data ER (60 k buffer) | 5 | `er/rot_mnist/2026-05-11_21-42-36_159824` |
| G3 — Balanced ER (1 k reservoir) | 1 | `er/rot_mnist/2026-05-11_21-49-32_011573` |
| G3 — Balanced ER (1 k reservoir) | 2 | `er/rot_mnist/2026-05-11_21-49-32_011793` |
| G3 — Balanced ER (1 k reservoir) | 3 | `er/rot_mnist/2026-05-11_21-49-32_011846` |
| G3 — Balanced ER (1 k reservoir) | 4 | `er/rot_mnist/2026-05-11_21-49-32_011802` |
| G3 — Balanced ER (1 k reservoir) | 5 | `er/rot_mnist/2026-05-11_21-49-32_011593` |
| G4 — Full-data + balanced (60 k buffer) | 1 | `er/rot_mnist/2026-05-11_21-54-29_119653` |
| G4 — Full-data + balanced (60 k buffer) | 2 | `er/rot_mnist/2026-05-11_21-54-29_119193` |
| G4 — Full-data + balanced (60 k buffer) | 3 | `er/rot_mnist/2026-05-11_21-54-29_119012` |
| G4 — Full-data + balanced (60 k buffer) | 4 | `er/rot_mnist/2026-05-11_21-54-29_120638` |
| G4 — Full-data + balanced (60 k buffer) | 5 | `er/rot_mnist/2026-05-11_21-54-29_118902` |
| NCL — Standard NCL | 1 | `ncl/rot_mnist/2026-05-11_21-59-54_456254` |
| NCL — Standard NCL | 2 | `ncl/rot_mnist/2026-05-11_21-59-54_455614` |
| NCL — Standard NCL | 3 | `ncl/rot_mnist/2026-05-11_21-59-54_454561` |
| NCL — Standard NCL | 4 | `ncl/rot_mnist/2026-05-11_21-59-54_455823` |
| NCL — Standard NCL | 5 | `ncl/rot_mnist/2026-05-11_21-59-54_455291` |

#### Decomposition µ=0.9

| Condition | Seed | Run dir |
|---|---|---|
| G1 — Vanilla ER (1 k reservoir) | 1 | `er/rot_mnist/2026-05-11_22-04-36_858334` |
| G1 — Vanilla ER (1 k reservoir) | 2 | `er/rot_mnist/2026-05-11_22-04-36_859230` |
| G1 — Vanilla ER (1 k reservoir) | 3 | `er/rot_mnist/2026-05-11_22-04-36_860780` |
| G1 — Vanilla ER (1 k reservoir) | 4 | `er/rot_mnist/2026-05-11_22-04-36_861722` |
| G1 — Vanilla ER (1 k reservoir) | 5 | `er/rot_mnist/2026-05-11_22-04-36_861118` |
| G2 — Full-data ER (60 k buffer) | 1 | `er/rot_mnist/2026-05-11_22-09-46_023949` |
| G2 — Full-data ER (60 k buffer) | 2 | `er/rot_mnist/2026-05-11_22-09-46_023564` |
| G2 — Full-data ER (60 k buffer) | 3 | `er/rot_mnist/2026-05-11_22-09-46_023290` |
| G2 — Full-data ER (60 k buffer) | 4 | `er/rot_mnist/2026-05-11_22-09-46_022926` |
| G2 — Full-data ER (60 k buffer) | 5 | `er/rot_mnist/2026-05-11_22-09-46_022808` |
| G3 — Balanced ER (1 k reservoir) | 1 | `er/rot_mnist/2026-05-11_22-15-20_519946` |
| G3 — Balanced ER (1 k reservoir) | 2 | `er/rot_mnist/2026-05-11_22-15-20_519628` |
| G3 — Balanced ER (1 k reservoir) | 3 | `er/rot_mnist/2026-05-11_22-15-20_519772` |
| G3 — Balanced ER (1 k reservoir) | 4 | `er/rot_mnist/2026-05-11_22-15-20_519769` |
| G3 — Balanced ER (1 k reservoir) | 5 | `er/rot_mnist/2026-05-11_22-15-20_519517` |
| G4 — Full-data + balanced (60 k buffer) | 1 | `er/rot_mnist/2026-05-11_22-20-12_492720` |
| G4 — Full-data + balanced (60 k buffer) | 2 | `er/rot_mnist/2026-05-11_22-20-12_487634` |
| G4 — Full-data + balanced (60 k buffer) | 3 | `er/rot_mnist/2026-05-11_22-20-12_489681` |
| G4 — Full-data + balanced (60 k buffer) | 4 | `er/rot_mnist/2026-05-11_22-20-12_491180` |
| G4 — Full-data + balanced (60 k buffer) | 5 | `er/rot_mnist/2026-05-11_22-20-12_490811` |
| NCL — Standard NCL | 1 | `ncl/rot_mnist/2026-05-11_22-25-41_057866` |
| NCL — Standard NCL | 2 | `ncl/rot_mnist/2026-05-11_22-25-41_058256` |
| NCL — Standard NCL | 3 | `ncl/rot_mnist/2026-05-11_22-25-41_057667` |
| NCL — Standard NCL | 4 | `ncl/rot_mnist/2026-05-11_22-25-41_058404` |
| NCL — Standard NCL | 5 | `ncl/rot_mnist/2026-05-11_22-25-41_061486` |

#### Curriculum µ=0

| Condition | Seed | Run dir |
|---|---|---|
| C1 — Linear, N = 50 | 1 | `er/rot_mnist/2026-05-12_00-04-15_401447` |
| C1 — Linear, N = 50 | 2 | `er/rot_mnist/2026-05-12_00-04-15_403020` |
| C1 — Linear, N = 50 | 3 | `er/rot_mnist/2026-05-12_00-04-15_403103` |
| C1 — Linear, N = 50 | 4 | `er/rot_mnist/2026-05-12_00-04-15_401456` |
| C1 — Linear, N = 50 | 5 | `er/rot_mnist/2026-05-12_00-04-15_400737` |
| C2 — Linear, N = 100 | 1 | `er/rot_mnist/2026-05-12_00-08-51_807087` |
| C2 — Linear, N = 100 | 2 | `er/rot_mnist/2026-05-12_00-08-51_808229` |
| C2 — Linear, N = 100 | 3 | `er/rot_mnist/2026-05-12_00-08-51_807067` |
| C2 — Linear, N = 100 | 4 | `er/rot_mnist/2026-05-12_00-08-51_807750` |
| C2 — Linear, N = 100 | 5 | `er/rot_mnist/2026-05-12_00-08-51_807136` |
| C3 — Linear, N = 200 | 1 | `er/rot_mnist/2026-05-12_00-13-25_081709` |
| C3 — Linear, N = 200 | 2 | `er/rot_mnist/2026-05-12_00-13-25_080910` |
| C3 — Linear, N = 200 | 3 | `er/rot_mnist/2026-05-12_00-13-25_082188` |
| C3 — Linear, N = 200 | 4 | `er/rot_mnist/2026-05-12_00-13-25_080874` |
| C3 — Linear, N = 200 | 5 | `er/rot_mnist/2026-05-12_00-13-25_082066` |
| C4 — Adaptive (λ_min = 0) | 1 | `er/rot_mnist/2026-05-12_00-57-07_925086` |
| C4 — Adaptive (λ_min = 0) | 2 | `er/rot_mnist/2026-05-12_00-57-07_925481` |
| C4 — Adaptive (λ_min = 0) | 3 | `er/rot_mnist/2026-05-12_00-57-07_925311` |
| C4 — Adaptive (λ_min = 0) | 4 | `er/rot_mnist/2026-05-12_00-57-07_925460` |
| C4 — Adaptive (λ_min = 0) | 5 | `er/rot_mnist/2026-05-12_00-57-07_925660` |
| C5 — Adaptive + λ_min = 0.05 | 1 | `er/rot_mnist/2026-05-12_01-01-53_472505` |
| C5 — Adaptive + λ_min = 0.05 | 2 | `er/rot_mnist/2026-05-12_01-01-53_471857` |
| C5 — Adaptive + λ_min = 0.05 | 3 | `er/rot_mnist/2026-05-12_01-01-53_470939` |
| C5 — Adaptive + λ_min = 0.05 | 4 | `er/rot_mnist/2026-05-12_01-01-53_461970` |
| C5 — Adaptive + λ_min = 0.05 | 5 | `er/rot_mnist/2026-05-12_01-01-53_471423` |
| C6 — Adaptive + λ_min = 0.10 | 1 | `er/rot_mnist/2026-05-12_01-09-58_588696` |
| C6 — Adaptive + λ_min = 0.10 | 2 | `er/rot_mnist/2026-05-12_01-09-58_588767` |
| C6 — Adaptive + λ_min = 0.10 | 3 | `er/rot_mnist/2026-05-12_01-09-58_589848` |
| C6 — Adaptive + λ_min = 0.10 | 4 | `er/rot_mnist/2026-05-12_01-09-58_589225` |
| C6 — Adaptive + λ_min = 0.10 | 5 | `er/rot_mnist/2026-05-12_01-09-58_588747` |
| C7 — Adaptive + λ_min = 0.20 | 1 | `er/rot_mnist/2026-05-12_01-18-33_382730` |
| C7 — Adaptive + λ_min = 0.20 | 2 | `er/rot_mnist/2026-05-12_01-18-33_384105` |
| C7 — Adaptive + λ_min = 0.20 | 3 | `er/rot_mnist/2026-05-12_01-18-33_384576` |
| C7 — Adaptive + λ_min = 0.20 | 4 | `er/rot_mnist/2026-05-12_01-18-33_382446` |
| C7 — Adaptive + λ_min = 0.20 | 5 | `er/rot_mnist/2026-05-12_01-18-33_382676` |

#### Curriculum µ=0.9

| Condition | Seed | Run dir |
|---|---|---|
| C1 — Linear, N = 50 | 1 | `er/rot_mnist/2026-05-12_01-28-00_801405` |
| C1 — Linear, N = 50 | 2 | `er/rot_mnist/2026-05-12_01-28-00_802016` |
| C1 — Linear, N = 50 | 3 | `er/rot_mnist/2026-05-12_01-28-00_802338` |
| C1 — Linear, N = 50 | 4 | `er/rot_mnist/2026-05-12_01-28-00_801800` |
| C1 — Linear, N = 50 | 5 | `er/rot_mnist/2026-05-12_01-28-00_802367` |
| C2 — Linear, N = 100 | 1 | `er/rot_mnist/2026-05-12_01-38-07_922189` |
| C2 — Linear, N = 100 | 2 | `er/rot_mnist/2026-05-12_01-38-07_920998` |
| C2 — Linear, N = 100 | 3 | `er/rot_mnist/2026-05-12_01-38-07_921501` |
| C2 — Linear, N = 100 | 4 | `er/rot_mnist/2026-05-12_01-38-07_924514` |
| C2 — Linear, N = 100 | 5 | `er/rot_mnist/2026-05-12_01-38-07_923681` |
| C3 — Linear, N = 200 | 1 | `er/rot_mnist/2026-05-12_01-48-26_433206` |
| C3 — Linear, N = 200 | 2 | `er/rot_mnist/2026-05-12_01-48-26_433988` |
| C3 — Linear, N = 200 | 3 | `er/rot_mnist/2026-05-12_01-48-26_433947` |
| C3 — Linear, N = 200 | 4 | `er/rot_mnist/2026-05-12_01-48-26_434176` |
| C3 — Linear, N = 200 | 5 | `er/rot_mnist/2026-05-12_01-48-26_433184` |
| C4 — Adaptive (λ_min = 0) | 1 | `er/rot_mnist/2026-05-12_02-26-49_450435` |
| C4 — Adaptive (λ_min = 0) | 2 | `er/rot_mnist/2026-05-12_02-26-49_450437` |
| C4 — Adaptive (λ_min = 0) | 3 | `er/rot_mnist/2026-05-12_02-26-49_450230` |
| C4 — Adaptive (λ_min = 0) | 4 | `er/rot_mnist/2026-05-12_02-26-49_450309` |
| C4 — Adaptive (λ_min = 0) | 5 | `er/rot_mnist/2026-05-12_02-26-49_449603` |
| C5 — Adaptive + λ_min = 0.05 | 1 | `er/rot_mnist/2026-05-12_04-03-06_036864` |
| C5 — Adaptive + λ_min = 0.05 | 2 | `er/rot_mnist/2026-05-12_04-03-06_036650` |
| C5 — Adaptive + λ_min = 0.05 | 3 | `er/rot_mnist/2026-05-12_04-03-06_036687` |
| C5 — Adaptive + λ_min = 0.05 | 4 | `er/rot_mnist/2026-05-12_04-03-06_036511` |
| C5 — Adaptive + λ_min = 0.05 | 5 | `er/rot_mnist/2026-05-12_04-03-06_036528` |
| C6 — Adaptive + λ_min = 0.10 | 1 | `er/rot_mnist/2026-05-12_04-10-22_172687` |
| C6 — Adaptive + λ_min = 0.10 | 2 | `er/rot_mnist/2026-05-12_04-10-22_172610` |
| C6 — Adaptive + λ_min = 0.10 | 3 | `er/rot_mnist/2026-05-12_04-10-22_172613` |
| C6 — Adaptive + λ_min = 0.10 | 4 | `er/rot_mnist/2026-05-12_04-10-22_172889` |
| C6 — Adaptive + λ_min = 0.10 | 5 | `er/rot_mnist/2026-05-12_04-10-22_172525` |
| C7 — Adaptive + λ_min = 0.20 | 1 | `er/rot_mnist/2026-05-12_04-15-10_473685` |
| C7 — Adaptive + λ_min = 0.20 | 2 | `er/rot_mnist/2026-05-12_04-15-10_473576` |
| C7 — Adaptive + λ_min = 0.20 | 3 | `er/rot_mnist/2026-05-12_04-15-10_473596` |
| C7 — Adaptive + λ_min = 0.20 | 4 | `er/rot_mnist/2026-05-12_04-15-10_473572` |
| C7 — Adaptive + λ_min = 0.20 | 5 | `er/rot_mnist/2026-05-12_04-15-10_473756` |

#### CIFAR-10 (µ=0.9)

| Condition | Seed | Run dir |
|---|---|---|
| D1 — Vanilla ER | 1 | `er/dom_cifar10/2026-05-19_00-44-18_201099` |
| D1 — Vanilla ER | 2 | `er/dom_cifar10/2026-05-19_03-10-55_527672` |
| D1 — Vanilla ER | 3 | `er/dom_cifar10/2026-05-19_10-54-31_742454` |
| D1 — Vanilla ER | 4 | `er/dom_cifar10/2026-05-19_11-36-07_700197` |
| D1 — Vanilla ER | 5 | `er/dom_cifar10/2026-05-19_12-18-13_766224` |
| D2 — Standard NCL | 1 | `ncl/dom_cifar10/2026-05-19_13-00-42_393348` |
| D2 — Standard NCL | 2 | `ncl/dom_cifar10/2026-05-19_13-39-40_904068` |
| D2 — Standard NCL | 3 | `ncl/dom_cifar10/2026-05-19_14-18-16_626258` |
| D2 — Standard NCL | 4 | `ncl/dom_cifar10/2026-05-19_14-56-49_529654` |
| D2 — Standard NCL | 5 | `ncl/dom_cifar10/2026-05-19_15-35-22_955623` |
| D3 — Adaptive, λ_min = 0.20 | 1 | `er/dom_cifar10/2026-05-19_16-14-01_729452` |
| D3 — Adaptive, λ_min = 0.20 | 2 | `er/dom_cifar10/2026-05-19_16-55-48_088532` |
| D3 — Adaptive, λ_min = 0.20 | 3 | `er/dom_cifar10/2026-05-19_17-38-11_159299` |
| D3 — Adaptive, λ_min = 0.20 | 4 | `er/dom_cifar10/2026-05-19_18-19-59_621589` |
| D3 — Adaptive, λ_min = 0.20 | 5 | `er/dom_cifar10/2026-05-19_19-01-20_140259` |
| D4 — Adaptive, λ_min = 0.10 | 1 | `er/dom_cifar10/2026-05-19_22-48-17_473358` |
| D4 — Adaptive, λ_min = 0.10 | 2 | `er/dom_cifar10/2026-05-19_23-30-35_670325` |
| D4 — Adaptive, λ_min = 0.10 | 3 | `er/dom_cifar10/2026-05-20_00-12-20_587159` |
| D4 — Adaptive, λ_min = 0.10 | 4 | `er/dom_cifar10/2026-05-20_00-54-14_530772` |
| D4 — Adaptive, λ_min = 0.10 | 5 | `er/dom_cifar10/2026-05-20_01-36-11_963817` |

