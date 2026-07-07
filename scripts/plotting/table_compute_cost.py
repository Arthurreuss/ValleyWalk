"""Compute-cost comparison for the two gates (Table~ref{tab:res_cost}).

Reports the wall-clock of the gate-active task (T1) from each run's
results/timing/wall_clock.csv, and the machine-independent slowdown
task1/task0 (task0 is un-gated, so the ratio cancels hardware differences).

    python -m scripts.plotting.table_compute_cost
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.plotting.vw_style as vw


def _timing(method, key, value):
    t0, t1, ratio = [], [], []
    for d in vw.run_dirs(method, key, value):
        f = d / "results" / "timing" / "wall_clock.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        r0 = c[c.task_id == 0].train_seconds
        r1 = c[c.task_id == 1].train_seconds
        if len(r0) and len(r1) and float(r0.iloc[0]) > 0:
            t0.append(float(r0.iloc[0])); t1.append(float(r1.iloc[0]))
            ratio.append(float(r1.iloc[0]) / float(r0.iloc[0]))
    if not t1:
        return None
    return dict(task1=np.mean(t1), ratio=np.mean(ratio), n=len(t1))


ROWS = [
    ("MLP rot-MNIST", [
        ("curriculum (scalar)", "er", "curriculum", "C7_adaptive_lmin0.20"),
        ("asym PER d0.1 (1k)", "precond_er", "per_asym_sweep", "P3_d0.1"),
        ("asym PER d0.1 (full)", "precond_er", "per_asym_sweep", "P3_d0.1_fullbuf"),
    ]),
    ("ResNet-18 BatchNorm", [
        ("vanilla ER", "er", "cifar_headline", "G1_vanilla_BN_buf5k"),
        ("curriculum (scalar)", "er", "cifar_headline", "G3_adaptive_lmin0.20_BN_buf5k"),
        ("asym PER d0.1", "precond_er", "cifar_per_asym", "G7_PER_asym_d0.1_BN_buf5k"),
    ]),
    ("ResNet-18 GroupNorm", [
        ("vanilla ER", "er", "cifar_headline", "G4_vanilla_GN_buf5k"),
        ("curriculum (scalar)", "er", "cifar_headline", "G6_adaptive_lmin0.20_GN_buf5k"),
        ("asym PER d0.1", "precond_er", "cifar_per_asym", "G8_PER_asym_d0.1_GN_buf5k"),
    ]),
]


def main() -> None:
    for block, conds in ROWS:
        print(f"\n== {block} ==")
        print(f"  {'gate':22s} {'task1 (s)':>10s} {'slowdown':>9s}  n")
        for label, m, k, v in conds:
            r = _timing(m, k, v)
            if r is None:
                print(f"  {label:22s} {'--':>10s}")
                continue
            print(f"  {label:22s} {r['task1']:10.0f} {r['ratio']:8.1f}x  {r['n']}")


if __name__ == "__main__":
    main()
