"""RQ4a figure: the stability-plasticity frontier the two gates trace.

Final accuracy (x, right = better) against gap depth (y, inverted so up = lower
depth = better), for every rot-MNIST gate condition, at momentum 0 (left) and
0.9 (right).  The lambda-path is the curriculum (scalar feedback gate, blue);
the delta-path is asymmetric PER (directional feedforward gate, aqua) on the 1k
and full buffers.  Anchors (grey): the no-gate baseline D1, the driver-removal
floor D4, and the diagnostic full-buffer floor C8.

At momentum 0 the delta-path dominates (up and to the right of the lambda-path).
Under momentum neither dominates: the lambda-path reaches the lowest depth, the
delta-path the highest accuracy -- the frontier, not a winner, is the result.
"""

from __future__ import annotations

import scripts.plotting.vw_style as vw
import matplotlib.pyplot as plt


def pt(method, key, value):
    """Return (ACC, depth_pp) for a condition, or None if it has no data."""
    a = vw.agg(method, key, value)
    if a["n"] == 0 or a["ACC_mean"] != a["ACC_mean"]:
        return None
    return a["ACC_mean"], a["depth_pp"]


def path(ax, method, key, values, color, marker, label, *, fill=True,
         connect="-", z=4):
    xs, ys = [], []
    for v in values:
        p = pt(method, key, v)
        if p is not None:
            xs.append(p[0]); ys.append(p[1])
    if not xs:
        return
    if connect:
        ax.plot(xs, ys, color=color, lw=1.3, linestyle=connect, alpha=0.55, zorder=z)
    ax.scatter(xs, ys, s=52, marker=marker, zorder=z + 1, label=label,
               facecolors=color if fill else "white",
               edgecolors=color, linewidths=1.6)


def anchor(ax, method, key, value, label, dx, dy, ha="left"):
    p = pt(method, key, value)
    if p is None:
        return
    ax.scatter(*p, s=120, marker="*", color=vw.MUTED, edgecolors=vw.INK_SOFT,
               linewidths=0.7, zorder=6)
    ax.annotate(label, xy=p, xytext=(p[0] + dx, p[1] + dy), ha=ha,
                fontsize=8.5, color=vw.INK_SOFT, zorder=7)


def panel(ax, suf, title):
    # lambda-path (curriculum, blue)
    path(ax, "er", "curriculum",
         [f"C{i}_adaptive{s}{suf}" for i, s in
          [(4, ""), (5, "_lmin0.05"), (6, "_lmin0.10"), (7, "_lmin0.20")]],
         vw.C_CURR, "o", "curriculum (adaptive)")
    path(ax, "er", "curriculum",
         [f"C1_linear_N50{suf}", f"C2_linear_N100{suf}", f"C3_linear_N200{suf}"],
         vw.C_CURR, "o", "curriculum (linear ramp)", fill=False, connect=":")
    # delta-path (asym PER, aqua)
    path(ax, "precond_er", "per_asym_sweep",
         [f"P1_d1.0{suf}", f"P2_d0.3{suf}", f"P3_d0.1{suf}", f"P4_d0.03{suf}"],
         vw.C_PER, "s", "asym PER (1k buffer)")
    path(ax, "precond_er", "per_asym_sweep",
         [f"P1_d1.0_fullbuf{suf}", f"P2_d0.3_fullbuf{suf}",
          f"P3_d0.1_fullbuf{suf}", f"P4_d0.03_fullbuf{suf}"],
         vw.C_PER, "D", "asym PER (full buffer)", fill=False, connect="--")
    # anchors
    anchor(ax, "er", "decomposition", f"D1_vanilla{suf}", "D1 vanilla", 0.002, 0.6)
    anchor(ax, "er", "decomposition", f"D4_fulldata_balanced{suf}",
           "D4 driver floor", 0.002, 0.8)
    anchor(ax, "er", "curriculum", f"C8_adaptive_lmin0.20_fullbuf{suf}",
           "C8 full-buffer floor", -0.002, 0.8, ha="right")

    ax.invert_yaxis()
    ax.set_xlabel("final accuracy (ACC)")
    ax.set_title(title)
    ax.grid(True, axis="both")


def main() -> None:
    vw.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=False)
    panel(axL, "", "(a) Momentum $0$")
    panel(axR, "_M", "(b) Momentum $0.9$")
    axL.set_ylabel("gap depth (pp) — up is better")
    axL.legend(loc="lower left", handletextpad=0.5, borderaxespad=0.5)
    vw.finalize(fig, vw.FIG_DIR / "F" / "frontier.png")


if __name__ == "__main__":
    main()
