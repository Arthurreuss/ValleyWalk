# Loss Landscape Visualization — Figure Description

**File:** `loss_landscape_visualization.png`
**Source:** Rendered from the interactive HTML sandbox `loss_landscape_visualization.html` (default state, gradient-magnitude toggle OFF — all gradient arrows are normalized to a fixed length).

## Overall layout

Four square panels arranged left-to-right on a black background, each with a white title bar:

1. **Task A loss (Reference)** — red heatmap
2. **Task B loss (Current)** — blue heatmap
3. **Joint loss: A + B** — purple heatmap
4. **Curriculum: A + λB** — red heatmap, highlighted with a gold border (the focus panel)

All panels share the same 2D parameter space and the same set of landmarks:

- **A** — red dot, upper-left: minimum of Task A.
- **B** — blue dot, upper-right: minimum of Task B.
- **A + B** — purple dot, bottom-center: minimum of the joint loss.
- A chain of five **white eval-point dots** strung along a path between A and A+B, each emitting gradient arrows.

## Panel 1 — Task A loss (Reference), red

- Smooth red loss contours: darkest (highest loss) in the corners, lightest (lowest loss) along a diagonal band from upper-left toward lower-center.
- Landmarks A, B, A+B as above.
- A **black dashed line** from A to A+B: the straight-line interpolation between the single-task solution and the joint minimum.
- A **green curved line** bowing to the right of the dashed line: the gradient-descent / valley-following trajectory.
- At each white eval point: a short **dark-red arrow** (Task A descent direction) and a longer **blue arrow** (Task B direction).

## Panel 2 — Task B loss (Current), blue

- Identical geometry, landmarks, dashed line, green curve, and eval points, drawn over a **blue** heatmap.
- The low-loss (white) region sits toward the upper-right, near B.
- Same red and blue gradient arrows at each eval point.

## Panel 3 — Joint loss: A + B, purple

- Same landmarks and trajectory over a **purple** heatmap with a single combined basin (lightest near the A+B minimum at bottom-center).
- At each eval point: red arrows, blue arrows, and additional **indigo / dark-purple arrows** representing the joint (summed) gradient direction.

## Panel 4 — Curriculum: A + λB, red (gold border)

- A **red** heatmap whose basin is reshaped into a tilted elliptical valley running diagonally from top toward bottom-center, reflecting a small λ weighting on Task B.
- A and B dots in the same upper positions.
- A chain of white eval points runs down the valley, each carrying a short **gold / olive arrow** (the curriculum gradient, direction of A + λB).
- Near the bottom: the **"Curr"** minimum — a yellow/gold dot with red and cyan arrows meeting at it, just above the **A + B** purple dot and label.
- No green trajectory or dashed interpolation line in this panel (suppressed for the curriculum panel by design).

## Notes / state captured

- This is the **default state** of the interactive tool rendered to PNG.
- The **gradient-magnitude toggle is OFF**: all red/blue/purple arrows are uniform fixed length (normalized direction only), so the figure conveys gradient *direction*, not magnitude.
- Eval points are in their default positions; λ is small (the curriculum basin is only mildly deformed toward B).
