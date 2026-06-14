"""Stability gap tracker for continual learning.

The *stability gap* (De Lange et al., 2023) is the transient accuracy dip
that occurs on previously learned tasks at the start of training on a new
task.  This module records fine-grained per-step accuracy curves and exposes
several scalar summaries:

  - **max_drop**: largest accuracy drop on any previously seen task within
    the first 200 steps of the new task (legacy metric, fixed window).
  - **gap_depth**: largest accuracy drop on any previously seen task across
    the full record history — same as max_drop but without the 200-step cap.
  - **gap_area**: cumulative drop integrated over the *entire* recorded
    history (trapezoidal rule), summed across all previously seen tasks.
    With ``reference="pre"`` (default) the drop is measured against the
    pre-task baseline; this is useful for a full-run forgetting picture but
    conflates the transient dip with any permanent below-baseline drift: a
    run that recovers fully in 50 steps but settles 0.2 pp below baseline
    will keep accruing area for thousands of steps.  With ``reference="end"``
    the drop is measured against ``min(pre-task baseline, recovered level)``,
    which isolates the genuine dip-and-recover transient (permanent
    forgetting and continued learning both score ≈ 0); exposed in
    ``metrics_summary.json`` as ``stability_gap_area_end``.
  - **recovery_steps**: the earliest step at which *all* previously seen tasks
    simultaneously recover to ≥ 90 % of their pre-task accuracy.

Usage (inside the outer task loop, once task_id > 0)::

    gap_tracker = StabilityGapTracker(
        eval_freq_steps=cfg.eval.stability_gap.eval_freq_steps,
        model=model,
        test_loaders=prev_task_loaders,  # dict or list of DataLoaders
    )

    for global_step, (x, y) in enumerate(train_loader):
        ...  # training step
        if global_step % cfg.eval.stability_gap.eval_freq_steps == 0:
            gap_tracker.record(global_step, all_test_loaders)

    depth  = gap_tracker.gap_depth()
    area   = gap_tracker.gap_area()
    recov  = gap_tracker.recovery_steps()
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class StabilityGapTracker:
    """Tracks per-step accuracy on previously seen tasks during a new task's training.

    The tracker evaluates the model (in inference mode) on a set of held-out
    test loaders at every call to :meth:`record`.  It computes the accuracy
    relative to a **pre-task baseline** captured at construction time, and
    exposes summary statistics.

    Args:
        eval_freq_steps: How often (in training steps) the caller should invoke
            :meth:`record`.  Stored for reference; the caller gates on step count.
        model: The model being trained.  Evaluated in-place (``model.eval()``)
            without gradient tracking.
        test_loaders: DataLoaders for all previously seen tasks.  Passed as a
            list (index = task id) or a dict ``{task_id: DataLoader}``.
            **Pre-task baseline accuracy is computed here at construction.**

    Example::

        tracker = StabilityGapTracker(eval_freq_steps=10, model=model,
                                      test_loaders={0: loader_0, 1: loader_1})
        for step in range(300):
            train_one_step(model, batch)
            if step % 10 == 0:
                tracker.record(step, all_test_loaders)
        print(tracker.gap_depth())        # e.g. 0.18
        print(tracker.gap_area())         # e.g. 14.7  (acc·steps)
        print(tracker.recovery_steps())   # e.g. 120
    """

    def __init__(
        self,
        eval_freq_steps: int,
        model: nn.Module,
        test_loaders: Union[List[DataLoader], Dict[int, DataLoader]],
    ) -> None:
        self.eval_freq_steps = eval_freq_steps
        self.model = model

        # Normalise to {task_id: DataLoader}
        if isinstance(test_loaders, list):
            test_loaders = {j: ldr for j, ldr in enumerate(test_loaders)}

        # Snapshot baseline (pre-task) accuracy for each previously seen task.
        # This is computed once — before the new task's training begins.
        self._pre_task_acc: Dict[int, float] = {
            j: self._eval(ldr) for j, ldr in test_loaders.items()
        }

        # Ordered list of (step_within_task, {task_id: accuracy}) records.
        self._records: List[tuple] = []

        # Global step at the first record() call; used to compute step_within_task.
        self._start_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------

    def record(
        self,
        global_step: int,
        test_loaders: Union[List[DataLoader], Dict[int, DataLoader]],
    ) -> None:
        """Evaluate and store per-task accuracy at *global_step*.

        The ``step_within_task`` stored internally is relative to the first
        call to this method (so it always starts at 0 regardless of
        *global_step*).

        Args:
            global_step: Monotonically increasing step counter (shared across
                tasks in the training loop).
            test_loaders: DataLoaders to evaluate.  Typically the same set
                passed to the constructor, or a superset.
        """
        if self._start_step is None:
            self._start_step = global_step

        step_within_task = global_step - self._start_step

        if isinstance(test_loaders, list):
            test_loaders = {j: ldr for j, ldr in enumerate(test_loaders)}

        accs = {j: self._eval(ldr) for j, ldr in test_loaders.items()}
        self._records.append((step_within_task, accs))
        return accs

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def max_drop(self) -> float:
        """Maximum accuracy drop on any previously seen task within the first 200 steps.

        For each task *j* with a recorded pre-task baseline, computes::

            drop_j = pre_task_acc[j] - min(acc_j  for step_within_task ≤ 200)

        Returns the maximum *drop_j* across all tasks.

        Returns:
            Non-negative float.  Returns 0.0 if there are no records within
            the first 200 steps.
        """
        if not self._records:
            return 0.0

        # Per-task minimum accuracy within the first 200 steps
        min_acc: Dict[int, float] = {}
        for step, accs in self._records:
            if step > 200:
                continue
            for j, acc in accs.items():
                if j not in min_acc or acc < min_acc[j]:
                    min_acc[j] = acc

        if not min_acc:
            return 0.0

        max_d = 0.0
        for j, min_a in min_acc.items():
            if j in self._pre_task_acc:
                drop = self._pre_task_acc[j] - min_a
                if drop > max_d:
                    max_d = drop
        return float(max_d)

    def gap_depth(self) -> float:
        """Maximum accuracy drop on any previously seen task across the full record history.

        Same definition as :meth:`max_drop` but uses *all* recorded steps
        rather than only those within the first 200 steps.  This is the
        right metric when the per-step cadence varies (e.g. tiny-lr runs
        where the dip extends far beyond 200 steps).

        Returns:
            Non-negative float in [0, 1].  Returns 0.0 if no records exist.
        """
        if not self._records:
            return 0.0

        min_acc: Dict[int, float] = {}
        for _step, accs in self._records:
            for j, acc in accs.items():
                if j not in min_acc or acc < min_acc[j]:
                    min_acc[j] = acc

        max_d = 0.0
        for j, min_a in min_acc.items():
            if j in self._pre_task_acc:
                drop = self._pre_task_acc[j] - min_a
                if drop > max_d:
                    max_d = drop
        return float(max_d)

    def gap_area(
        self,
        window_steps: Optional[int] = None,
        reference: str = "pre",
    ) -> float:
        """Trapezoidal integral of the per-task drop curve, summed across tasks.

        For each previously seen task *j*, the per-step drop is
        ``max(0, b_j - acc_j(step))`` — only accuracy *below* the reference
        ``b_j`` contributes (gains above it do not offset losses).  Each
        task's drop curve is integrated over its ``step_within_task`` axis
        by the trapezoidal rule, then summed across tasks.

        The reference ``b_j`` is selected by *reference*:

        - ``"pre"`` (default): ``b_j = pre_task_acc[j]``.  Measures the dip
          against the pre-task baseline; bundles the transient dip with any
          permanent below-baseline drift.
        - ``"end"``: ``b_j = min(pre_task_acc[j], end_mean_j)``, where
          ``end_mean_j`` is the trailing-window mean of task *j*'s curve over
          its last ``max(5, ceil(0.1 * n))`` samples (``n`` = number of
          samples for task *j* within the window).  Referencing the
          recovered level isolates the genuine dip-and-recover transient:
          permanent forgetting (``end_mean_j < pre``) and continued learning
          (``end_mean_j > pre``, so ``b_j = pre``) both contribute ``≈ 0``.
          This is the metric reported as ``stability_gap_area_end``.

        Args:
            window_steps: If given, only integrate over records with
                ``step_within_task < window_steps``.  This is the recommended
                "transient-only" mode: permanent below-baseline drift after
                the dip recovers (or fails to recover) is excluded.  If
                ``None``, integrates over the full history — kept for
                backward compatibility with old summary files.

        Returns:
            Non-negative float with units ``accuracy · steps``.  Returns 0.0
            if fewer than two records fall within the window.
        """
        records = (
            [r for r in self._records if r[0] < window_steps]
            if window_steps is not None
            else self._records
        )
        if len(records) < 2:
            return 0.0

        total = 0.0
        for j, pre_acc_j in self._pre_task_acc.items():
            # Ordered (step, acc) series for task j within the window.
            series = [(step, accs[j]) for step, accs in records if j in accs]
            if len(series) < 2:
                continue

            if reference == "end":
                w = max(5, math.ceil(0.1 * len(series)))
                tail = series[-w:]
                end_mean = sum(a for _, a in tail) / len(tail)
                b_j = min(pre_acc_j, end_mean)
            else:
                b_j = pre_acc_j

            prev_step: Optional[int] = None
            prev_drop: float = 0.0
            for step, acc in series:
                drop = max(0.0, b_j - acc)
                if prev_step is not None:
                    dt = step - prev_step
                    total += dt * 0.5 * (drop + prev_drop)
                prev_step = step
                prev_drop = drop
        return float(total)

    def _end_baselines(
        self, tail_frac: float = 0.1, tail_min: int = 5
    ) -> Dict[int, float]:
        """Per-task settled accuracy: the trailing-window mean of each task's curve.

        For each previously seen task *j*, collects ``acc_j`` over all records
        in order and returns the mean of the last
        ``max(tail_min, ceil(tail_frac · n_j))`` values — an estimate of the
        level the task recovers to by the end of the new task's training.

        Returns:
            ``{task_id: settled_accuracy}``.  A task with no records falls back
            to its pre-task baseline.
        """
        import math

        series: Dict[int, List[float]] = {j: [] for j in self._pre_task_acc}
        for _step, accs in self._records:
            for j, acc in accs.items():
                if j in series:
                    series[j].append(acc)

        baselines: Dict[int, float] = {}
        for j, vals in series.items():
            if not vals:
                baselines[j] = self._pre_task_acc[j]
                continue
            k = max(tail_min, math.ceil(tail_frac * len(vals)))
            tail = vals[-k:]
            baselines[j] = sum(tail) / len(tail)
        return baselines

    def recovery_steps(self) -> Optional[int]:
        """First step at which all previously seen tasks recover to ≥ 90 % of their baseline,
        measured from the point of maximum drop.

        Only meaningful after a drop below the 90 % threshold has been observed.
        Returns ``None`` if no such drop ever occurs (no stability gap) or if
        recovery is never achieved within the recorded history.

        Returns:
            ``step_within_task`` of the first qualifying record after the drop,
            or ``None`` if no drop or no recovery is observed.
        """
        drop_seen = False
        for step, accs in self._records:
            if not drop_seen:
                if not self._all_recovered(accs):
                    drop_seen = True
            else:
                if self._all_recovered(accs):
                    return step
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval(self, loader: DataLoader) -> float:
        """Evaluate ``self.model`` on *loader* and return classification accuracy.

        Moves batches to the same device as the model parameters.  Runs with
        ``torch.no_grad()`` and temporarily sets the model to eval mode.

        Args:
            loader: DataLoader yielding ``(x, y)`` batches.

        Returns:
            Accuracy in [0, 1].  Returns 0.0 if the loader is empty.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return correct / total if total > 0 else 0.0

    def _all_recovered(self, accs: Dict[int, float]) -> bool:
        """Return True iff every baseline task has recovered to ≥ 90 % of its pre-task accuracy.

        Args:
            accs: Mapping ``{task_id: current_accuracy}`` from a single record.

        Returns:
            True if recovery threshold is met for all baseline tasks.
        """
        for j, pre_acc in self._pre_task_acc.items():
            if j not in accs:
                return False
            if accs[j] < 0.90 * pre_acc:
                return False
        return True


class GradientTracker:
    """Records per-step buffer-fidelity diagnostics during the new task's training.

    Captures direct, mechanism-level evidence for the directional and
    magnitude bias contributors to the stability gap (Chapter 3.2.1 / 3.2.2):

      - ``cos(g_replay, g_true)``: directional alignment between the
        buffer-replay gradient ``g_replay`` and the *true past-task gradient*
        ``g_true`` (computed on the full past-task training set — see
        ``ER.set_diagnostic_data``; zero sampling noise).  Persistent values
        ``< 1`` are direct evidence of directional bias from buffer
        sampling / composition error.

      - ``‖g_replay‖ / ‖g_true‖``: magnitude ratio between the buffer
        estimate and the true past gradient.  Values ``< 1`` indicate the
        buffer underestimates the true past-gradient magnitude.

    Both are *buffer-fidelity* metrics — they compare the buffer-derived
    gradient against what the buffer is *trying* to estimate, and are
    logged under the ``true_grad_*`` namespace.  This is distinct from
    ``method.get_last_grad_ratio()`` (logged as ``grad_ratio``), which
    compares the replay gradient to the current-task gradient — the
    update-side magnitude bias of Chapter 3.2.1.

    Args:
        max_steps: Record only the first ``max_steps`` steps of the task.
            Pass ``0`` or ``None`` to record all steps.

    Example::

        grad_tracker = GradientTracker(max_steps=500)
        for step, batch in enumerate(train_loader):
            result = method.observe(...)
            grad_tracker.record(
                step,
                method.get_last_true_grad_cosine(),     # cos(g_replay, g_true)
                method.get_last_true_grad_mag_ratio(),  # ‖g_replay‖/‖g_true‖
            )
        print(grad_tracker.mean_true_grad_cosine())
        print(grad_tracker.mean_true_grad_mag_ratio())
    """

    def __init__(self, max_steps: Optional[int] = None) -> None:
        self.max_steps = max_steps or 0
        self._true_grad_cosines: List[Tuple[int, float]] = []
        self._true_grad_mag_ratios: List[Tuple[int, float]] = []

    def record(
        self,
        step: int,
        true_grad_cosine: float,
        true_grad_mag_ratio: float,
    ) -> None:
        """Store one step's true-gradient buffer-fidelity metrics.

        Args:
            step: Step index within the current task (0-based).
            true_grad_cosine: cos(g_replay, g_true) for this step.
            true_grad_mag_ratio: ‖g_replay‖ / ‖g_true‖ for this step.
        """
        if self.max_steps > 0 and step >= self.max_steps:
            return
        self._true_grad_cosines.append((step, true_grad_cosine))
        self._true_grad_mag_ratios.append((step, true_grad_mag_ratio))

    def mean_true_grad_cosine(self) -> Optional[float]:
        """Mean cos(g_replay, g_true) over all recorded steps."""
        if not self._true_grad_cosines:
            return None
        return float(
            sum(v for _, v in self._true_grad_cosines) / len(self._true_grad_cosines)
        )

    def min_true_grad_cosine(self) -> Optional[float]:
        """Minimum cos(g_replay, g_true) — worst-case directional conflict."""
        if not self._true_grad_cosines:
            return None
        return float(min(v for _, v in self._true_grad_cosines))

    def mean_true_grad_mag_ratio(self) -> Optional[float]:
        """Mean ‖g_replay‖ / ‖g_true‖ over all recorded steps."""
        if not self._true_grad_mag_ratios:
            return None
        return float(
            sum(v for _, v in self._true_grad_mag_ratios)
            / len(self._true_grad_mag_ratios)
        )

    def true_grad_cosine_series(self) -> List[Tuple[int, float]]:
        """All recorded (step, cos(g_replay, g_true)) pairs, insertion order."""
        return list(self._true_grad_cosines)

    def true_grad_mag_ratio_series(self) -> List[Tuple[int, float]]:
        """All recorded (step, ‖g_replay‖/‖g_true‖) pairs, insertion order."""
        return list(self._true_grad_mag_ratios)
