"""Continual-learning evaluation metrics (De Lange et al., ICLR 2023).

Implements all metrics from "A Continual Learning Survey" (ICLR 2023):

Stability metrics:
  FORG    — average accuracy drop from right-after-learning to final model
  min-ACC — average minimum accuracy on previous tasks since they were learned
  WF_w    — windowed forgetting: max accuracy drop in any w-eval window

Plasticity metric:
  WP_w    — windowed plasticity: max accuracy gain in any w-eval window

Trade-off metrics:
  ACC     — average accuracy across all tasks at the final model
  WC-ACC  — worst-case accuracy: blends current-task acc with min-ACC

All metrics are derived from step-level accuracy recordings per task plus
task-boundary markers.

Recording API::

    metrics = ContinualMetrics()

    # During training — call at every periodic eval step:
    metrics.record_step(task_j=j, step=global_step, accuracy=acc)

    # After each task's training loop ends (before boundary eval):
    metrics.notify_task_end(task_i=task_id, step=global_step)

    # Task-boundary eval (same global_step as notify_task_end):
    metrics.record_step(task_j=j, step=global_step, accuracy=acc)

    # At the end:
    print(metrics.to_dict())
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class ContinualMetrics:
    """Accumulates per-step accuracy observations and computes CL metrics.

    Step-level history is stored per task, enabling both task-boundary metrics
    (ACC, FORG) and within-training metrics (min-ACC, WF_w, WP_w, WC-ACC).

    Formulas (N tasks, 0-based indices, E_i = evaluation set of task i):

        ACC      = (1/N) Σ_j A(E_j, f_{t|T_{N-1}})

        FORG     = (1/(N-1)) Σ_{i=0}^{N-2} [A(E_i, f_{t|Ti}) − A(E_i, f_{t|T_{N-1}})]

        min-ACC  = (1/(N-1)) Σ_{i=0}^{N-2} min{ A(E_i, f_n) | n > t|Ti| }

        WF_w     = (1/N) Σ_i max_t ∆w,−_{t,Ei}
                   where ∆w,−_{t,Ei} = max_{m<n, m,n∈[t-w+1,t]} (A(E_i,f_m)−A(E_i,f_n))

        WP_w     = (1/N) Σ_i max_t ∆w,+_{t,Ei}
                   where ∆w,+_{t,Ei} = max_{m>n, m,n∈[t-w+1,t]} (A(E_i,f_m)−A(E_i,f_n))

        WC-ACC   = (1/N)·A(E_{N-1}, f_{t|T_{N-1}}) + (1 − 1/N)·min-ACC
    """

    def __init__(self) -> None:
        # task_j → list of (step, accuracy), in insertion (ascending step) order
        self._history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # task_i → global step when task i's training ended (= boundary eval step)
        self._task_end_step: Dict[int, int] = {}
        self._num_tasks: int = 0

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_step(self, task_j: int, step: int, accuracy: float) -> None:
        """Record accuracy on task_j at global training step.

        Call this at every periodic eval and at every task-boundary eval.

        Args:
            task_j:   Zero-based index of the evaluated task.
            step:     Global training step (monotonically increasing).
            accuracy: Test accuracy in [0, 1].
        """
        self._history[task_j].append((step, float(accuracy)))
        self._num_tasks = max(self._num_tasks, task_j + 1)

    def notify_task_end(self, task_i: int, step: int) -> None:
        """Mark that training on task_i ended at global step.

        Call this *before* the task-boundary evaluation so that the boundary
        eval (recorded at the same step) is treated as the reference point for
        FORG (A(E_i, f_{t|Ti})).  Subsequent step recordings for task i are
        included in min-ACC and WF_w.

        Args:
            task_i: Zero-based index of the task that just finished training.
            step:   Global step at which training ended.
        """
        self._task_end_step[task_i] = step
        self._num_tasks = max(self._num_tasks, task_i + 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _acc_at_boundary(self, task_j: int, task_i: int) -> Optional[float]:
        """Return the last accuracy for task_j recorded at or before task_i's end step."""
        end = self._task_end_step.get(task_i)
        if end is None:
            return None
        result = None
        for s, acc in self._history[task_j]:
            if s <= end:
                result = acc
            else:
                break
        return result

    def _accs_after_task(self, task_j: int, task_i: int) -> List[float]:
        """Return accuracy values for task_j recorded strictly after task_i ended."""
        end = self._task_end_step.get(task_i)
        if end is None:
            return []
        return [acc for s, acc in self._history[task_j] if s > end]

    def _all_accs(self, task_j: int) -> List[float]:
        """Return all recorded accuracy values for task_j in chronological order."""
        return [acc for _, acc in self._history[task_j]]

    @staticmethod
    def _max_drop_in_window(accs: List[float], w: int) -> float:
        """Max accuracy drop in any sliding window of w consecutive evaluations.

        For each right-endpoint n, the max drop is:
            max(accs[max(0, n-w+1) : n]) − accs[n]

        Taking the max over all n gives WF_w for this task.
        """
        if len(accs) < 2:
            return 0.0
        max_drop = 0.0
        for n in range(1, len(accs)):
            start = max(0, n - w + 1)
            peak = max(accs[start:n])
            drop = peak - accs[n]
            if drop > max_drop:
                max_drop = drop
        return max_drop

    @staticmethod
    def _max_gain_in_window(accs: List[float], w: int) -> float:
        """Max accuracy gain in any sliding window of w consecutive evaluations.

        For each right-endpoint n, the max gain is:
            accs[n] − min(accs[max(0, n-w+1) : n])

        Taking the max over all n gives WP_w for this task.
        """
        if len(accs) < 2:
            return 0.0
        max_gain = 0.0
        for n in range(1, len(accs)):
            start = max(0, n - w + 1)
            trough = min(accs[start:n])
            gain = accs[n] - trough
            if gain > max_gain:
                max_gain = gain
        return max_gain

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def average_accuracy(self) -> float:
        """ACC: mean accuracy across all tasks at the final model.

        ACC = (1/N) Σ_{j=0}^{N-1} A(E_j, f_{t|T_{N-1}})
        """
        N = self._num_tasks
        if N == 0:
            return 0.0
        accs = [
            self._acc_at_boundary(j, N - 1)
            for j in range(N)
        ]
        valid = [a for a in accs if a is not None]
        return float(np.mean(valid)) if valid else 0.0

    def forgetting(self) -> float:
        """FORG: average accuracy drop from right-after-learning to final model.

        FORG = (1/(N-1)) Σ_{i=0}^{N-2}
                   [A(E_i, f_{t|Ti}) − A(E_i, f_{t|T_{N-1}})]

        Positive FORG indicates catastrophic forgetting; negative indicates
        backward knowledge transfer.  Returns 0.0 when N < 2.
        """
        N = self._num_tasks
        if N < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(N - 1):
            acc_diag  = self._acc_at_boundary(i, i)      # A(E_i, f_{t|Ti})
            acc_final = self._acc_at_boundary(i, N - 1)  # A(E_i, f_{t|T_{N-1}})
            if acc_diag is not None and acc_final is not None:
                total += acc_diag - acc_final
                count += 1
        return float(total / count) if count > 0 else 0.0

    def min_acc(self) -> float:
        """min-ACC: average minimum accuracy on previous tasks since learned.

        min-ACC = (1/(N-1)) Σ_{i=0}^{N-2}
                      min{ A(E_i, f_n) | n > t|Ti| }

        Captures worst-case retention of previously learned knowledge.
        Returns 0.0 when N < 2.
        """
        N = self._num_tasks
        if N < 2:
            return 0.0
        mins = []
        for i in range(N - 1):
            accs = self._accs_after_task(i, i)
            if accs:
                mins.append(min(accs))
        return float(np.mean(mins)) if mins else 0.0

    def windowed_forgetting(self, w: int) -> float:
        """WF_w: max accuracy drop in any window of w consecutive evaluations.

        WF^w = (1/N) Σ_i max_t ∆w,−_{t,Ei}

        where ∆w,−_{t,Ei} = max_{m<n, m,n∈[t-w+1,t]} (A(E_i,f_m) − A(E_i,f_n))

        A worst-case stability metric; larger values indicate more forgetting.

        Args:
            w: Window size in number of consecutive evaluations (e.g. 10, 100).
        """
        N = self._num_tasks
        if N == 0:
            return 0.0
        per_task = [
            self._max_drop_in_window(self._all_accs(j), w)
            for j in range(N)
        ]
        return float(np.mean(per_task))

    def windowed_plasticity(self, w: int) -> float:
        """WP_w: max accuracy gain in any window of w consecutive evaluations.

        WP^w = (1/N) Σ_i max_t ∆w,+_{t,Ei}

        where ∆w,+_{t,Ei} = max_{m>n, m,n∈[t-w+1,t]} (A(E_i,f_m) − A(E_i,f_n))

        A plasticity metric; larger values indicate faster learning.

        Args:
            w: Window size in number of consecutive evaluations (e.g. 10, 100).
        """
        N = self._num_tasks
        if N == 0:
            return 0.0
        per_task = [
            self._max_gain_in_window(self._all_accs(j), w)
            for j in range(N)
        ]
        return float(np.mean(per_task))

    def wc_acc(self) -> float:
        """WC-ACC: worst-case accuracy trade-off at the final model.

        WC-ACC = (1/N)·A(E_{N-1}, f_{t|T_{N-1}}) + (1 − 1/N)·min-ACC

        Blends current-task plasticity with worst-case stability on previous
        tasks.  Provides a lower bound on ACC.  Returns 0.0 when N == 0;
        equals the single-task accuracy when N == 1.
        """
        N = self._num_tasks
        if N == 0:
            return 0.0
        curr = self._acc_at_boundary(N - 1, N - 1)
        if curr is None:
            return 0.0
        if N == 1:
            return float(curr)
        return float((1.0 / N) * curr + (1.0 - 1.0 / N) * self.min_acc())

    # ------------------------------------------------------------------
    # Accuracy matrix (for logging / checkpointing)
    # ------------------------------------------------------------------

    def get_accuracy_matrix(self) -> np.ndarray:
        """Task-boundary accuracy matrix R where R[i,j] = acc on task j after training task i.

        Entries are NaN for task-pairs that were not observed (j > i).

        Returns:
            np.ndarray of shape (N, N), dtype float64.
        """
        N = self._num_tasks
        R = np.full((N, N), np.nan, dtype=np.float64)
        for i in range(N):
            for j in range(i + 1):
                a = self._acc_at_boundary(j, i)
                if a is not None:
                    R[i, j] = a
        return R

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as a flat dictionary.

        Returns:
            Dict with keys ``ACC``, ``FORG``, ``min_ACC``, ``WF10``,
            ``WF100``, ``WP10``, ``WP100``, ``WC_ACC``.
        """
        return {
            "ACC":     self.average_accuracy(),
            "FORG":    self.forgetting(),
            "min_ACC": self.min_acc(),
            "WF10":    self.windowed_forgetting(10),
            "WF100":   self.windowed_forgetting(100),
            "WP10":    self.windowed_plasticity(10),
            "WP100":   self.windowed_plasticity(100),
            "WC_ACC":  self.wc_acc(),
        }
