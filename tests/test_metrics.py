"""Tests for ContinualMetrics (De Lange et al., ICLR 2023 metrics).

Scenarios:
  (a) Three-task scenario with per-step history — verifies ACC, FORG, min-ACC,
      WF_w, WP_w, WC-ACC against hand-computed values.
  (b) No-forgetting scenario — FORG and min-ACC are both 0.
  (c) Edge cases: single task, empty metrics.

Setup:

  Three tasks, evaluated at steps 0, 10, 20 (task boundaries) and at steps
  5, 15 (mid-task).

  Task boundary accuracies (R[i,j] = acc on task j right after training task i):

           task0  task1  task2
    time0 [ 0.90  —      —   ]   step=10, end of task 0
    time1 [ 0.70  0.80  —   ]   step=20, end of task 1
    time2 [ 0.50  0.60  0.85]   step=30, end of task 2

  Mid-task accuracies (during task 1 and task 2 training):
    step=15: task0=0.65, task1=0.75
    step=25: task0=0.55, task1=0.58, task2=0.82

  Full history per task:
    task0: [(5,0.88), (10,0.90), (15,0.65), (20,0.70), (25,0.55), (30,0.50)]
    task1: [(15,0.75), (20,0.80), (25,0.58), (30,0.60)]
    task2: [(25,0.82), (30,0.85)]

  Task end steps: {0: 10, 1: 20, 2: 30}

Hand-computed expected values:

  ACC = (0.50 + 0.60 + 0.85) / 3 = 0.6500

  FORG:
    i=0: A(E0, f_{t|T0}) − A(E0, f_{t|T2}) = 0.90 − 0.50 = 0.40
    i=1: A(E1, f_{t|T1}) − A(E1, f_{t|T2}) = 0.80 − 0.60 = 0.20
    FORG = (0.40 + 0.20) / 2 = 0.30

  min-ACC:
    i=0: min of task0 accs after step 10 = min(0.65, 0.70, 0.55, 0.50) = 0.50
         (steps 15, 20, 25, 30 — 20 is also the boundary of task 1 but > 10)
         Wait — step 20 is the boundary of task 1. notify_task_end(1, 20) then
         record_step(0, 20, 0.70). So 0.70 IS after step 10 (end of task 0).
    i=0: min(0.65, 0.70, 0.55, 0.50) = 0.50
    i=1: min of task1 accs after step 20 = min(0.58, 0.60) = 0.58
    min-ACC = (0.50 + 0.58) / 2 = 0.54

  WF_10 and WF_100 (window >= full history → same as WF_∞ for small sequences):
    task0 full history: [0.88, 0.90, 0.65, 0.70, 0.55, 0.50]
      Max drop: peak before each n
        n=1(0.90): peak=0.88, drop=−0.02 → 0
        n=2(0.65): peak=max(0.88,0.90)=0.90, drop=0.25
        n=3(0.70): peak=0.90, drop=0.20
        n=4(0.55): peak=0.90, drop=0.35
        n=5(0.50): peak=0.90, drop=0.40
      WF task0 = 0.40
    task1 full history: [0.75, 0.80, 0.58, 0.60]
      n=1(0.80): peak=0.75, drop=−0.05 → 0
      n=2(0.58): peak=max(0.75,0.80)=0.80, drop=0.22
      n=3(0.60): peak=0.80, drop=0.20
      WF task1 = 0.22
    task2 full history: [0.82, 0.85]
      n=1(0.85): peak=0.82, drop=−0.03 → 0
      WF task2 = 0.0
    WF_100 = (0.40 + 0.22 + 0.0) / 3 = 0.2067 (rounded)

  WF_2 (window of 2 consecutive evals):
    task0: windows of 2: [0.88,0.90],[0.90,0.65],[0.65,0.70],[0.70,0.55],[0.55,0.50]
      drops: 0, 0.25, 0, 0.15, 0.05 → WF_2 task0 = 0.25
    task1: [0.75,0.80],[0.80,0.58],[0.58,0.60]
      drops: 0, 0.22, 0 → WF_2 task1 = 0.22
    task2: [0.82,0.85] → drop = 0
    WF_2 = (0.25 + 0.22 + 0.0) / 3 = 0.15667

  WP_100 (window >= full history):
    task0: [0.88, 0.90, 0.65, 0.70, 0.55, 0.50]
      max gain: trough before each n
        n=1(0.90): trough=0.88, gain=0.02
        n=2(0.65): trough=min(0.88,0.90)=0.88, gain=−0.23 → 0
        n=3(0.70): trough=0.65, gain=0.05
        n=4(0.55): trough=0.65, gain=−0.10 → 0
        n=5(0.50): trough=0.50, gain=0
      WP task0 = 0.05
    task1: [0.75, 0.80, 0.58, 0.60]
      n=1(0.80): trough=0.75, gain=0.05
      n=2(0.58): trough=0.75, gain=−0.17 → 0
      n=3(0.60): trough=0.58, gain=0.02
      WP task1 = 0.05
    task2: [0.82, 0.85]
      n=1(0.85): trough=0.82, gain=0.03
      WP task2 = 0.03
    WP_100 = (0.05 + 0.05 + 0.03) / 3 = 0.04333

  WC-ACC:
    N=3, curr = A(E2, f_{t|T2}) = 0.85, min-ACC = 0.54
    WC-ACC = (1/3)*0.85 + (2/3)*0.54 = 0.2833 + 0.36 = 0.6433
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import ContinualMetrics


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _build_three_task_metrics() -> ContinualMetrics:
    """Build a ContinualMetrics with the 3-task scenario described above."""
    m = ContinualMetrics()

    # ---- Task 0 training (steps 0–10) ----
    m.record_step(task_j=0, step=5,  accuracy=0.88)   # mid-task eval
    m.notify_task_end(task_i=0, step=10)
    m.record_step(task_j=0, step=10, accuracy=0.90)   # boundary eval

    # ---- Task 1 training (steps 10–20) ----
    m.record_step(task_j=0, step=15, accuracy=0.65)   # mid-task eval task0
    m.record_step(task_j=1, step=15, accuracy=0.75)   # mid-task eval task1
    m.notify_task_end(task_i=1, step=20)
    m.record_step(task_j=0, step=20, accuracy=0.70)   # boundary eval task0
    m.record_step(task_j=1, step=20, accuracy=0.80)   # boundary eval task1

    # ---- Task 2 training (steps 20–30) ----
    m.record_step(task_j=0, step=25, accuracy=0.55)   # mid-task eval task0
    m.record_step(task_j=1, step=25, accuracy=0.58)   # mid-task eval task1
    m.record_step(task_j=2, step=25, accuracy=0.82)   # mid-task eval task2
    m.notify_task_end(task_i=2, step=30)
    m.record_step(task_j=0, step=30, accuracy=0.50)   # boundary eval task0
    m.record_step(task_j=1, step=30, accuracy=0.60)   # boundary eval task1
    m.record_step(task_j=2, step=30, accuracy=0.85)   # boundary eval task2

    return m


# ---------------------------------------------------------------------------
# (a) Three-task scenario — all metrics
# ---------------------------------------------------------------------------

class TestThreeTaskScenario:

    def setup_method(self):
        self.m = _build_three_task_metrics()

    def test_accuracy_matrix_shape_and_values(self):
        R = self.m.get_accuracy_matrix()
        assert R.shape == (3, 3)
        # Diagonal and lower triangle
        assert pytest.approx(R[0, 0], abs=1e-9) == 0.90
        assert pytest.approx(R[1, 0], abs=1e-9) == 0.70
        assert pytest.approx(R[1, 1], abs=1e-9) == 0.80
        assert pytest.approx(R[2, 0], abs=1e-9) == 0.50
        assert pytest.approx(R[2, 1], abs=1e-9) == 0.60
        assert pytest.approx(R[2, 2], abs=1e-9) == 0.85
        # Upper triangle should be NaN
        assert np.isnan(R[0, 1]) and np.isnan(R[0, 2]) and np.isnan(R[1, 2])

    def test_acc(self):
        expected = (0.50 + 0.60 + 0.85) / 3
        assert pytest.approx(self.m.average_accuracy(), abs=1e-9) == expected

    def test_forg(self):
        # FORG = ((0.90−0.50) + (0.80−0.60)) / 2 = (0.40 + 0.20) / 2 = 0.30
        expected = (0.40 + 0.20) / 2
        assert pytest.approx(self.m.forgetting(), abs=1e-9) == expected

    def test_min_acc(self):
        # task0 after step 10: [0.65, 0.70, 0.55, 0.50] → min = 0.50
        # task1 after step 20: [0.58, 0.60] → min = 0.58
        expected = (0.50 + 0.58) / 2
        assert pytest.approx(self.m.min_acc(), abs=1e-9) == expected

    def test_wf_large_window(self):
        # WF_100 covers full history (only 6/4/2 evals per task)
        # task0: max drop = 0.90 − 0.50 = 0.40
        # task1: max drop = 0.80 − 0.58 = 0.22
        # task2: max drop = 0 (only increase 0.82→0.85)
        expected = (0.40 + 0.22 + 0.0) / 3
        assert pytest.approx(self.m.windowed_forgetting(100), abs=1e-9) == expected

    def test_wf_window_2(self):
        # Window of 2 consecutive evals
        # task0 consecutive pairs: (0.88,0.90)→0, (0.90,0.65)→0.25,
        #   (0.65,0.70)→0, (0.70,0.55)→0.15, (0.55,0.50)→0.05 → max=0.25
        # task1: (0.75,0.80)→0, (0.80,0.58)→0.22, (0.58,0.60)→0 → max=0.22
        # task2: (0.82,0.85)→0 → max=0
        expected = (0.25 + 0.22 + 0.0) / 3
        assert pytest.approx(self.m.windowed_forgetting(2), abs=1e-9) == expected

    def test_wp_large_window(self):
        # WP_100 covers full history
        # task0: gains per n: 0.02, 0, 0.05, 0, 0 → max=0.05
        # task1: gains: 0.05, 0, 0.02 → max=0.05
        # task2: gain=0.03
        expected = (0.05 + 0.05 + 0.03) / 3
        assert pytest.approx(self.m.windowed_plasticity(100), abs=1e-9) == expected

    def test_wc_acc(self):
        # WC-ACC = (1/3)*0.85 + (2/3)*0.54
        min_acc = (0.50 + 0.58) / 2
        expected = (1 / 3) * 0.85 + (2 / 3) * min_acc
        assert pytest.approx(self.m.wc_acc(), abs=1e-9) == expected

    def test_to_dict_keys(self):
        d = self.m.to_dict()
        assert set(d.keys()) == {"ACC", "FORG", "min_ACC", "WF10", "WF100", "WP10", "WP100", "WC_ACC"}

    def test_to_dict_values_consistent(self):
        d = self.m.to_dict()
        assert pytest.approx(d["ACC"])     == self.m.average_accuracy()
        assert pytest.approx(d["FORG"])    == self.m.forgetting()
        assert pytest.approx(d["min_ACC"]) == self.m.min_acc()
        assert pytest.approx(d["WF10"])    == self.m.windowed_forgetting(10)
        assert pytest.approx(d["WF100"])   == self.m.windowed_forgetting(100)
        assert pytest.approx(d["WP10"])    == self.m.windowed_plasticity(10)
        assert pytest.approx(d["WP100"])   == self.m.windowed_plasticity(100)
        assert pytest.approx(d["WC_ACC"])  == self.m.wc_acc()

    def test_wf_geq_wf_small_window(self):
        """WF with larger window >= WF with smaller window (more history = more forgetting)."""
        assert self.m.windowed_forgetting(100) >= self.m.windowed_forgetting(2)


# ---------------------------------------------------------------------------
# (b) No-forgetting scenario
# ---------------------------------------------------------------------------

class TestNoForgetting:
    """When accuracy never drops after initial training, FORG and min-ACC are 0."""

    def setup_method(self):
        m = ContinualMetrics()
        # task0: accuracy stays at 0.90 forever
        m.notify_task_end(task_i=0, step=10)
        m.record_step(task_j=0, step=10, accuracy=0.90)

        m.record_step(task_j=0, step=15, accuracy=0.90)  # no drop
        m.notify_task_end(task_i=1, step=20)
        m.record_step(task_j=0, step=20, accuracy=0.90)
        m.record_step(task_j=1, step=20, accuracy=0.80)

        m.record_step(task_j=0, step=25, accuracy=0.90)
        m.record_step(task_j=1, step=25, accuracy=0.80)
        m.notify_task_end(task_i=2, step=30)
        m.record_step(task_j=0, step=30, accuracy=0.90)
        m.record_step(task_j=1, step=30, accuracy=0.80)
        m.record_step(task_j=2, step=30, accuracy=0.85)

        self.m = m

    def test_forg_is_zero(self):
        assert pytest.approx(self.m.forgetting(), abs=1e-9) == 0.0

    def test_min_acc_equals_boundary_acc(self):
        # min-ACC = (min(0.90,0.90,0.90,0.90) + min(0.80,0.80)) / 2
        expected = (0.90 + 0.80) / 2
        assert pytest.approx(self.m.min_acc(), abs=1e-9) == expected

    def test_wf_is_zero(self):
        assert pytest.approx(self.m.windowed_forgetting(100), abs=1e-9) == 0.0

    def test_wc_acc_geq_acc(self):
        # WC-ACC uses min-ACC which equals boundary acc here, so it should be close to ACC
        # (WC-ACC <= ACC is guaranteed by the paper; when no forgetting they should be equal)
        assert pytest.approx(self.m.wc_acc(), abs=1e-9) == self.m.average_accuracy()


# ---------------------------------------------------------------------------
# (c) Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_task(self):
        m = ContinualMetrics()
        m.notify_task_end(task_i=0, step=10)
        m.record_step(task_j=0, step=10, accuracy=0.85)

        assert pytest.approx(m.average_accuracy(), abs=1e-9) == 0.85
        assert pytest.approx(m.forgetting(), abs=1e-9) == 0.0   # N < 2
        assert pytest.approx(m.min_acc(), abs=1e-9) == 0.0      # N < 2
        assert pytest.approx(m.wc_acc(), abs=1e-9) == 0.85      # single task = current acc

    def test_empty_metrics(self):
        m = ContinualMetrics()
        assert m.average_accuracy() == 0.0
        assert m.forgetting() == 0.0
        assert m.min_acc() == 0.0
        assert m.windowed_forgetting(10) == 0.0
        assert m.windowed_plasticity(10) == 0.0
        assert m.wc_acc() == 0.0

    def test_wf_single_eval_per_task(self):
        """With only one eval per task, windowed metrics are 0 (no pairs to compare)."""
        m = ContinualMetrics()
        m.notify_task_end(task_i=0, step=10)
        m.record_step(task_j=0, step=10, accuracy=0.90)
        m.notify_task_end(task_i=1, step=20)
        m.record_step(task_j=0, step=20, accuracy=0.70)
        m.record_step(task_j=1, step=20, accuracy=0.80)
        # task0 has 2 evals, task1 has 1 → WF can be computed for task0
        # task0: drop 0.90 → 0.70 = 0.20
        assert self.m_or(m).windowed_forgetting(100) == pytest.approx((0.20 + 0.0) / 2, abs=1e-9)

    @staticmethod
    def m_or(m):
        return m
