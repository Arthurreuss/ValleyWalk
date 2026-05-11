"""CACL diagnostics CSV writers — T6.3.

Writes three run-level (cross-task) CSV files during CACL training, providing
continuous diagnostic traces that span the entire training run:

  1. ``diagnostics/eigenvalue_spectrum.csv``
       Columns: ``step, lambda_1, ..., lambda_d``
       Written only on Lanczos-recompute steps (every ``amortize_K`` steps).
       Also logged to W&B as a histogram for per-step distribution visualization.

  2. ``diagnostics/trust_radius_history.csv``
       Columns: ``step, radius, rho, accepted``
       Written every training step.

  3. ``diagnostics/cone_fallback_history.csv``
       Columns: ``step, fallback, beta, g_deflated_norm``
       Written every training step.

These are *run-level* files — kept open across all tasks so the step counter
is continuous.  Per-task DualLogger CSVs (from T6.2) already record the same
scalars in per-task chunks; these files provide the concatenated, run-wide view
needed for figures (e.g., trust-radius evolution over the full training run).

Usage::

    with CACLDiagnosticsWriter(method, cfg.outputs.dir, run) as writer:
        for task_id, train_loader, test_loaders in dataset.task_iterator():
            for x, y in train_loader:
                result = method.observe(x, y, task_id)
                diags  = method.get_step_diagnostics()
                writer.log_step(global_step, diags)
                global_step += 1
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional


# Guard import so csv_only mode has zero W&B dependency.
# W&B is only imported inside log_step() when a histogram needs to be logged.
from src.utils.tracking import _NullRun


class CACLDiagnosticsWriter:
    """Run-level CSV writer for CACL-specific diagnostics.

    Opens three CSV files on construction (creating parent directories) and
    keeps them open across the entire training run (all tasks).  Call
    :meth:`close` (or use as a context manager) to flush and close them.

    The writer reads extended state from the CACL method object directly via
    the T6.3 accessor methods (:meth:`~src.methods.cacl.CACL.get_last_eigenvalues`,
    :meth:`~src.methods.cacl.CACL.get_last_lanczos_updated`,
    :meth:`~src.methods.cacl.CACL.get_last_trust_accepted`) rather than storing
    eigenvalue lists inside the ``get_step_diagnostics()`` dict, which must
    contain exactly 7 scalar keys (T5.5 contract).

    Parameters
    ----------
    method : CACL
        The active CACL method instance.  Must expose the three T6.3 accessor
        methods.
    output_dir : str
        Base output directory (e.g. ``"results/"``).  Files are written to
        ``{output_dir}/diagnostics/``.
    run : wandb.Run | _NullRun
        Active W&B run (or no-op null run).  Used to log eigenvalue histograms
        when W&B is enabled.
    """

    def __init__(self, method: Any, output_dir: str, run: Any, local_cfg: Any = None) -> None:
        self._method = method
        self._run = run

        # Resolve tracking.local flags (default to True when cfg not provided)
        def _flag(name: str, default: bool) -> bool:
            return bool(getattr(local_cfg, name, default)) if local_cfg is not None else default

        self._log_trust  = _flag("log_trust_radius",  True)
        self._log_cone   = _flag("log_cone_fallback",  True)
        self._log_eig    = _flag("log_eigenvalues",    False)

        diag_dir = os.path.join(output_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        # ── Trust radius history (fixed schema, header written at open) ────
        if self._log_trust:
            trust_path = os.path.join(diag_dir, "trust_radius_history.csv")
            self._trust_file = open(trust_path, "w", newline="")
            self._trust_writer: Optional[csv.DictWriter] = csv.DictWriter(
                self._trust_file,
                fieldnames=["step", "radius", "rho", "accepted"],
            )
            self._trust_writer.writeheader()
            self._trust_file.flush()
        else:
            self._trust_file = None
            self._trust_writer = None

        # ── Cone fallback history (fixed schema, header written at open) ───
        if self._log_cone:
            cone_path = os.path.join(diag_dir, "cone_fallback_history.csv")
            self._cone_file = open(cone_path, "w", newline="")
            self._cone_writer: Optional[csv.DictWriter] = csv.DictWriter(
                self._cone_file,
                fieldnames=["step", "fallback", "beta", "g_deflated_norm"],
            )
            self._cone_writer.writeheader()
            self._cone_file.flush()
        else:
            self._cone_file = None
            self._cone_writer = None

        # ── Gradient ratio history (fixed schema, header written at open) ───
        grad_ratio_path = os.path.join(diag_dir, "grad_ratio_history.csv")
        self._grad_ratio_file = open(grad_ratio_path, "w", newline="")
        self._grad_ratio_writer: csv.DictWriter = csv.DictWriter(
            self._grad_ratio_file,
            fieldnames=["step", "grad_ratio"],
        )
        self._grad_ratio_writer.writeheader()
        self._grad_ratio_file.flush()

        # ── Eigenvalue spectrum (schema lazily determined on first write) ──
        # The number of retained eigenvalues d may vary (e.g., if the
        # parameter count is smaller than the configured d).  We set the
        # column count from the first actual Lanczos output to avoid
        # hard-coding d here.
        if self._log_eig:
            eig_path = os.path.join(diag_dir, "eigenvalue_spectrum.csv")
            self._eig_file = open(eig_path, "w", newline="")
        else:
            self._eig_file = None
        self._eig_writer: Optional[csv.DictWriter] = None  # lazily initialised

    # ------------------------------------------------------------------
    # Core logging method
    # ------------------------------------------------------------------

    def log_step(self, step: int, diagnostics: Dict[str, Any]) -> None:
        """Write one row to the appropriate CSV files for a single training step.

        Trust radius and cone fallback rows are written every step.  The
        eigenvalue spectrum row is written only when the CACL method reports
        that Lanczos was recomputed this step
        (``method.get_last_lanczos_updated() == True``).

        Parameters
        ----------
        step : int
            Global training step (monotonically increasing across all tasks).
        diagnostics : dict
            Diagnostics dict returned by
            ``method.get_step_diagnostics()`` — must contain
            ``trust_radius``, ``trust_rho``, ``cone_fallback``,
            ``beta``, ``g_deflated_norm``.
        """
        # ── Trust radius — every step ──────────────────────────────────────
        if self._log_trust:
            self._trust_writer.writerow({
                "step":     step,
                "radius":   diagnostics.get("trust_radius", 0.0),
                "rho":      diagnostics.get("trust_rho", 0.0),
                "accepted": self._method.get_last_trust_accepted(),
            })
            self._trust_file.flush()

        # ── Cone fallback — every step ─────────────────────────────────────
        if self._log_cone:
            self._cone_writer.writerow({
                "step":           step,
                "fallback":       diagnostics.get("cone_fallback", False),
                "beta":           diagnostics.get("beta", 0.0),
                "g_deflated_norm": diagnostics.get("g_deflated_norm", 0.0),
            })
            self._cone_file.flush()

        # ── Gradient ratio — every step ────────────────────────────────────
        self._grad_ratio_writer.writerow({
            "step":       step,
            "grad_ratio": self._method.get_last_grad_ratio(),
        })
        self._grad_ratio_file.flush()

        # ── Eigenvalue spectrum — only on Lanczos-recompute steps ──────────
        if self._log_eig and self._method.get_last_lanczos_updated():
            eigenvalues: List[float] = self._method.get_last_eigenvalues()
            if eigenvalues:
                self._write_eigenvalues(step, eigenvalues)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_eigenvalues(self, step: int, eigenvalues: List[float]) -> None:
        """Write one eigenvalue spectrum row and log a W&B histogram.

        The CSV header is lazily initialised on the first call, using the
        actual number of eigenvalues returned (top-d from Lanczos).

        Parameters
        ----------
        step : int
            Global training step.
        eigenvalues : list of float
            Top-d Ritz eigenvalues, descending order.
        """
        # Lazily create the eigenvalue CSV writer with the correct column count.
        if self._eig_writer is None:
            fieldnames = ["step"] + [
                f"lambda_{i + 1}" for i in range(len(eigenvalues))
            ]
            self._eig_writer = csv.DictWriter(
                self._eig_file,
                fieldnames=fieldnames,
                extrasaction="ignore",   # tolerate future d changes gracefully
            )
            self._eig_writer.writeheader()

        row: Dict[str, Any] = {"step": step}
        for i, lam in enumerate(eigenvalues):
            row[f"lambda_{i + 1}"] = lam
        self._eig_writer.writerow(row)
        self._eig_file.flush()

        # W&B eigenvalue histogram (no-op for _NullRun)
        if not isinstance(self._run, _NullRun):
            import wandb  # noqa: PLC0415
            self._run.log({"eigenvalues": wandb.Histogram(eigenvalues)}, step=step)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close any open CSV files.  Idempotent — safe to call twice."""
        for fh in (self._eig_file, self._trust_file, self._cone_file, self._grad_ratio_file):
            if fh is not None and not fh.closed:
                fh.close()

    def __enter__(self) -> "CACLDiagnosticsWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
