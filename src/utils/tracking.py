"""Dual-write tracking utility — T6.1.

Every metric is written to both a local CSV (ground truth for figures/archival)
and a W&B run (live monitoring).  When ``cfg.tracking.backend == "csv_only"``,
all W&B calls become no-ops so the code runs identically without a W&B
account or internet access — no ``if/else`` scattered through training code.

Public API
----------
init_tracking(cfg)
    Start a W&B run (or null-run) and capture git state.
    Returns ``(run, git_commit, git_dirty)``.

DualLogger(csv_path, wandb_run)
    Per-task CSV + W&B writer.  First ``log()`` call creates the header.
    Call ``close()`` when done (or rely on ``__del__`` / context manager).

save_and_log_checkpoint(model, task_id, run, cfg, local_dir)
    Save model state-dict locally; optionally upload as W&B Artifact.

save_and_log_results(results_dir, run, cfg)
    Upload ``results/`` directory as a W&B Artifact at run end.

log_accuracy_matrix_table(accuracy_matrix, run, num_tasks)
    Upload R[i,j] as a W&B Table for dashboard heatmap comparison.

log_metrics_summary_table(metrics, run)
    Log final CL metrics as a single-row W&B Table.

write_run_manifest(run_dir, run, cfg, git_commit, git_dirty,
                   final_metrics, task_timings, ...)
    Write ``run_manifest.json`` — the traceability anchor from any number
    in the paper back to the exact code + config + seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Module-level store for run start timestamps.
# Keyed by id(run) to avoid monkey-patching the run object.
# ---------------------------------------------------------------------------
_run_started_at: Dict[int, str] = {}


def _flatten_dict(d: dict, prefix: str = "", sep: str = "/") -> Dict[str, Any]:
    """Recursively flatten a nested dict into dot/slash-separated top-level keys.

    Example::

        _flatten_dict({"ncl": {"damping": 0.3}}, prefix="hparam")
        # → {"hparam/ncl/damping": 0.3}
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Null run — no-op stand-in for wandb.Run in csv_only mode
# ---------------------------------------------------------------------------

class _NullRun:
    """A no-op stand-in for a ``wandb.Run``.

    All methods silently do nothing.  Attribute access for ``id``, ``url``,
    and ``name`` returns ``None`` / ``"csv_only"`` so callers can safely read
    them without special-casing.
    """

    id: Optional[str] = None
    url: Optional[str] = None
    name: str = "csv_only"

    def log(self, metrics: dict, step: Optional[int] = None, **kwargs: Any) -> None:
        """No-op."""

    def log_artifact(self, artifact: Any, **kwargs: Any) -> None:
        """No-op."""

    def finish(self, **kwargs: Any) -> None:
        """No-op."""


# ---------------------------------------------------------------------------
# 4.1 init_tracking
# ---------------------------------------------------------------------------

def init_tracking(cfg: Any) -> tuple:
    """Initialise tracking for a training run.

    Captures the git state once at startup, constructs a W&B run (or a
    no-op null run in ``csv_only`` mode), and records the wall-clock start
    time for later use in :func:`write_run_manifest`.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The resolved Hydra config.  ``cfg.tracking.backend`` controls
        whether a real W&B run is created (``"wandb"``) or a null run
        (``"csv_only"``).

    Returns
    -------
    run : wandb.Run | _NullRun
        Live W&B run (or no-op null run).
    git_commit : str
        Full SHA of the current HEAD commit, or ``"unknown"`` if unavailable.
    git_dirty : bool
        ``True`` if the working tree has uncommitted changes.
    """
    # ── Git state ────────────────────────────────────────────────────────
    try:
        git_commit: str = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    try:
        git_dirty: bool = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_dirty = False

    started_at: str = datetime.now(timezone.utc).isoformat()

    # ── Build run ────────────────────────────────────────────────────────
    backend: str = str(cfg.tracking.backend)

    if backend == "csv_only":
        run: Any = _NullRun()
    else:
        # Deferred import so csv_only mode has zero W&B dependency.
        import wandb  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        ablation_key = getattr(cfg, "ablation_key", None)
        ablation_value = getattr(cfg, "ablation_value", None)

        group = ablation_key if ablation_key else "baseline"

        ablation_infix = f"_{ablation_key}_{ablation_value}" if ablation_key else ""
        run = wandb.init(
            project=cfg.tracking.wandb.project,
            entity=cfg.tracking.wandb.entity,
            group=group,
            job_type="ablation" if ablation_key else "baseline",
            name=(
                f"{cfg.method.name}"
                f"_{cfg.dataset.name}"
                f"{ablation_infix}"
                f"_s{cfg.seed}"
            ),
            tags=[
                cfg.method.name,
                cfg.dataset.name,
                f"seed_{cfg.seed}",
            ] + list(cfg.tracking.wandb.tags or []),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        # Also write every method hyperparameter as a flat top-level config
        # key so they're easily findable in the W&B UI (e.g. searching
        # "damping" finds "hparam/ncl/damping" directly).
        method_name = str(cfg.method.name)
        flat_hparams: Dict[str, Any] = {}
        # Method-specific sub-config (e.g. cfg.method.ncl, cfg.method.gem …)
        if hasattr(cfg.method, method_name):
            method_sub = OmegaConf.to_container(
                cfg.method[method_name], resolve=True
            )
            if isinstance(method_sub, dict):
                flat_hparams.update(
                    _flatten_dict(method_sub, prefix=f"hparam/{method_name}")
                )
        # Always surface top-level training knobs at the same depth.
        if hasattr(cfg, "training"):
            training_sub = OmegaConf.to_container(cfg.training, resolve=True)
            if isinstance(training_sub, dict):
                flat_hparams.update(
                    _flatten_dict(training_sub, prefix="hparam/training")
                )
        if flat_hparams:
            run.config.update(flat_hparams, allow_val_change=True)

    # Store start timestamp — retrieved by write_run_manifest via id(run)
    _run_started_at[id(run)] = started_at

    return run, git_commit, git_dirty


# ---------------------------------------------------------------------------
# 4.2 DualLogger
# ---------------------------------------------------------------------------

class DualLogger:
    """Write every metric to both a local CSV and a W&B run.

    The CSV header is fixed to the keys of the *first* :meth:`log` call.
    Subsequent rows with extra keys are silently dropped; missing keys
    produce empty cells.  The file is flushed after every row (crash-safe).

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the output CSV file.  Parent directories
        are created on the first :meth:`log` call.
    wandb_run : wandb.Run | _NullRun
        Live W&B run (or no-op null run).  Receives ``run.log(metrics, step=step)``.

    Usage::

        logger = DualLogger("results/task_curves/task_00_train.csv", run)
        for step, (x, y) in enumerate(train_loader):
            loss = method.observe(x, y, task_id).loss
            logger.log(step, {"loss": loss, "eta": eta})
        logger.close()
    """

    def __init__(self, csv_path: Optional[str], wandb_run: Any) -> None:
        self.csv_path: Optional[str] = csv_path  # None → skip local CSV
        self.wandb_run: Any = wandb_run
        self._csv_file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None

    def log(self, step: int, metrics_dict: Dict[str, Any]) -> None:
        """Log one row.  First call creates the CSV file and writes the header.

        Parameters
        ----------
        step : int
            Global training step (written as the ``step`` column).
        metrics_dict : dict
            Metric name → scalar value.  All values should be JSON-serialisable.
        """
        # W&B (no-op when wandb_run is a _NullRun)
        self.wandb_run.log(metrics_dict, step=step)

        # Local CSV — skipped when csv_path is None (outputs.save_task_curves=false)
        if self.csv_path is None:
            return

        if self._writer is None:
            # Ensure parent directory exists (handles both absolute and relative paths)
            dirpath = os.path.dirname(os.path.abspath(self.csv_path))
            os.makedirs(dirpath, exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")
            fieldnames = ["step"] + sorted(metrics_dict.keys())
            self._writer = csv.DictWriter(
                self._csv_file,
                fieldnames=fieldnames,
                extrasaction="ignore",  # silently drop keys not in header
            )
            self._writer.writeheader()

        self._writer.writerow({"step": step, **metrics_dict})
        self._csv_file.flush()  # crash-safe: every row is persisted immediately

    def close(self) -> None:
        """Close the underlying CSV file.  Idempotent — safe to call multiple times."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

    def __enter__(self) -> "DualLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 4.3 save_and_log_checkpoint
# ---------------------------------------------------------------------------

def save_and_log_checkpoint(
    model: torch.nn.Module,
    task_id: int,
    run: Any,
    cfg: Any,
    local_dir: str,
) -> str:
    """Save a model checkpoint locally; optionally upload as a W&B Artifact.

    Parameters
    ----------
    model : nn.Module
        The model whose ``state_dict()`` is saved.
    task_id : int
        Zero-based task index — determines the filename.
    run : wandb.Run | _NullRun
        Active W&B run (or no-op null run).
    cfg : omegaconf.DictConfig
        Full Hydra config.
    local_dir : str
        Directory in which to write ``model_task_XX.pt``.

    Returns
    -------
    str
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, f"model_task_{task_id:02d}.pt")
    torch.save(model.state_dict(), path)

    backend: str = str(cfg.tracking.backend)
    if backend != "csv_only" and cfg.tracking.wandb.artifacts.log_checkpoints:
        import wandb  # noqa: PLC0415

        artifact = wandb.Artifact(
            name=f"{run.id}_model_task_{task_id:02d}",
            type=cfg.tracking.wandb.artifacts.checkpoint_type,
            metadata={
                "task_id": task_id,
                "seed": cfg.seed,
                "method": cfg.method.name,
                "dataset": cfg.dataset.name,
            },
        )
        artifact.add_file(path)
        run.log_artifact(artifact)

    return path


# ---------------------------------------------------------------------------
# 4.4 save_and_log_results
# ---------------------------------------------------------------------------

def save_and_log_results(results_dir: str, run: Any, cfg: Any) -> None:
    """Upload the entire ``results/`` directory as a W&B Artifact at run end.

    In ``csv_only`` mode this is a no-op — local files are written by the
    training loop regardless.

    Parameters
    ----------
    results_dir : str
        Path to the directory to upload.
    run : wandb.Run | _NullRun
        Active W&B run (or no-op null run).
    cfg : omegaconf.DictConfig
        Full Hydra config.
    """
    backend: str = str(cfg.tracking.backend)
    if backend == "csv_only" or not cfg.tracking.wandb.artifacts.log_results:
        return

    import wandb  # noqa: PLC0415

    # Try to load final metrics from local file for artifact metadata.
    metrics_path = os.path.join(results_dir, "metrics_summary.json")
    try:
        with open(metrics_path) as fh:
            final_metrics: dict = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        final_metrics = {}

    artifact = wandb.Artifact(
        name=f"{run.id}_results",
        type=cfg.tracking.wandb.artifacts.results_type,
        metadata={
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "seed": cfg.seed,
            "final_metrics": final_metrics,
        },
    )
    artifact.add_dir(results_dir)
    run.log_artifact(artifact)


# ---------------------------------------------------------------------------
# 4.5 log_accuracy_matrix_table
# ---------------------------------------------------------------------------

def log_accuracy_matrix_table(
    accuracy_matrix: Any,
    run: Any,
    num_tasks: int,
) -> None:
    """Upload R[i,j] as a W&B Table for dashboard heatmap comparison.

    In ``csv_only`` mode (``_NullRun``) this is a no-op.

    Parameters
    ----------
    accuracy_matrix : array-like, shape (num_tasks, num_tasks)
        ``R[i, j]`` = accuracy on task j evaluated after training on task i.
    run : wandb.Run | _NullRun
    num_tasks : int
        Number of tasks (determines column count).
    """
    if isinstance(run, _NullRun):
        return

    import wandb  # noqa: PLC0415

    columns = [f"task_{j}" for j in range(num_tasks)]
    table = wandb.Table(columns=["eval_after_task"] + columns)
    for i in range(num_tasks):
        row = accuracy_matrix[i]
        row_list = row.tolist() if hasattr(row, "tolist") else list(row)
        table.add_data(f"task_{i}", *row_list)
    run.log({"accuracy_matrix_table": table})


# ---------------------------------------------------------------------------
# 4.6 log_metrics_summary_table
# ---------------------------------------------------------------------------

def log_metrics_summary_table(metrics: Dict[str, Any], run: Any) -> None:
    """Log final CL metrics as a single-row W&B Table.

    W&B can then show a unified table across all runs in a group for direct
    comparison.

    In ``csv_only`` mode (``_NullRun``) this is a no-op.

    Parameters
    ----------
    metrics : dict
        Must contain keys: ``ACC``, ``FORG``, ``min_ACC``, ``WF10``, ``WF100``,
        ``WP10``, ``WP100``, ``WC_ACC``, ``stability_gap_max_drop``,
        ``stability_gap_depth``, ``stability_gap_area``,
        ``stability_gap_area_end``, ``stability_gap_recovery_steps``.
        Missing keys produce ``None`` entries.
    run : wandb.Run | _NullRun
    """
    if isinstance(run, _NullRun):
        return

    import wandb  # noqa: PLC0415

    table = wandb.Table(
        columns=[
            "ACC", "FORG", "min_ACC", "WF10", "WF100", "WP10", "WP100", "WC_ACC",
            "stab_gap_max_drop", "stab_gap_depth", "stab_gap_area",
            "stab_gap_area_end", "stab_gap_recovery_steps",
        ],
        data=[[
            metrics.get("ACC"),
            metrics.get("FORG"),
            metrics.get("min_ACC"),
            metrics.get("WF10"),
            metrics.get("WF100"),
            metrics.get("WP10"),
            metrics.get("WP100"),
            metrics.get("WC_ACC"),
            metrics.get("stability_gap_max_drop"),
            metrics.get("stability_gap_depth"),
            metrics.get("stability_gap_area"),
            metrics.get("stability_gap_area_end"),
            metrics.get("stability_gap_recovery_steps"),
        ]],
    )
    run.log({"metrics_summary_table": table})


# ---------------------------------------------------------------------------
# write_run_manifest
# ---------------------------------------------------------------------------

def write_run_manifest(
    run_dir: str,
    run: Any,
    cfg: Any,
    git_commit: str,
    git_dirty: bool,
    final_metrics: Dict[str, Any],
    task_timings: List[Dict[str, Any]],
    wandb_artifact_ids: Optional[Dict[str, str]] = None,
    status: str = "completed",
) -> str:
    """Write ``run_manifest.json`` — the traceability anchor for the run.

    This file maps any number in the paper back to the exact code + config +
    seed + W&B run that produced it.

    Parameters
    ----------
    run_dir : str
        Directory where ``run_manifest.json`` is written (typically ``"."``
        when called from inside a Hydra job directory).
    run : wandb.Run | _NullRun
        Active W&B run for ``wandb_run_id`` / ``wandb_run_url`` fields.
    cfg : omegaconf.DictConfig
        Full resolved Hydra config.  Used for ``method``, ``dataset``,
        ``seed``, ``ablation_key``, ``ablation_value``, and ``config_hash``.
    git_commit : str
        Full SHA of HEAD as returned by :func:`init_tracking`.
    git_dirty : bool
        ``True`` if the working tree had uncommitted changes.
    final_metrics : dict
        Summary metrics (ACC, BWT, etc.) as returned by ``ContinualMetrics.to_dict()``.
    task_timings : list of dict
        Per-task timing records (each has ``"total_seconds"``).
    wandb_artifact_ids : dict, optional
        Map of artifact role → W&B artifact ID string.  Defaults to ``{}``.
    status : str, optional
        Run status — typically ``"completed"``.  Set to ``"failed"`` on error.

    Returns
    -------
    str
        Absolute path to the written ``run_manifest.json`` file.
    """
    if wandb_artifact_ids is None:
        wandb_artifact_ids = {}

    # ── Config hash (SHA-256 of serialised resolved config) ──────────────
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415
        config_str = OmegaConf.to_yaml(cfg)
    except Exception:
        # Fallback for non-OmegaConf configs (e.g., SimpleNamespace in tests)
        try:
            config_str = json.dumps(
                {k: v for k, v in vars(cfg).items()},
                default=str,
                sort_keys=True,
            )
        except Exception:
            config_str = repr(cfg)
    config_hash = "sha256:" + hashlib.sha256(config_str.encode()).hexdigest()

    # ── Hydra overrides (available only when running under Hydra) ─────────
    try:
        from hydra.core.hydra_config import HydraConfig  # noqa: PLC0415
        hydra_overrides: List[str] = list(HydraConfig.get().overrides.task)
    except Exception:
        hydra_overrides = []

    # ── Run IDs ───────────────────────────────────────────────────────────
    if not isinstance(run, _NullRun) and run.id:
        wandb_run_id: Optional[str] = run.id
        wandb_run_url: Optional[str] = run.url
        run_id: str = run.name or f"{cfg.method.name}_{cfg.dataset.name}_seed{cfg.seed}"
    else:
        wandb_run_id = None
        wandb_run_url = None
        run_id = (
            f"{cfg.method.name}_{cfg.dataset.name}"
            f"_seed{cfg.seed}"
            f"_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
        )

    # ── Timestamps ────────────────────────────────────────────────────────
    started_at: Optional[str] = _run_started_at.get(id(run))
    finished_at: str = datetime.now(timezone.utc).isoformat()

    # ── GPU info ──────────────────────────────────────────────────────────
    try:
        gpu: str = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        )
    except Exception:
        gpu = "unknown"

    # ── Checkpoint files ──────────────────────────────────────────────────
    # Glob for model_task_*.pt files in the configured checkpointing dir.
    ckpt_sub = getattr(getattr(cfg, "checkpointing", None), "dir", "checkpoints")
    ckpt_dir = os.path.join(run_dir, ckpt_sub)
    if os.path.isdir(ckpt_dir):
        checkpoint_files: List[str] = sorted(
            os.path.join(ckpt_sub, f)
            for f in os.listdir(ckpt_dir)
            if f.startswith("model_task_") and f.endswith(".pt")
        )
    else:
        checkpoint_files = []

    # ── Ablation fields (optional; may not exist on the config) ───────────
    ablation_key: Optional[str] = getattr(cfg, "ablation_key", None)
    ablation_value: Optional[Any] = getattr(cfg, "ablation_value", None)

    # ── Assemble manifest ─────────────────────────────────────────────────
    wall_clock_total = float(
        sum(t.get("total_seconds", 0.0) for t in task_timings)
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "wandb_run_id": wandb_run_id,
        "wandb_run_url": wandb_run_url,
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "seed": int(cfg.seed),
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty),
        "config_hash": config_hash,
        "hydra_overrides": hydra_overrides,
        "ablation_key": ablation_key,
        "ablation_value": ablation_value,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "final_metrics": final_metrics,
        "wall_clock_total_seconds": wall_clock_total,
        "gpu": gpu,
        "pytorch_version": torch.__version__,
        "checkpoint_files": checkpoint_files,
        "wandb_artifact_ids": wandb_artifact_ids,
    }

    # ── Write JSON ────────────────────────────────────────────────────────
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "run_manifest.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    return out_path
