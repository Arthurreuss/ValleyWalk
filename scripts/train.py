"""Main training entry point (Hydra app).

Implements the full CACL training loop (T6.2).  The script is method-agnostic:
it builds the model, dataset, and method from the resolved Hydra config, then
drives the continual-learning loop:

  For each task:
    1. Run the inner batch loop — observe(), log loss+diagnostics, periodic eval.
    2. Track the stability gap (per-step accuracy drop on previously seen tasks).
    3. Call method.end_task() for post-task bookkeeping.
    4. Evaluate all seen tasks to populate the accuracy matrix.
    5. Save model checkpoint (and optionally buffer state).

  After all tasks:
    - Save the accuracy matrix (.npy) and upload to W&B as a Table.
    - Write metrics_summary.json and upload to W&B.
    - Save per-task wall-clock timing CSV.
    - Upload results/ directory as a W&B Artifact.
    - Write run_manifest.json — the traceability anchor.
"""

from __future__ import annotations

import csv
import json
import os
import time
import warnings
from typing import Optional

# torchvision's CIFAR loader triggers this with NumPy >= 2.4 because the
# pickled dataset files contain dtype objects created with align=0 (an int).
# NumPy 2.4 tightened the check; the warning is harmless and unfixable
# without patching torchvision, so suppress it at the entry point.
warnings.filterwarnings(
    "ignore",
    message="dtype\\(\\): align should be passed as Python or NumPy boolean",
    category=DeprecationWarning,
)

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.data.continual_dataset import ContinualDataset
from src.eval.metrics import ContinualMetrics
from src.eval.stability_gap import GradientTracker, StabilityGapTracker
from src.methods.cacl import CACL
from src.methods.er import ER
from src.methods.gem import GEM
from src.methods.ncl import NCL
from src.models.mlp import MLP
from src.models.resnet import ResNet18
from src.utils.diagnostics import CACLDiagnosticsWriter
from src.utils.seeding import set_global_seed
from src.utils.tracking import (
    _NullRun,
    DualLogger,
    init_tracking,
    log_accuracy_matrix_table,
    log_metrics_summary_table,
    save_and_log_checkpoint,
    save_and_log_results,
    write_run_manifest,
)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_cfg: str) -> torch.device:
    """Resolve the ``device`` config field to a concrete ``torch.device``.

    ``"auto"`` selects the best available backend in priority order:
    MPS (Apple Silicon) > CUDA > CPU.

    Args:
        device_cfg: Value of ``cfg.device`` — ``"auto"``, ``"cpu"``,
                    ``"cuda"``, or ``"mps"``.

    Returns:
        A ``torch.device`` ready to use with ``.to(device)``.
    """
    spec = str(device_cfg).lower()
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Factory: model
# ---------------------------------------------------------------------------

def build_model(model_cfg):
    """Instantiate the model specified by ``model_cfg.name``.

    Supported names: ``"mlp"``, ``"resnet18"``.

    Args:
        model_cfg: Hydra model sub-config (``cfg.model``).

    Returns:
        nn.Module on CPU.  Caller is responsible for moving to the target
        device with ``model.to(device)`` before passing to ``build_method``.

    Raises:
        ValueError: If the model name is not recognised.
    """
    name = str(model_cfg.name)
    if name == "mlp":
        return MLP(model_cfg)
    if name == "resnet18":
        return ResNet18(model_cfg)
    raise ValueError(f"Unknown model '{name}'. Supported: mlp, resnet18")


# ---------------------------------------------------------------------------
# Diagnostic data sampling for g_true
# ---------------------------------------------------------------------------

def _load_full_past_task_data(dataset, task_id: int):
    """Concatenate every past task's full training set into a single (x, y).

    Used to compute ``g_true`` ≔ ∇_θ L_past(θ) at every step of task
    ``task_id`` for the buffer-fidelity diagnostics in
    ``ER.set_diagnostic_data()``.  Because the returned tensors contain
    *every* past-task training sample (no sampling, no subsetting), the
    resulting ``g_true`` is the deterministic empirical past-task gradient
    at the current parameters — zero sampling noise.  This is the
    reference against which the (stochastic) buffer-replay gradient
    ``g_replay`` is compared by ``cos(g_replay, g_true)`` and
    ``‖g_replay‖ / ‖g_true‖``.

    Args:
        dataset: The ``ContinualDataset`` instance.
        task_id: Index of the current task (must be > 0; otherwise no past
                 tasks exist and this should not be called).

    Returns:
        ``(x, y)`` — every past-task training sample concatenated, on CPU.
        For rot-MNIST 2-task with task_id == 1 this is the full 60 000-
        sample T₀ training set.  Caller is responsible for moving to
        device.
    """
    if task_id <= 0:
        raise ValueError("_load_full_past_task_data requires task_id > 0")

    xs, ys = [], []
    for j in range(task_id):
        # Fresh loader (independent of the buffer) iterates every sample.
        past_loader, _ = dataset._dataset.get_task_loaders(j)
        for x, y in past_loader:
            xs.append(x)
            ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# ---------------------------------------------------------------------------
# Factory: method
# ---------------------------------------------------------------------------

def build_method(cfg, model, buffer):
    """Instantiate the continual-learning method specified by ``cfg.method.name``.

    All method constructors receive the **full** ``cfg`` (not just
    ``cfg.method``) because they also read ``cfg.training`` for optimizer
    hyper-parameters.  The ``buffer`` argument is ignored for regularisation-
    only methods (NCL).

    Supported method names: ``"er"``, ``"gem"``, ``"agem"``,
    ``"ncl"``, ``"cacl"``.

    Args:
        cfg:    Full resolved Hydra config.
        model:  The nn.Module being trained.
        buffer: ReservoirBuffer shared between dataset and method.
                Passed to replay-based methods; ignored for NCL.

    Returns:
        BaseMethod subclass instance.

    Raises:
        ValueError: If the method name is not recognised.
    """
    name = str(cfg.method.name)
    if name == "er":
        return ER(model, cfg, buffer)
    if name in ("gem", "agem"):
        return GEM(model, cfg, buffer)
    if name == "ncl":
        # NCL is regularisation-only — no replay buffer.
        return NCL(model, cfg)
    if name == "cacl":
        return CACL(model, cfg, buffer)
    raise ValueError(
        f"Unknown method '{name}'. "
        f"Supported: er, gem, agem, ncl, cacl"
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Full continual-learning training loop — one run, one seed."""

    # ── Reproducibility & tracking ────────────────────────────────────────
    set_global_seed(cfg.seed)
    run, git_commit, git_dirty = init_tracking(cfg)

    # ── Device selection ──────────────────────────────────────────────────
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # ── Build model, dataset, method ──────────────────────────────────────
    model = build_model(cfg.model)
    model.to(device)
    dataset = ContinualDataset(
        cfg.dataset,
        cfg.memory,
        batch_size=cfg.training.batch_size,
    )
    # buffer is shared: dataset fills it after each task; method reads it.
    method = build_method(cfg, model, dataset.buffer)
    metrics = ContinualMetrics()

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs(cfg.checkpointing.dir, exist_ok=True)
    os.makedirs(cfg.outputs.dir, exist_ok=True)

    # ── CACL diagnostics writer (run-level, cross-task CSVs) ─────────────
    # Created only for CACL runs and only when save_diagnostics is enabled.
    # The writer stays open across all tasks so the step counter is continuous.
    diagnostics_writer: Optional[CACLDiagnosticsWriter] = (
        CACLDiagnosticsWriter(method, cfg.outputs.dir, run, cfg.tracking.local)
        if (isinstance(method, CACL) and cfg.outputs.save_diagnostics)
        else None
    )

    # ── Combined accuracy CSV (run-level, all tasks, all steps) ──────────
    # One file per run that records every task's accuracy at every eval step,
    # giving a single continuous curve spanning the full training run.
    _combined_acc_path = os.path.join(cfg.outputs.dir, "accuracy_curves.csv")
    _combined_acc_file = open(_combined_acc_path, "w", newline="")
    _combined_acc_fieldnames = ["step"] + [
        f"task_{j}_acc" for j in range(dataset.num_tasks)
    ]
    combined_acc_writer = csv.DictWriter(
        _combined_acc_file,
        fieldnames=_combined_acc_fieldnames,
        extrasaction="ignore",
    )
    combined_acc_writer.writeheader()
    _combined_acc_file.flush()

    # ── Task loop ─────────────────────────────────────────────────────────
    task_timings = []
    global_step = 0   # monotonically increasing across ALL tasks
    gap_tracker = None  # re-instantiated at the start of each task > 0

    for task_id, train_loader, all_test_loaders in dataset.task_iterator():
        task_start = time.time()

        # ── Per-task dual-write loggers ────────────────────────────────────
        # csv_path=None disables local CSV; W&B logging still happens via run.
        _curve_dir = f"{cfg.outputs.dir}/task_curves" if cfg.outputs.save_task_curves else None
        train_logger = DualLogger(
            f"{_curve_dir}/task_{task_id:02d}_train.csv" if _curve_dir else None,
            run,
        )
        # eval_logger uses _NullRun so it only writes the local CSV.
        # W&B receives eval metrics merged into the train_logger payload below,
        # ensuring exactly one wandb.log() call per step (W&B silently drops
        # any second log() call targeting the same step value).
        eval_logger = DualLogger(
            f"{_curve_dir}/task_{task_id:02d}_eval.csv" if _curve_dir else None,
            _NullRun(),
        )

        # ── Stability gap tracker ──────────────────────────────────────────
        # Only relevant from task 1 onward.  The tracker evaluates the model
        # on *previously* seen tasks at construction time to capture the
        # pre-task baseline; the current task has no baseline.
        # Read the g_true-diagnostics toggle (off by default) — present in
        # configs/config.yaml under eval.stability_gap.grad_diagnostics.
        # OmegaConf .get with a fallback keeps older configs working.
        _gd_cfg = cfg.eval.stability_gap.get("grad_diagnostics", None)
        _gd_enabled = bool(_gd_cfg.enabled) if _gd_cfg is not None else False

        if task_id > 0 and cfg.eval.stability_gap.enabled:
            prev_test_loaders = {j: all_test_loaders[j] for j in range(task_id)}
            gap_tracker = StabilityGapTracker(
                eval_freq_steps=cfg.eval.stability_gap.eval_freq_steps,
                model=model,
                test_loaders=prev_test_loaders,
            )
            grad_tracker = GradientTracker(max_steps=500) if _gd_enabled else None

            # Wire g_true diagnostics only when the toggle is on.  Computing
            # g_true requires an extra forward+backward per step on the full
            # past-task training set — off by default, enable for the
            # G-series.  Using the *full* past data (not a sub-sample) makes
            # g_true the exact empirical past-task gradient: cos(g_replay,
            # g_true) and ‖g_replay‖/‖g_true‖ then measure buffer fidelity
            # against a deterministic reference, not against another noisy
            # estimator.
            if _gd_enabled and hasattr(method, "set_diagnostic_data"):
                diag_x, diag_y = _load_full_past_task_data(dataset, task_id)
                method.set_diagnostic_data(diag_x, diag_y)
            elif hasattr(method, "set_diagnostic_data"):
                method.set_diagnostic_data(None, None)
        else:
            gap_tracker = None
            grad_tracker = None
            if hasattr(method, "set_diagnostic_data"):
                method.set_diagnostic_data(None, None)

        # ── Inner batch loop ───────────────────────────────────────────────
        task_step = 0   # steps elapsed within this task (resets each task)
        steps_per_task = len(train_loader) * cfg.training.epochs_per_task
        for _epoch in range(cfg.training.epochs_per_task):
            for _batch_idx, (x, y) in enumerate(train_loader):
                # One gradient step + optional method-specific diagnostics
                step_result = method.observe(x, y, task_id)
                diagnostics = method.get_step_diagnostics()

                # Dual-write: loss, method diagnostics, and gradient metrics.
                grad_ratio = (
                    method.get_last_grad_ratio()
                    if hasattr(method, "get_last_grad_ratio")
                    else None
                )
                # cos(g_replay, g_true) and ‖g_replay‖/‖g_true‖ are only
                # meaningful when g_true was actually computed — i.e. when
                # the grad_diagnostics toggle is on.  Otherwise the methods
                # return neutral defaults (0.0 / 1.0) which would pollute
                # the logs, so skip them entirely here.
                true_grad_cosine = (
                    method.get_last_true_grad_cosine()
                    if (_gd_enabled and hasattr(method, "get_last_true_grad_cosine"))
                    else None
                )
                true_grad_mag_ratio = (
                    method.get_last_true_grad_mag_ratio()
                    if (_gd_enabled and hasattr(method, "get_last_true_grad_mag_ratio"))
                    else None
                )
                log_payload = {"loss": step_result["loss"], **diagnostics}
                if "replay_loss" in step_result:
                    log_payload["replay_loss"] = step_result["replay_loss"]
                if grad_ratio is not None:
                    log_payload["grad_ratio"] = grad_ratio
                if true_grad_cosine is not None and task_id > 0:
                    log_payload["true_grad_cosine"] = true_grad_cosine
                if true_grad_mag_ratio is not None and task_id > 0:
                    log_payload["true_grad_mag_ratio"] = true_grad_mag_ratio
                if grad_tracker is not None and true_grad_cosine is not None:
                    grad_tracker.record(
                        task_step, true_grad_cosine, true_grad_mag_ratio,
                    )

                # CACL run-level diagnostics (eigenvalue spectrum, trust radius,
                # cone fallback) — written to continuous cross-task CSV files.
                if diagnostics_writer is not None:
                    diagnostics_writer.log_step(global_step, diagnostics)

                # Periodic evaluation — always evaluate all tasks seen so far
                # so that accuracy_curves.csv has a continuous view of every
                # task across the full training run (including the stability gap
                # period where old-task accuracy temporarily drops).
                if global_step % cfg.eval.eval_every_n_steps == 0:
                    eval_accs = {
                        f"task_{j}_acc": method.evaluate(all_test_loaders[j])
                        for j in range(task_id + 1)
                    }
                    # Merge eval metrics into the train log payload so that all
                    # metrics for this step go out in a single wandb.log() call.
                    # W&B commits a step on the first log() call with that step
                    # value; a second call with the same step is silently dropped,
                    # which would cause task accuracy curves to never appear.
                    log_payload.update(eval_accs)
                    eval_logger.log(global_step, eval_accs)  # writes local CSV only
                    combined_acc_writer.writerow({"step": global_step, **eval_accs})
                    _combined_acc_file.flush()
                    for j in range(task_id + 1):
                        metrics.record_step(j, global_step, eval_accs[f"task_{j}_acc"])

                train_logger.log(global_step, log_payload)

                # Fine-grained stability-gap eval cadence:
                #   • first window_steps steps of a new task (post-switch)
                #   • last pre_switch_steps steps of the current task (pre-switch)
                # Both windows use eval_freq_steps; everything else uses eval_every_n_steps.
                in_post_switch_window = task_step < cfg.eval.stability_gap.window_steps
                in_pre_switch_window  = task_step >= steps_per_task - cfg.eval.stability_gap.pre_switch_steps
                sg_freq = (
                    cfg.eval.stability_gap.eval_freq_steps
                    if (in_post_switch_window or in_pre_switch_window)
                    else cfg.eval.eval_every_n_steps
                )
                if (
                    gap_tracker is not None
                    and global_step % sg_freq == 0
                ):
                    sg_accs = gap_tracker.record(global_step, all_test_loaders)
                    # Persist the fine-grained sample to the SAME sinks the
                    # regular eval block writes to — per-task CSV, combined
                    # accuracy CSV, in-memory metrics tracker (which drives
                    # WF10/WF100/WP10/WP100/min_ACC), and W&B.  Without this
                    # the fine samples used to be live only on W&B and inside
                    # gap_tracker._records, leaving local plots blind to the
                    # dip and ContinualMetrics blind to the dense gap window.
                    #
                    # Skip on steps already handled by the regular eval block
                    # (a multiple of eval_every_n_steps) — those use the same
                    # model state on the same loaders so the values are
                    # identical, and re-writing would duplicate rows and (on
                    # W&B) be silently dropped anyway.
                    if global_step % cfg.eval.eval_every_n_steps != 0:
                        sg_payload = {
                            f"task_{j}_acc": v for j, v in sg_accs.items()
                        }
                        eval_logger.log(global_step, sg_payload)
                        combined_acc_writer.writerow(
                            {"step": global_step, **sg_payload}
                        )
                        _combined_acc_file.flush()
                        for j, acc in sg_accs.items():
                            metrics.record_step(j, global_step, acc)
                        if not isinstance(run, _NullRun):
                            import wandb  # noqa: PLC0415
                            wandb.log(sg_payload, step=global_step)

                global_step += 1
                task_step += 1

        # Close loggers promptly (flush CSVs)
        train_logger.close()
        eval_logger.close()

        # ── End-of-task bookkeeping ────────────────────────────────────────
        # Fills replay buffer, computes Fisher / reference gradients, etc.
        method.end_task(task_id, train_loader)
        task_train_time = time.time() - task_start

        # ── Full evaluation at task boundary → accuracy matrix ─────────────
        eval_start = time.time()
        metrics.notify_task_end(task_i=task_id, step=global_step)
        for j, test_loader_j in all_test_loaders.items():
            acc = method.evaluate(test_loader_j)
            metrics.record_step(task_j=j, step=global_step, accuracy=acc)
        task_eval_time = time.time() - eval_start

        task_timings.append({
            "task_id": task_id,
            "train_seconds": task_train_time,
            "eval_seconds": task_eval_time,
            "total_seconds": task_train_time + task_eval_time,
        })

        # ── Model + buffer checkpoint ──────────────────────────────────────
        is_last_task = (task_id == dataset.num_tasks - 1)
        should_checkpoint = (
            (task_id + 1) % cfg.checkpointing.save_every_n_tasks == 0
            or (is_last_task and cfg.checkpointing.save_final)
        )
        if should_checkpoint:
            save_and_log_checkpoint(model, task_id, run, cfg, cfg.checkpointing.dir)
            if cfg.checkpointing.save_buffer:
                torch.save(
                    dataset.buffer.state_dict(),
                    os.path.join(
                        cfg.checkpointing.dir, f"buffer_task_{task_id:02d}.pt"
                    ),
                )

    # ── End of all tasks ──────────────────────────────────────────────────

    # Close combined accuracy CSV
    _combined_acc_file.close()

    # Close CACL diagnostics writer (flushes all 3 run-level CSV files)
    if diagnostics_writer is not None:
        diagnostics_writer.close()

    # Save accuracy matrix as .npy
    R = metrics.get_accuracy_matrix()
    if cfg.outputs.save_accuracy_matrix:
        np.save(os.path.join(cfg.outputs.dir, "accuracy_matrix.npy"), R)
    log_accuracy_matrix_table(R, run, cfg.dataset.num_tasks)

    # Build final metrics dict.
    final = {
        **metrics.to_dict(),
        "stability_gap_max_drop": (
            gap_tracker.max_drop() if gap_tracker is not None else None
        ),
        "stability_gap_depth": (
            gap_tracker.gap_depth() if gap_tracker is not None else None
        ),
        "stability_gap_area": (
            gap_tracker.gap_area() if gap_tracker is not None else None
        ),
        "stability_gap_recovery_steps": (
            gap_tracker.recovery_steps() if gap_tracker is not None else None
        ),
        "true_grad_cosine_mean": (
            grad_tracker.mean_true_grad_cosine() if grad_tracker is not None else None
        ),
        "true_grad_cosine_min": (
            grad_tracker.min_true_grad_cosine() if grad_tracker is not None else None
        ),
        "true_grad_mag_ratio_mean": (
            grad_tracker.mean_true_grad_mag_ratio() if grad_tracker is not None else None
        ),
    }

    # Write local metrics summary JSON
    with open(os.path.join(cfg.outputs.dir, "metrics_summary.json"), "w") as fh:
        json.dump(final, fh, indent=2)
    log_metrics_summary_table(final, run)

    # W&B summary scalars (shows up in the leaderboard-style run comparison)
    if not isinstance(run, _NullRun):
        import wandb  # noqa: PLC0415
        for k, v in final.items():
            wandb.summary[k] = v

    # Save per-task wall-clock timing
    if cfg.tracking.local.log_wall_clock:
        timing_dir = os.path.join(cfg.outputs.dir, "timing")
        os.makedirs(timing_dir, exist_ok=True)
        pd.DataFrame(task_timings).to_csv(
            os.path.join(timing_dir, "wall_clock.csv"), index=False
        )
    if not isinstance(run, _NullRun):
        import wandb  # noqa: PLC0415
        wandb.summary["wall_clock_total"] = sum(
            t["total_seconds"] for t in task_timings
        )

    # Upload results bundle as W&B Artifact at run end
    save_and_log_results(cfg.outputs.dir, run, cfg)

    # Write run manifest — traceability anchor for every number in the paper
    write_run_manifest(
        run_dir=".",
        run=run,
        cfg=cfg,
        git_commit=git_commit,
        git_dirty=git_dirty,
        final_metrics=final,
        task_timings=task_timings,
    )

    # Finish the W&B run (no-op for _NullRun)
    run.finish()


if __name__ == "__main__":
    main()
