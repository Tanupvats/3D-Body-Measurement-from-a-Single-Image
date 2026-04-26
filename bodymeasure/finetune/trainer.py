

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import FineTuneConfig
from ..utils.logging import get_logger
from .corrector import FEATURE_DIM, NUM_BETAS, _build_mlp

logger = get_logger(__name__)


def compute_jacobian(body_model, betas, measurement_fn, eps: float = 0.01):
    """Numerical Jacobian d_measurements / d_betas at the given point.

    body_model: passed for API consistency but not used — measurement_fn
                is expected to close over its own body model so this
                function stays framework-agnostic.

    measurement_fn: callable that takes a SMPLXParams-like object with
                    .betas/.body_pose/.global_orient/.transl and returns
                    a dict[name -> value_cm].

    Returns: (J, baseline_dict) where J is dict[name -> np.ndarray(NUM_BETAS)].
    """
    from ..core.types import SMPLXParams

    base_params = SMPLXParams(
        betas=betas.reshape(1, -1).astype(np.float32),
        body_pose=np.zeros((1, 63), dtype=np.float32),
        global_orient=np.zeros((1, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    baseline = measurement_fn(base_params)
    J = {name: np.zeros(NUM_BETAS, dtype=np.float32) for name in baseline}
    for k in range(NUM_BETAS):
        b_plus = betas.copy()
        b_plus[k] += eps
        params_p = SMPLXParams(
            betas=b_plus.reshape(1, -1).astype(np.float32),
            body_pose=np.zeros((1, 63), dtype=np.float32),
            global_orient=np.zeros((1, 3), dtype=np.float32),
            transl=np.zeros((1, 3), dtype=np.float32),
        )
        m_plus = measurement_fn(params_p)
        for name in J:
            v_b = baseline.get(name, np.nan)
            v_p = m_plus.get(name, np.nan)
            if np.isfinite(v_b) and np.isfinite(v_p):
                J[name][k] = (v_p - v_b) / eps
    return J, baseline


def train_corrector(
    features: np.ndarray,           # (N, FEATURE_DIM)
    init_betas: np.ndarray,         # (N, NUM_BETAS)
    jacobians: list[dict],          # length N; each: name -> (NUM_BETAS,)
    baselines: list[dict],          # length N; each: name -> value_cm
    ground_truth: list[dict],       # length N; each: name -> value_cm
    cfg: FineTuneConfig,
    output_dir: Path,
) -> dict:
    """Train the shape corrector and write a checkpoint + report."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError("torch required for fine-tuning") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    N = features.shape[0]
    if N < 4:
        raise ValueError(f"need at least 4 subjects to train; got {N}")

    # Build per-sample (J, baseline, gt) tensors aligned by measurement names.
    # We use the union of measurements across samples; missing entries are
    # masked out of the loss.
    all_names = sorted({n for gt in ground_truth for n in gt})
    M = len(all_names)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    logger.info(f"training on {N} subjects, {M} measurement columns: {all_names}")

    J_arr = np.zeros((N, M, NUM_BETAS), dtype=np.float32)
    base_arr = np.full((N, M), np.nan, dtype=np.float32)
    gt_arr = np.full((N, M), np.nan, dtype=np.float32)
    for i in range(N):
        for name, vec in jacobians[i].items():
            j = name_to_idx[name]
            J_arr[i, j] = vec
        for name, val in baselines[i].items():
            j = name_to_idx[name]
            base_arr[i, j] = val
        for name, val in ground_truth[i].items():
            j = name_to_idx[name]
            gt_arr[i, j] = val
    mask = (~np.isnan(gt_arr)) & (~np.isnan(base_arr))

    # Per-measurement loss weighting: equal across measurements by default.
    # Could weight by inverse of population variance later.
    loss_weights = np.ones(M, dtype=np.float32)

    # Train/val split
    perm = rng.permutation(N)
    n_val = max(1, int(N * cfg.val_split))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_mlp().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    feat_t = torch.tensor(features, dtype=torch.float32, device=device)
    J_t = torch.tensor(J_arr, dtype=torch.float32, device=device)
    base_t = torch.tensor(np.nan_to_num(base_arr), dtype=torch.float32, device=device)
    gt_t = torch.tensor(np.nan_to_num(gt_arr), dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask.astype(np.float32), dtype=torch.float32, device=device)
    w_t = torch.tensor(loss_weights, dtype=torch.float32, device=device)

    history = {"epoch": [], "train_l1": [], "val_l1": [], "lr": []}
    best_val = float("inf")
    best_state = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        rng.shuffle(tr_idx)
        for start in range(0, len(tr_idx), cfg.batch_size):
            batch = tr_idx[start:start + cfg.batch_size]
            x = feat_t[batch]
            J_b = J_t[batch]
            base_b = base_t[batch]
            gt_b = gt_t[batch]
            m_b = mask_t[batch]

            delta = model(x)  # (B, NUM_BETAS)
            # measurement_after = baseline + J @ delta
            pred = base_b + (J_b @ delta.unsqueeze(-1)).squeeze(-1)
            err = torch.abs(pred - gt_b) * m_b * w_t  # (B, M)
            n_valid = m_b.sum().clamp(min=1.0)
            meas_loss = err.sum() / n_valid

            beta_reg = (delta ** 2).sum(dim=-1).mean()
            loss = cfg.measurement_loss_weight * meas_loss + cfg.beta_reg_weight * beta_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            x_val = feat_t[val_idx]
            J_v = J_t[val_idx]
            base_v = base_t[val_idx]
            gt_v = gt_t[val_idx]
            m_v = mask_t[val_idx]
            delta_v = model(x_val)
            pred_v = base_v + (J_v @ delta_v.unsqueeze(-1)).squeeze(-1)
            val_err = (torch.abs(pred_v - gt_v) * m_v).sum() / m_v.sum().clamp(min=1.0)
            val_l1 = float(val_err.cpu())

            x_tr = feat_t[tr_idx]
            J_tr = J_t[tr_idx]
            base_tr = base_t[tr_idx]
            gt_tr = gt_t[tr_idx]
            m_tr = mask_t[tr_idx]
            delta_tr = model(x_tr)
            pred_tr = base_tr + (J_tr @ delta_tr.unsqueeze(-1)).squeeze(-1)
            tr_err = (torch.abs(pred_tr - gt_tr) * m_tr).sum() / m_tr.sum().clamp(min=1.0)
            tr_l1 = float(tr_err.cpu())

        history["epoch"].append(epoch)
        history["train_l1"].append(tr_l1)
        history["val_l1"].append(val_l1)
        history["lr"].append(cfg.lr)

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.info(f"epoch {epoch:4d} | train_l1 {tr_l1:.3f}cm | val_l1 {val_l1:.3f}cm")

        if val_l1 < best_val - 1e-4:
            best_val = val_l1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                logger.info(f"early stop at epoch {epoch} (best val {best_val:.3f})")
                break

    # Save best checkpoint
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ckpt_path = output_dir / "corrector.pt"
    torch.save({"state_dict": best_state, "feature_dim": FEATURE_DIM, "num_betas": NUM_BETAS,
                "measurement_columns": all_names}, ckpt_path)
    logger.info(f"saved corrector checkpoint -> {ckpt_path}")

    # Per-measurement before/after MAE on val set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_val = feat_t[val_idx]
        J_v = J_t[val_idx]
        base_v = base_t[val_idx]
        gt_v = gt_t[val_idx]
        m_v = mask_t[val_idx]
        delta_v = model(x_val)
        pred_v = base_v + (J_v @ delta_v.unsqueeze(-1)).squeeze(-1)

    per_meas = {}
    for j, name in enumerate(all_names):
        m_col = m_v[:, j].cpu().numpy().astype(bool)
        if m_col.sum() == 0:
            continue
        gt_col = gt_v[:, j].cpu().numpy()[m_col]
        base_col = base_v[:, j].cpu().numpy()[m_col]
        pred_col = pred_v[:, j].cpu().numpy()[m_col]
        per_meas[name] = {
            "n": int(m_col.sum()),
            "before_mae_cm": float(np.mean(np.abs(base_col - gt_col))),
            "after_mae_cm":  float(np.mean(np.abs(pred_col - gt_col))),
        }

    report = {
        "n_subjects_total": int(N),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "feature_dim": FEATURE_DIM,
        "num_betas": NUM_BETAS,
        "measurement_columns": all_names,
        "best_val_l1_cm": best_val,
        "history": history,
        "per_measurement_val_mae": per_meas,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"wrote report -> {report_path}")
    logger.info(f"summary: best val L1 = {best_val:.3f}cm")
    for name, stats in per_meas.items():
        delta = stats["after_mae_cm"] - stats["before_mae_cm"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"  {name:30s} n={stats['n']:3d}  "
                    f"before={stats['before_mae_cm']:.2f}cm  "
                    f"after={stats['after_mae_cm']:.2f}cm  "
                    f"({sign}{delta:.2f}cm)")
    return report
