"""RiskTrainer: AdamW optimiser, Huber loss, early stopping, checkpoint saving.

Bug fix vs. original RISKAI.txt
-------------------------------
* ``PortfolioManager.analyze_risk`` called ``self.model.evaluate(...)`` which
  does not exist on ``nn.Module``.  Now ``RiskTrainer.evaluate`` is the
  single evaluation entry-point and ``PortfolioManager`` calls the trainer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from mefai_risk.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)


class RiskTrainer:
    """Training loop with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
        A :class:`TemporalFusionTransformer` (or compatible).
    cfg : Settings, optional
    device : str, optional
        ``"cuda"`` or ``"cpu"``; auto-detected if *None*.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[Settings] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or _default_settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            weight_decay=self.cfg.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5,
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.  Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            risk_score, _volatility = self.model(features)
            loss = self.loss_fn(risk_score.squeeze(-1), labels.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate on a dataloader.  Returns dict with loss, spearman, predictions."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: List[float] = []
        all_labels: List[float] = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                risk_score, _volatility = self.model(features)
                loss = self.loss_fn(risk_score.squeeze(-1), labels.squeeze(-1))

                total_loss += loss.item()
                n_batches += 1

                all_preds.extend(risk_score.squeeze(-1).cpu().tolist())
                all_labels.extend(labels.squeeze(-1).cpu().tolist())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        # Spearman rank correlation
        if len(preds_arr) > 2 and np.std(preds_arr) > 0 and np.std(labels_arr) > 0:
            sp_corr = float(spearmanr(preds_arr, labels_arr).correlation)
        else:
            sp_corr = 0.0

        return {
            "loss": total_loss / max(n_batches, 1),
            "spearman": sp_corr,
            "predictions": preds_arr,
            "true_labels": labels_arr,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """Full training loop with early stopping.

        Returns a history dict with per-epoch metrics.
        """
        epochs = epochs or self.cfg.EPOCHS
        patience = self.cfg.EARLY_STOPPING_PATIENCE

        os.makedirs(self.cfg.MODEL_DIR, exist_ok=True)
        best_path = os.path.join(self.cfg.MODEL_DIR, "best_model.pth")

        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_spearman": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_spearman"].append(val_metrics["spearman"])

            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  val_spearman=%.4f",
                epoch, epochs, train_loss, val_loss, val_metrics["spearman"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                logger.info("  -> saved best model (val_loss=%.6f)", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Reload best weights
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
            logger.info("Loaded best model from %s", best_path)

        return history
