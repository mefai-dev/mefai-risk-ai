"""PortfolioManager: risk scoring and inverse-volatility allocation.

Bug fix vs. original RISKAI.txt
-------------------------------
* Original called ``self.model.evaluate(dataloader)`` — but ``evaluate``
  lives on :class:`RiskTrainer`, not on the PyTorch model.  The manager
  now accepts a ``trainer`` and delegates evaluation to it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from mefai_risk.config import Settings, settings as _default_settings
from mefai_risk.data.dataset import CryptoDataset
from mefai_risk.training.trainer import RiskTrainer

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Generate risk assessments and portfolio allocations.

    Parameters
    ----------
    trainer : RiskTrainer
        Trained (or at least initialised) trainer that owns the model.
    dataset : CryptoDataset
        Dataset used for inference.
    cfg : Settings, optional
    """

    def __init__(
        self,
        trainer: RiskTrainer,
        dataset: CryptoDataset,
        cfg: Optional[Settings] = None,
    ) -> None:
        self.trainer = trainer
        self.dataset = dataset
        self.cfg = cfg or _default_settings
        self.risk_assessment: Dict[str, Dict[str, Any]] = {}
        self.portfolio_allocation: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def analyze_risk(self) -> Dict[str, Dict[str, Any]]:
        """Run inference on the full dataset and build per-coin risk scores."""
        loader = DataLoader(self.dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=False)
        metrics = self.trainer.evaluate(loader)

        preds = metrics["predictions"]
        labels = metrics["true_labels"]

        # Aggregate predictions per coin (mean over windows)
        coin_preds: Dict[str, List[float]] = {}
        coin_labels: Dict[str, List[float]] = {}
        for idx, (coin, _start) in enumerate(self.dataset._index):
            if idx < len(preds):
                coin_preds.setdefault(coin, []).append(float(preds[idx]))
                coin_labels.setdefault(coin, []).append(float(labels[idx]))

        for coin in self.dataset.coins:
            if coin in coin_preds:
                avg_pred = float(np.mean(coin_preds[coin]))
                avg_vol = float(np.mean(coin_labels[coin]))
            else:
                avg_pred = 0.5
                avg_vol = 0.0

            self.risk_assessment[coin] = {
                "risk_score": avg_pred,
                "volatility": avg_vol,
                "risk_category": self._categorize_risk(avg_pred),
                "beta_to_btc": self.dataset.market_beta.get(coin, 1.0),
            }

        return self.risk_assessment

    # ------------------------------------------------------------------
    def _categorize_risk(self, score: float) -> str:
        thresholds = self.cfg.RISK_THRESHOLDS
        if score < thresholds["low"]:
            return "low"
        elif score < thresholds["high"]:
            return "medium"
        return "high"

    # ------------------------------------------------------------------
    def optimize_portfolio(self, risk_tolerance: str = "medium") -> Dict[str, Dict[str, float]]:
        """Compute inverse-volatility weighted allocation filtered by risk tolerance."""
        if not self.risk_assessment:
            self.analyze_risk()

        valid = list(self.risk_assessment.keys())

        if risk_tolerance == "low":
            selected = [c for c in valid if self.risk_assessment[c]["risk_category"] == "low"]
        elif risk_tolerance == "high":
            selected = valid
        else:
            selected = [c for c in valid if self.risk_assessment[c]["risk_category"] in ("low", "medium")]

        if not selected:
            selected = valid  # fallback

        vols = np.array([self.risk_assessment[c]["volatility"] for c in selected])
        inv_vol = 1.0 / (vols + 1e-6)
        weights = inv_vol / inv_vol.sum()

        risks = np.array([self.risk_assessment[c]["risk_score"] for c in selected])
        adj = 1.0 / (1.0 + risks)
        adj_weights = weights * adj
        adj_weights /= adj_weights.sum()

        self.portfolio_allocation = {
            coin: {
                "weight": float(w),
                "leverage": self._suggested_leverage(self.risk_assessment[coin]["risk_score"]),
            }
            for coin, w in zip(selected, adj_weights)
        }
        return self.portfolio_allocation

    @staticmethod
    def _suggested_leverage(risk_score: float) -> float:
        if risk_score < 0.3:
            return 3.0
        elif risk_score < 0.6:
            return 2.0
        return 1.0

    # ------------------------------------------------------------------
    def generate_report(self) -> Dict[str, Any]:
        """Full JSON-serialisable risk report."""
        if not self.risk_assessment:
            self.analyze_risk()
        if not self.portfolio_allocation:
            self.optimize_portfolio()

        avg_risk = float(np.mean([v["risk_score"] for v in self.risk_assessment.values()]))
        avg_vol = float(np.mean([v["volatility"] for v in self.risk_assessment.values()]))

        if avg_risk < 0.4 and avg_vol < 0.02:
            market_condition = "low_risk"
        elif avg_risk > 0.7 and avg_vol > 0.05:
            market_condition = "high_risk"
        else:
            market_condition = "neutral"

        corr_dict = (
            self.dataset.correlation_matrix.to_dict()
            if not self.dataset.correlation_matrix.empty
            else {}
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_condition": market_condition,
            "risk_assessment": self.risk_assessment,
            "portfolio_allocation": self.portfolio_allocation,
            "correlation_matrix": corr_dict,
            "market_betas": self.dataset.market_beta,
        }

    def save_report(self, path: str = "risk_report.json") -> None:
        report = self.generate_report()
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Report saved to %s", path)
