#!/usr/bin/env python3
"""Quick-start example for mefai-risk-ai.

This script:
1. Fetches live Binance data for a small set of coins.
2. Builds a sliding-window dataset with proper train/val split.
3. Trains a Temporal Fusion Transformer for a few epochs.
4. Generates a risk report and portfolio allocation.

Usage::

    pip install -e .
    python examples/quick_start.py
"""

from __future__ import annotations

import json
import logging

import torch

from mefai_risk.config import Settings
from mefai_risk.data.dataset import CryptoDataset, build_dataloaders
from mefai_risk.models.tft import TemporalFusionTransformer
from mefai_risk.portfolio.manager import PortfolioManager
from mefai_risk.training.trainer import RiskTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick_start")


def main() -> None:
    # Use a small coin set and few epochs for a fast demo
    cfg = Settings()
    cfg.COINS = ["BTC", "ETH", "SOL"]
    cfg.EPOCHS = 5
    cfg.BATCH_SIZE = 16
    cfg.WINDOW_SIZE = 200  # fewer candles to download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- Data -----------------------------------------------------------
    logger.info("Fetching data and engineering features ...")
    train_loader, val_loader, dataset = build_dataloaders(
        coins=cfg.COINS,
        cfg=cfg,
        window_size=64,
        stride=4,
    )
    n_feat = dataset.n_features
    logger.info("Features per time-step: %d", n_feat)
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # ---- Model ----------------------------------------------------------
    model = TemporalFusionTransformer(
        input_size=n_feat,
        hidden_size=64,
        num_layers=2,
        n_heads=4,
    )
    trainer = RiskTrainer(model, cfg=cfg, device=device)

    # ---- Train ----------------------------------------------------------
    logger.info("Training for %d epochs ...", cfg.EPOCHS)
    history = trainer.fit(train_loader, val_loader)

    # ---- Portfolio report -----------------------------------------------
    logger.info("Generating portfolio report ...")
    manager = PortfolioManager(trainer, dataset, cfg=cfg)
    report = manager.generate_report()

    print("\n" + "=" * 60)
    print("  RISK REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))

    # Show allocation summary
    print("\n" + "-" * 40)
    print("  ALLOCATION SUMMARY")
    print("-" * 40)
    for coin, info in sorted(
        report["portfolio_allocation"].items(),
        key=lambda x: x[1]["weight"],
        reverse=True,
    ):
        print(f"  {coin:>5s}  weight={info['weight']:.2%}  leverage={info['leverage']:.1f}x")
    print()


if __name__ == "__main__":
    main()
