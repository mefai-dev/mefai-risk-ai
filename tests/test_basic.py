"""Smoke tests for mefai-risk-ai.

These tests run without network access by using synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mefai_risk.config import Settings
from mefai_risk.data.features import FeatureEngine
from mefai_risk.data.onchain import MockOnChainProvider
from mefai_risk.models.attention import MultiHeadAttention
from mefai_risk.models.tft import TemporalFusionTransformer
from mefai_risk.training.trainer import RiskTrainer


# ---- helpers --------------------------------------------------------

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.abs(close) + 1.0  # ensure positive
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    low = np.maximum(low, 0.01)
    opn = (high + low) / 2
    volume = rng.uniform(100, 10_000, n)
    df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": volume})
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


# ---- tests ----------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self) -> None:
        layer = MultiHeadAttention(input_dim=16, n_heads=4)
        x = torch.randn(2, 10, 16)
        out = layer(x)
        assert out.shape == (2, 10, 16)

    def test_invalid_dims(self) -> None:
        with pytest.raises(ValueError):
            MultiHeadAttention(input_dim=15, n_heads=4)


class TestTFT:
    def test_forward_shape(self) -> None:
        model = TemporalFusionTransformer(input_size=25, hidden_size=32, num_layers=1, n_heads=4)
        x = torch.randn(4, 10, 25)
        risk, vol = model(x)
        assert risk.shape == (4, 1)
        assert vol.shape == (4, 1)
        assert (risk >= 0).all() and (risk <= 1).all(), "risk_head should output [0,1]"
        assert (vol >= 0).all(), "volatility_head should output >= 0"


class TestFeatureEngine:
    def test_fixed_width(self) -> None:
        cfg = Settings()
        engine = FeatureEngine(cfg)
        df = _make_ohlcv(300)
        features_df = engine.compute_timeframe_features(df)
        assert features_df.shape[1] == cfg.N_FEATURES
        assert not features_df.isna().any().any()

    def test_build_feature_matrix(self) -> None:
        cfg = Settings()
        cfg.TIMEFRAMES = ["1h"]
        engine = FeatureEngine(cfg)
        df = _make_ohlcv(300)
        onchain = MockOnChainProvider().fetch("BTC")
        features, labels = engine.build_feature_matrix({"1h": df}, onchain, target_len=100)
        expected_width = cfg.N_FEATURES * len(cfg.TIMEFRAMES) + 8
        assert features.shape == (100, expected_width)
        assert labels.shape == (100,)


class TestMockOnChainProvider:
    def test_returns_all_keys(self) -> None:
        provider = MockOnChainProvider()
        data = provider.fetch("ETH")
        expected_keys = {
            "nvt", "velocity", "sopr", "exchange_netflow",
            "hashrate", "active_addresses", "whale_transactions", "fees_usd",
        }
        assert set(data.keys()) == expected_keys

    def test_warns(self) -> None:
        MockOnChainProvider._warned = False  # reset
        with pytest.warns(UserWarning, match="SYNTHETIC"):
            MockOnChainProvider().fetch("BTC")


class TestRiskTrainer:
    def test_train_and_evaluate(self) -> None:
        """Ensure one train epoch + evaluate runs without error."""
        n_feat = 25
        model = TemporalFusionTransformer(input_size=n_feat, hidden_size=16, num_layers=1, n_heads=4)
        cfg = Settings()
        trainer = RiskTrainer(model, cfg=cfg, device="cpu")

        # Fake dataloader
        data = [
            {"features": torch.randn(10, n_feat), "label": torch.tensor([0.5])}
            for _ in range(16)
        ]
        loader = torch.utils.data.DataLoader(data, batch_size=4)

        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float) and loss >= 0

        metrics = trainer.evaluate(loader)
        assert "loss" in metrics and "spearman" in metrics
