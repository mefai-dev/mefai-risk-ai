"""
Global configuration for the mefai-risk-ai package.

Settings are read from environment variables where applicable.
No fake API keys are generated; public Binance endpoints need no auth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Settings:
    """Central settings object.  Override via environment variables."""

    # --- Assets -----------------------------------------------------------
    COINS: List[str] = field(
        default_factory=lambda: [
            "BTC", "ETH", "BNB", "SOL", "AVAX",
            "XRP", "DOGE", "LTC", "ADA", "DOT",
        ]
    )
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])

    # --- Binance ----------------------------------------------------------
    API_BASE: str = "https://api.binance.com/api/v3"

    # --- Model / training -------------------------------------------------
    WINDOW_SIZE: int = 256
    BATCH_SIZE: int = 32
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 3
    N_HEADS: int = 4
    EPOCHS: int = 500
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    EARLY_STOPPING_PATIENCE: int = 20

    # --- Feature schema ---------------------------------------------------
    # Fixed number of features produced by FeatureEngine per row.
    # This prevents dimension mismatches across timeframes.
    N_FEATURES: int = 25

    # --- Risk thresholds --------------------------------------------------
    RISK_THRESHOLDS: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.3, "medium": 0.6, "high": 0.8}
    )

    # --- Validation split -------------------------------------------------
    VAL_RATIO: float = 0.2  # last 20 % of time series for validation

    # --- Paths ------------------------------------------------------------
    MODEL_DIR: str = "checkpoints"

    def __post_init__(self) -> None:
        """Read overrides from environment variables."""
        coins_env = os.environ.get("MEFAI_COINS")
        if coins_env:
            self.COINS = [c.strip() for c in coins_env.split(",")]

        tf_env = os.environ.get("MEFAI_TIMEFRAMES")
        if tf_env:
            self.TIMEFRAMES = [t.strip() for t in tf_env.split(",")]

        self.API_BASE = os.environ.get("MEFAI_API_BASE", self.API_BASE)
        self.WINDOW_SIZE = int(os.environ.get("MEFAI_WINDOW_SIZE", self.WINDOW_SIZE))
        self.BATCH_SIZE = int(os.environ.get("MEFAI_BATCH_SIZE", self.BATCH_SIZE))
        self.EPOCHS = int(os.environ.get("MEFAI_EPOCHS", self.EPOCHS))
        self.MODEL_DIR = os.environ.get("MEFAI_MODEL_DIR", self.MODEL_DIR)

    # --- HTTP helpers -----------------------------------------------------
    def get_headers(self) -> Dict[str, str]:
        """Return HTTP headers for Binance requests.

        The public klines endpoint needs no API key.  If the user has set
        ``BINANCE_API_KEY`` we include it; otherwise we send a plain
        User-Agent header only.
        """
        headers: Dict[str, str] = {"User-Agent": "mefai-risk-ai/0.1"}
        api_key: Optional[str] = os.environ.get("BINANCE_API_KEY")
        if api_key:
            headers["X-MBX-APIKEY"] = api_key
        return headers


# Module-level convenience instance
settings = Settings()
