"""Fetch OHLCV candle data from the Binance public REST API."""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from mefai_risk.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)

_KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume"]


class BinanceFetcher:
    """Download public kline (candlestick) data from Binance.

    Parameters
    ----------
    cfg : Settings, optional
        Configuration instance; uses the module-level default if *None*.
    max_retries : int
        Number of retries on transient HTTP errors.
    """

    def __init__(self, cfg: Optional[Settings] = None, max_retries: int = 3) -> None:
        self.cfg = cfg or _default_settings
        self.max_retries = max_retries

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """Fetch kline data for *symbol* (e.g. ``"BTCUSDT"``).

        Returns a DataFrame with numeric OHLCV columns and a datetime
        index, or *None* on failure.
        """
        url = f"{self.cfg.API_BASE}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        headers = self.cfg.get_headers()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                return self._parse(resp.json())
            except requests.RequestException as exc:
                logger.warning(
                    "Attempt %d/%d fetching %s %s failed: %s",
                    attempt, self.max_retries, symbol, interval, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # exponential back-off
        logger.error("All %d attempts failed for %s %s", self.max_retries, symbol, interval)
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
        df[_NUMERIC] = df[_NUMERIC].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_coin(self, coin: str, interval: str = "1h", limit: int = 1000) -> Optional[pd.DataFrame]:
        """Convenience: fetch ``{coin}USDT`` klines."""
        return self.fetch_klines(f"{coin}USDT", interval=interval, limit=limit)
