"""Feature engineering with a fixed-width schema.

Every timeframe is projected to exactly ``N_FEATURES`` columns so that
concatenation across timeframes always yields a predictable shape.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import (
    OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice,
)

from mefai_risk.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Produce a fixed-width feature matrix from raw OHLCV data.

    Each timeframe produces exactly ``cfg.N_FEATURES`` columns.
    On-chain metrics are appended as 8 additional columns (constant per
    row — they are scalar snapshots, not time-series).

    Total width per sample:
        ``len(cfg.TIMEFRAMES) * cfg.N_FEATURES + 8``
    """

    # The 25 features we extract per timeframe (order matters):
    _TF_FEATURE_NAMES = [
        "open", "high", "low", "close", "volume",
        "returns", "log_returns", "volatility_12", "volatility_24",
        "rsi_14", "stochrsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "ema_12", "ema_26", "sma_50",
        "atr_14",
        "obv",
        "vwap",
        "volume_ratio",
    ]

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self.cfg = cfg or _default_settings
        assert len(self._TF_FEATURE_NAMES) == self.cfg.N_FEATURES, (
            f"Feature list length ({len(self._TF_FEATURE_NAMES)}) "
            f"!= N_FEATURES ({self.cfg.N_FEATURES})"
        )

    # ------------------------------------------------------------------
    def compute_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract exactly ``N_FEATURES`` columns from an OHLCV DataFrame.

        Missing / NaN rows are forward-filled and then back-filled so the
        output has no NaNs.
        """
        out = pd.DataFrame(index=df.index)

        out["open"] = df["open"]
        out["high"] = df["high"]
        out["low"] = df["low"]
        out["close"] = df["close"]
        out["volume"] = df["volume"]

        out["returns"] = df["close"].pct_change()
        out["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        out["volatility_12"] = df["close"].pct_change().rolling(12).std()
        out["volatility_24"] = df["close"].pct_change().rolling(24).std()

        rsi = RSIIndicator(close=df["close"], window=14)
        out["rsi_14"] = rsi.rsi()

        srsi = StochRSIIndicator(close=df["close"], window=14)
        out["stochrsi_14"] = srsi.stochrsi()

        macd = MACD(close=df["close"])
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_diff"] = macd.macd_diff()

        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        out["bb_upper"] = bb.bollinger_hband()
        out["bb_middle"] = bb.bollinger_mavg()
        out["bb_lower"] = bb.bollinger_lband()
        out["bb_width"] = bb.bollinger_wband()

        out["ema_12"] = EMAIndicator(close=df["close"], window=12).ema_indicator()
        out["ema_26"] = EMAIndicator(close=df["close"], window=26).ema_indicator()
        out["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()

        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        out["atr_14"] = atr.average_true_range()

        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        out["obv"] = obv.on_balance_volume()

        vwap = VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20,
        )
        out["vwap"] = vwap.volume_weighted_average_price()

        # Volume ratio (current volume / rolling mean)
        vol_ma = df["volume"].rolling(20).mean()
        out["volume_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)

        out = out.ffill().bfill().fillna(0.0)
        return out

    # ------------------------------------------------------------------
    def build_feature_matrix(
        self,
        timeframe_dfs: Dict[str, Optional[pd.DataFrame]],
        onchain: Dict[str, float],
        target_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine multi-timeframe features + on-chain into a 2-D array.

        Parameters
        ----------
        timeframe_dfs : dict
            ``{timeframe: DataFrame}`` from :class:`BinanceFetcher`.
        onchain : dict
            Scalar on-chain metrics (8 values).
        target_len : int
            Desired number of rows (time-steps).  Each timeframe is
            truncated / padded to this length.

        Returns
        -------
        features : ndarray of shape ``(target_len, total_features)``
        labels : ndarray of shape ``(target_len,)``
            Future-returns volatility used as a risk proxy.
        """
        tf_arrays = []
        for tf in self.cfg.TIMEFRAMES:
            df = timeframe_dfs.get(tf)
            if df is not None and len(df) >= 50:  # need enough rows for indicators
                feat_df = self.compute_timeframe_features(df)
                arr = feat_df.values[-target_len:]
            else:
                logger.warning("Timeframe %s missing or too short — filling zeros", tf)
                arr = np.zeros((target_len, self.cfg.N_FEATURES))

            # Pad if shorter than target_len
            if arr.shape[0] < target_len:
                pad = np.zeros((target_len - arr.shape[0], arr.shape[1]))
                arr = np.vstack([pad, arr])

            tf_arrays.append(arr)

        price_features = np.concatenate(tf_arrays, axis=1)

        # On-chain features (broadcast scalar snapshot across time)
        oc_vals = np.array(list(onchain.values()), dtype=np.float64)
        onchain_features = np.tile(oc_vals, (target_len, 1))

        features = np.concatenate([price_features, onchain_features], axis=1)

        # --- Labels: future 24-step rolling volatility of hourly returns ---
        hourly_df = timeframe_dfs.get(self.cfg.TIMEFRAMES[0])
        if hourly_df is not None and len(hourly_df) > 24:
            future_vol = (
                hourly_df["close"]
                .pct_change()
                .shift(-24)
                .rolling(12)
                .std()
            )
            labels = future_vol.values[-target_len:]
            labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            labels = np.zeros(target_len)

        if len(labels) < target_len:
            labels = np.concatenate([np.zeros(target_len - len(labels)), labels])

        return features.astype(np.float32), labels.astype(np.float32)


def compute_drawdown_duration(equity_curve):
    """Calculate maximum drawdown duration in periods.

    Returns the longest streak of being below the previous
    equity high watermark.
    """
    import numpy as np
    peak = np.maximum.accumulate(equity_curve)
    in_drawdown = equity_curve < peak
    durations = []
    current = 0
    for dd in in_drawdown:
        if dd:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    return max(durations) if durations else 0
