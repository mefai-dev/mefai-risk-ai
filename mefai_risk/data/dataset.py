"""PyTorch Dataset with sliding-window sampling and time-based split.

Bug fixes vs. original RISKAI.txt
----------------------------------
* ``data_cache`` now stores ``np.ndarray`` (not dicts), so
  ``_calculate_market_metrics`` no longer indexes arrays as dicts.
* Sliding windows produce thousands of samples instead of one per coin.
* Time-based train/val split prevents look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

from mefai_risk.config import Settings, settings as _default_settings
from mefai_risk.data.features import FeatureEngine
from mefai_risk.data.fetcher import BinanceFetcher
from mefai_risk.data.onchain import MockOnChainProvider, OnChainProvider

logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    """Sliding-window dataset over multi-coin Binance data.

    Each item is a ``(features_window, label)`` pair where
    ``features_window`` has shape ``(window_size, n_features)`` and
    ``label`` is a scalar risk proxy.

    Parameters
    ----------
    coins : list[str]
        Coin tickers (without ``USDT`` suffix).
    cfg : Settings
        Configuration.
    onchain_provider : OnChainProvider, optional
        Defaults to :class:`MockOnChainProvider`.
    fetcher : BinanceFetcher, optional
        Defaults to a fresh instance.
    window_size : int
        Number of time-steps per sliding window.
    stride : int
        Step between successive windows.
    """

    def __init__(
        self,
        coins: List[str],
        cfg: Optional[Settings] = None,
        onchain_provider: Optional[OnChainProvider] = None,
        fetcher: Optional[BinanceFetcher] = None,
        window_size: int = 64,
        stride: int = 1,
    ) -> None:
        self.cfg = cfg or _default_settings
        self.coins = coins
        self.window_size = window_size
        self.stride = stride
        self.fetcher = fetcher or BinanceFetcher(self.cfg)
        self.onchain = onchain_provider or MockOnChainProvider()
        self.engine = FeatureEngine(self.cfg)
        self.scaler = RobustScaler()

        # Stores per-coin: (features_array, labels_array)
        self._coin_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Stores per-coin raw close series for cross-asset metrics
        self._close_series: Dict[str, np.ndarray] = {}

        # Index mapping: list of (coin, start_idx) for __getitem__
        self._index: List[Tuple[str, int]] = []

        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.market_beta: Dict[str, float] = {}

        self._preload()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _preload(self) -> None:
        limit = min(self.cfg.WINDOW_SIZE + 100, 1000)  # extra rows for indicator warm-up

        for coin in self.coins:
            tf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            for tf in self.cfg.TIMEFRAMES:
                tf_dfs[tf] = self.fetcher.fetch_coin(coin, interval=tf, limit=limit)

            onchain_data = self.onchain.fetch(coin)
            features, labels = self.engine.build_feature_matrix(
                tf_dfs, onchain_data, target_len=self.cfg.WINDOW_SIZE,
            )

            # Fit scaler per coin then transform
            features = self.scaler.fit_transform(features)
            self._coin_data[coin] = (features, labels)

            # Store close prices from first timeframe for market metrics
            first_tf = self.cfg.TIMEFRAMES[0]
            if tf_dfs.get(first_tf) is not None:
                self._close_series[coin] = (
                    tf_dfs[first_tf]["close"].values[-self.cfg.WINDOW_SIZE:]
                )

            # Build sliding-window index
            n_steps = features.shape[0]
            for start in range(0, n_steps - self.window_size + 1, self.stride):
                self._index.append((coin, start))

        self._calculate_market_metrics()
        logger.info(
            "Dataset ready: %d coins, %d samples (window=%d, stride=%d)",
            len(self.coins), len(self._index), self.window_size, self.stride,
        )

    # ------------------------------------------------------------------
    def _calculate_market_metrics(self) -> None:
        """Cross-asset correlation matrix and beta-to-BTC."""
        closes = []
        valid_coins = []
        for coin in self.coins:
            if coin in self._close_series:
                closes.append(self._close_series[coin])
                valid_coins.append(coin)

        if len(closes) < 2:
            return

        # Align lengths
        min_len = min(len(c) for c in closes)
        aligned = [c[-min_len:] for c in closes]

        returns_df = pd.DataFrame(
            {coin: pd.Series(c).pct_change() for coin, c in zip(valid_coins, aligned)}
        ).dropna()

        self.correlation_matrix = returns_df.corr()

        if "BTC" in returns_df.columns:
            btc_ret = returns_df["BTC"]
            btc_var = btc_ret.var()
            for coin in valid_coins:
                if coin != "BTC" and btc_var > 0:
                    cov = returns_df[coin].cov(btc_ret)
                    self.market_beta[coin] = float(cov / btc_var)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        """Number of features per time-step."""
        first = next(iter(self._coin_data.values()))
        return first[0].shape[1]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        coin, start = self._index[idx]
        features, labels = self._coin_data[coin]
        end = start + self.window_size
        window = features[start:end]
        label = labels[end - 1]  # target at the last step of the window
        return {
            "features": torch.as_tensor(window, dtype=torch.float32),
            "label": torch.as_tensor([label], dtype=torch.float32),
        }


# ------------------------------------------------------------------
# Convenience: build train/val DataLoaders with time-based split
# ------------------------------------------------------------------

def build_dataloaders(
    coins: Optional[List[str]] = None,
    cfg: Optional[Settings] = None,
    window_size: int = 64,
    stride: int = 1,
) -> Tuple[DataLoader, DataLoader, CryptoDataset]:
    """Build train and validation DataLoaders.

    The split is **time-based**: the first ``(1 - val_ratio)`` fraction
    of each coin's windows go to training, the rest to validation.
    This avoids look-ahead bias.

    Returns
    -------
    train_loader, val_loader, dataset
    """
    cfg = cfg or _default_settings
    coins = coins or cfg.COINS

    dataset = CryptoDataset(
        coins=coins, cfg=cfg, window_size=window_size, stride=stride,
    )

    # Time-based split per coin
    train_indices: List[int] = []
    val_indices: List[int] = []

    # Group indices by coin
    coin_groups: Dict[str, List[int]] = {}
    for global_idx, (coin, _start) in enumerate(dataset._index):
        coin_groups.setdefault(coin, []).append(global_idx)

    for coin, indices in coin_groups.items():
        split = int(len(indices) * (1.0 - cfg.VAL_RATIO))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    logger.info("Train samples: %d  |  Val samples: %d", len(train_indices), len(val_indices))
    return train_loader, val_loader, dataset
