"""On-chain data providers.

In production you would integrate with Glassnode, Nansen, or similar
services.  The ``MockOnChainProvider`` generates synthetic data and
**emits an explicit warning** so users are never silently misled.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class OnChainProvider(ABC):
    """Abstract base class for on-chain data sources."""

    @abstractmethod
    def fetch(self, coin: str) -> Dict[str, float]:
        """Return a dict of on-chain metrics for *coin*.

        Expected keys (at minimum):
            nvt, velocity, sopr, exchange_netflow,
            hashrate, active_addresses, whale_transactions, fees_usd
        """
        ...


class MockOnChainProvider(OnChainProvider):
    """Synthetic on-chain data for development / testing.

    .. warning::
        This provider produces **random numbers**, not real on-chain
        data.  A ``UserWarning`` is emitted on first call.
    """

    _warned: bool = False

    def fetch(self, coin: str) -> Dict[str, float]:
        if not MockOnChainProvider._warned:
            warnings.warn(
                "MockOnChainProvider is generating SYNTHETIC on-chain data. "
                "Do NOT use these values for real trading decisions. "
                "Integrate a real provider (e.g. Glassnode) for production use.",
                UserWarning,
                stacklevel=2,
            )
            MockOnChainProvider._warned = True

        rng = np.random.default_rng()
        return {
            "nvt": float(rng.normal(50, 10)),
            "velocity": float(rng.uniform(0.5, 2.0)),
            "sopr": float(rng.normal(1.0, 0.1)),
            "exchange_netflow": float(rng.integers(-1000, 1000)),
            "hashrate": float(rng.uniform(1e6, 1e7)),
            "active_addresses": float(rng.integers(10_000, 50_000)),
            "whale_transactions": float(rng.integers(10, 100)),
            "fees_usd": float(rng.uniform(1_000, 100_000)),
        }
