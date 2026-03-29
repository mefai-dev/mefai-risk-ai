# mefai-risk-ai

[![CI](https://github.com/mefai-dev/mefai-risk-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/mefai-dev/mefai-risk-ai/actions/workflows/ci.yml)


AI powered cryptocurrency risk assessment using a Temporal Fusion Transformer (BiLSTM + multi head attention) trained on Binance market data.

## Features

- **Temporal Fusion Transformer** -- bidirectional LSTM encoder, dual self-attention blocks, and two output heads (risk score + volatility).
- **Multi-timeframe analysis** -- fetches 1h, 4h, and 1d candles from the Binance public API.
- **Fixed feature schema** -- 25 technical indicators per timeframe (RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, etc.) via the `ta` library. No C-level `ta-lib` dependency.
- **Sliding-window dataset** -- generates thousands of training samples per coin instead of one.
- **Time-based train val split** -- no look-ahead bias.
- **Early stopping + LR scheduling** -- AdamW optimiser with Huber loss.
- **Portfolio optimisation** -- inverse-volatility weighting with risk-score adjustment.
- **CLI** -- `mefai-risk train`, `mefai-risk evaluate`, `mefai-risk portfolio`.

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd mefai-risk-ai

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- See `requirements.txt` for the full list.

## Quick start

```bash
# Train on BTC, ETH, SOL for 10 epochs
mefai-risk train --coins BTC,ETH,SOL --epochs 10 --window-size 64

# Evaluate the saved checkpoint
mefai-risk evaluate --coins BTC,ETH,SOL

# Generate a risk report and portfolio allocation
mefai-risk portfolio --coins BTC,ETH,SOL --risk-tolerance medium --output report.json
```

Or run the example script:

```bash
python examples/quick_start.py
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `MEFAI_COINS` | Space separated coin list | BTC,ETH,BNB,... |
| `MEFAI_TIMEFRAMES` | Space separated timeframes | 1h,4h,1d |
| `MEFAI_EPOCHS` | Max training epochs | 500 |
| `MEFAI_BATCH_SIZE` | Batch size | 32 |
| `MEFAI_WINDOW_SIZE` | Candles to download per timeframe | 256 |
| `MEFAI_MODEL_DIR` | Checkpoint directory | checkpoints |
| `BINANCE_API_KEY` | Optional Binance API key for higher rate limits | *(none)* |

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
mefai_risk/
  config.py          Settings dataclass, env var reading
  cli.py             Click CLI entry-point
  models/
    attention.py     MultiHeadAttention
    tft.py           TemporalFusionTransformer
  data/
    fetcher.py       BinanceFetcher (public klines API)
    onchain.py       OnChainProvider ABC + MockProvider (with warning)
    features.py      FeatureEngine (fixed N_FEATURES schema)
    dataset.py       CryptoDataset (sliding windows) + build_dataloaders
  training/
    trainer.py       RiskTrainer (AdamW, Huber, early stopping)
  portfolio/
    manager.py       PortfolioManager (risk scoring + allocation)
```

## Bugs fixed from original RISKAI.txt

1. **data_cache type mismatch** -- `_calculate_market_metrics` indexed `np.ndarray` as dict. Fixed: close prices stored in a separate `_close_series` dict.
2. **model.evaluate()** -- `PortfolioManager` called `self.model.evaluate()` which does not exist on `nn.Module`. Fixed: delegates to `RiskTrainer.evaluate()`.
3. **Feature dimension mismatch** -- different timeframes produced different column counts. Fixed: `FeatureEngine` always outputs exactly `N_FEATURES` columns per timeframe.
4. **No train val split** -- original used the same data for training and validation. Fixed: time based split via `build_dataloaders()`.
5. **Fake API keys** -- `Config.get_headers()` generated a SHA-256 hash as a fake key. Fixed: reads `BINANCE_API_KEY` from env; public endpoints need no key.
6. **Too few samples** -- `DataLoader(batch_size=32)` with only 10 items (one per coin). Fixed: sliding window dataset produces thousands of samples.
7. **Silent mock data** -- on-chain metrics were random numbers with no warning. Fixed: `MockOnChainProvider` emits an explicit `warnings.warn()`.

## License

MIT
