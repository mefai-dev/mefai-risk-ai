"""CLI entry-point: ``mefai-risk train | evaluate | portfolio``."""

from __future__ import annotations

import json
import logging
import sys

import click
import torch

from mefai_risk.config import Settings


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """mefai-risk -- AI-powered cryptocurrency risk assessment."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ------------------------------------------------------------------
# train
# ------------------------------------------------------------------
@cli.command()
@click.option("--coins", default=None, help="Comma-separated coin list (default: config).")
@click.option("--epochs", default=None, type=int, help="Override max epochs.")
@click.option("--batch-size", default=None, type=int, help="Override batch size.")
@click.option("--window-size", default=64, type=int, help="Sliding window length.")
@click.option("--stride", default=1, type=int, help="Sliding window stride.")
def train(
    coins: str | None,
    epochs: int | None,
    batch_size: int | None,
    window_size: int,
    stride: int,
) -> None:
    """Fetch data, build features, and train the TFT model."""
    from mefai_risk.data.dataset import build_dataloaders
    from mefai_risk.models.tft import TemporalFusionTransformer
    from mefai_risk.training.trainer import RiskTrainer

    cfg = Settings()
    if coins:
        cfg.COINS = [c.strip().upper() for c in coins.split(",")]
    if epochs:
        cfg.EPOCHS = epochs
    if batch_size:
        cfg.BATCH_SIZE = batch_size

    click.echo(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    click.echo(f"Coins : {cfg.COINS}")
    click.echo(f"Epochs: {cfg.EPOCHS}  |  Batch: {cfg.BATCH_SIZE}  |  Window: {window_size}")
    click.echo()

    click.echo("Fetching data and building features ...")
    train_loader, val_loader, dataset = build_dataloaders(
        coins=cfg.COINS, cfg=cfg, window_size=window_size, stride=stride,
    )

    n_feat = dataset.n_features
    click.echo(f"Features per step: {n_feat}")
    click.echo(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    click.echo()

    model = TemporalFusionTransformer(
        input_size=n_feat,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        n_heads=cfg.N_HEADS,
    )
    trainer = RiskTrainer(model, cfg=cfg)

    click.echo("Training ...")
    history = trainer.fit(train_loader, val_loader)

    final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")
    click.echo(f"\nDone. Final val_loss = {final_val:.6f}")


# ------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------
@cli.command()
@click.option("--coins", default=None, help="Comma-separated coin list.")
@click.option("--checkpoint", default=None, help="Path to model checkpoint.")
@click.option("--window-size", default=64, type=int)
def evaluate(coins: str | None, checkpoint: str | None, window_size: int) -> None:
    """Evaluate a trained model on the validation set."""
    import os

    from mefai_risk.data.dataset import build_dataloaders
    from mefai_risk.models.tft import TemporalFusionTransformer
    from mefai_risk.training.trainer import RiskTrainer

    cfg = Settings()
    if coins:
        cfg.COINS = [c.strip().upper() for c in coins.split(",")]

    click.echo("Fetching data ...")
    _train_loader, val_loader, dataset = build_dataloaders(
        coins=cfg.COINS, cfg=cfg, window_size=window_size,
    )

    model = TemporalFusionTransformer(
        input_size=dataset.n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        n_heads=cfg.N_HEADS,
    )

    ckpt = checkpoint or os.path.join(cfg.MODEL_DIR, "best_model.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        click.echo(f"Loaded checkpoint: {ckpt}")
    else:
        click.echo(f"WARNING: No checkpoint found at {ckpt}; using random weights.")

    trainer = RiskTrainer(model, cfg=cfg)
    metrics = trainer.evaluate(val_loader)

    click.echo(f"Val loss    : {metrics['loss']:.6f}")
    click.echo(f"Val Spearman: {metrics['spearman']:.4f}")


# ------------------------------------------------------------------
# portfolio
# ------------------------------------------------------------------
@cli.command()
@click.option("--coins", default=None, help="Comma-separated coin list.")
@click.option("--checkpoint", default=None, help="Path to model checkpoint.")
@click.option("--risk-tolerance", default="medium", type=click.Choice(["low", "medium", "high"]))
@click.option("--output", default="risk_report.json", help="Report output path.")
@click.option("--window-size", default=64, type=int)
def portfolio(
    coins: str | None,
    checkpoint: str | None,
    risk_tolerance: str,
    output: str,
    window_size: int,
) -> None:
    """Generate a risk report and portfolio allocation."""
    import os

    from mefai_risk.data.dataset import CryptoDataset
    from mefai_risk.models.tft import TemporalFusionTransformer
    from mefai_risk.portfolio.manager import PortfolioManager
    from mefai_risk.training.trainer import RiskTrainer

    cfg = Settings()
    if coins:
        cfg.COINS = [c.strip().upper() for c in coins.split(",")]

    click.echo("Fetching data ...")
    dataset = CryptoDataset(coins=cfg.COINS, cfg=cfg, window_size=window_size)

    model = TemporalFusionTransformer(
        input_size=dataset.n_features,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        n_heads=cfg.N_HEADS,
    )

    ckpt = checkpoint or os.path.join(cfg.MODEL_DIR, "best_model.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        click.echo(f"Loaded checkpoint: {ckpt}")
    else:
        click.echo("WARNING: No checkpoint found; using random weights.")

    trainer = RiskTrainer(model, cfg=cfg)
    manager = PortfolioManager(trainer, dataset, cfg=cfg)

    click.echo(f"Generating portfolio (risk_tolerance={risk_tolerance}) ...")
    allocation = manager.optimize_portfolio(risk_tolerance=risk_tolerance)
    manager.save_report(output)

    click.echo(f"\nReport saved to {output}")
    click.echo("\nAllocation:")
    for coin, info in sorted(allocation.items(), key=lambda x: x[1]["weight"], reverse=True):
        click.echo(f"  {coin:>5s}  weight={info['weight']:.2%}  leverage={info['leverage']:.1f}x")


# ------------------------------------------------------------------
if __name__ == "__main__":
    cli()
