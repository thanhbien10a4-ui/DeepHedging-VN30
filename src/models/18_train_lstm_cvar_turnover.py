"""Train a path-level Deep Hedging LSTM with CVaR + turnover penalty.

Purpose
-------
Step 17 showed that daily-variance minimization learns sensible hedge ratios,
but it can still have worse terminal 60-day dispersion than simpler baselines.
This step therefore optimizes risk at the PATH level.

Objective
---------
For each synthetic path:
    gross_terminal_pnl = sum_t (r_spot,t - h_t * r_fut,t)

A normalized turnover penalty is applied to opening, rebalancing and closing:
    turnover = |h_0| + sum_{t=1}^{T-1}|h_t-h_{t-1}| + |h_{T-1}|
    net_terminal_pnl = gross_terminal_pnl - lambda_tc * turnover

The training loss is empirical CVaR of LOSS in the worst 5% tail:
    loss_path = -net_terminal_pnl
    CVaR_5% = mean(largest 5% of loss_path within the batch)

Important: lambda_tc=0.001 is a normalized regularization coefficient for
turnover. It is NOT claimed to be the official VN30F1M fee/tax/slippage rate.
A later market-cost module must use verified official rules.

Leakage policy
--------------
- Train on synthetic TRAIN only.
- Early stop on synthetic VALIDATION only.
- Evaluate synthetic TEST only after selecting the best checkpoint.
- Real 2025-2026 market TEST data is never opened.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20261006
HIDDEN_SIZE = 128
NUM_LAYERS = 1
HEDGE_MAX = 1.5
BATCH_SIZE = 512
LEARNING_RATE = 5e-4
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP_NORM = 1.0
TAIL_FRACTION = 0.05
LAMBDA_TC = 0.001
LOSS_SCALE = 100.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/models/16_prepare_deep_hedging_sequences.py first."
        )
    with np.load(path) as z:
        required = {"features", "spot_forward_log_return", "fut_forward_log_return"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        x = z["features"].astype(np.float32)
        spot = z["spot_forward_log_return"].astype(np.float32)
        fut = z["fut_forward_log_return"].astype(np.float32)

    if x.ndim != 3 or spot.shape != x.shape[:2] or fut.shape != x.shape[:2]:
        raise ValueError(f"{path.name}: inconsistent tensor shapes.")
    if not np.isfinite(x).all() or not np.isfinite(spot).all() or not np.isfinite(fut).all():
        raise ValueError(f"{path.name}: NaN/inf detected.")
    return x, spot, fut


class LSTMHedgePolicy(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        raw = self.head(hidden).squeeze(-1)
        return HEDGE_MAX * torch.sigmoid(raw)


def path_turnover(h: torch.Tensor) -> torch.Tensor:
    open_cost = torch.abs(h[:, 0])
    rebalance = torch.sum(torch.abs(h[:, 1:] - h[:, :-1]), dim=1)
    close_cost = torch.abs(h[:, -1])
    return open_cost + rebalance + close_cost


def path_pnl(
    spot: torch.Tensor,
    fut: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    daily = spot - h * fut
    gross_terminal = torch.sum(daily, dim=1)
    turnover = path_turnover(h)
    net_terminal = gross_terminal - LAMBDA_TC * turnover
    return gross_terminal, net_terminal, turnover


def empirical_cvar_loss(losses: torch.Tensor, tail_fraction: float) -> torch.Tensor:
    if losses.ndim != 1:
        raise ValueError("losses must be a 1D tensor of path losses")
    k = max(1, int(math.ceil(tail_fraction * losses.numel())))
    worst, _ = torch.topk(losses, k=k, largest=True, sorted=False)
    return torch.mean(worst)


def objective(
    spot: torch.Tensor,
    fut: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    _, net_terminal, _ = path_pnl(spot, fut, h)
    losses = -net_terminal
    return empirical_cvar_loss(losses, TAIL_FRACTION) * LOSS_SCALE


def make_loader(
    x: np.ndarray,
    spot: np.ndarray,
    fut: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(spot),
        torch.from_numpy(fut),
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def evaluate_cvar(
    model: nn.Module,
    x: np.ndarray,
    spot: np.ndarray,
    fut: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    loader = make_loader(x, spot, fut, shuffle=False)
    losses_all: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, sb, fb in loader:
            xb = xb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)
            h = model(xb)
            _, net_terminal, _ = path_pnl(sb, fb, h)
            losses_all.append((-net_terminal).cpu())
    losses = torch.cat(losses_all, dim=0)
    return float(empirical_cvar_loss(losses, TAIL_FRACTION).item() * LOSS_SCALE)


def predict_hedges(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dummy = np.zeros(x.shape[:2], dtype=np.float32)
    loader = make_loader(x, dummy, dummy, shuffle=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in loader:
            h = model(xb.to(device, non_blocking=True))
            chunks.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def static_minvar_ratio(spot: np.ndarray, fut: np.ndarray) -> float:
    s = spot.reshape(-1).astype(np.float64)
    f = fut.reshape(-1).astype(np.float64)
    var_f = float(np.var(f, ddof=1))
    if var_f <= 0:
        raise ValueError("Synthetic futures return variance is non-positive.")
    return float(np.cov(s, f, ddof=1)[0, 1] / var_f)


def np_turnover(h: np.ndarray) -> np.ndarray:
    h64 = h.astype(np.float64)
    return (
        np.abs(h64[:, 0])
        + np.sum(np.abs(h64[:, 1:] - h64[:, :-1]), axis=1)
        + np.abs(h64[:, -1])
    )


def cvar_numpy(losses: np.ndarray, tail_fraction: float) -> float:
    x = np.asarray(losses, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(tail_fraction * len(x))))
    idx = np.argpartition(x, len(x) - k)[len(x) - k :]
    return float(np.mean(x[idx]))


def summarize_strategy(
    name: str,
    spot: np.ndarray,
    fut: np.ndarray,
    hedge: np.ndarray,
) -> dict:
    s = spot.astype(np.float64)
    f = fut.astype(np.float64)
    h = hedge.astype(np.float64)

    daily = s - h * f
    gross_terminal = np.sum(daily, axis=1)
    turnover = np_turnover(h)
    tc_penalty = LAMBDA_TC * turnover
    net_terminal = gross_terminal - tc_penalty
    losses = -net_terminal

    flat_daily = daily.reshape(-1)
    unhedged_flat = s.reshape(-1)
    variance_reduction = 1.0 - np.var(flat_daily, ddof=1) / np.var(unhedged_flat, ddof=1)

    return {
        "strategy": name,
        "daily_std_gross": float(np.std(flat_daily, ddof=1)),
        "variance_reduction_gross": float(variance_reduction),
        "gross_terminal_mean": float(np.mean(gross_terminal)),
        "gross_terminal_std": float(np.std(gross_terminal, ddof=1)),
        "net_terminal_mean": float(np.mean(net_terminal)),
        "net_terminal_std": float(np.std(net_terminal, ddof=1)),
        "net_terminal_q01": float(np.quantile(net_terminal, 0.01)),
        "net_terminal_q05": float(np.quantile(net_terminal, 0.05)),
        "cvar_loss_worst_5pct": cvar_numpy(losses, TAIL_FRACTION),
        "turnover_mean_open_rebalance_close": float(np.mean(turnover)),
        "tc_penalty_mean": float(np.mean(tc_penalty)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
        "hedge_ratio_min": float(np.min(h)),
        "hedge_ratio_max": float(np.max(h)),
    }


def main() -> None:
    seed_everything(SEED)

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed" / "deep_hedging"
    metrics_dir = root / "results" / "metrics"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    models_dir = root / "results" / "models"
    for d in (metrics_dir, tables_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    x_train, s_train, f_train = load_split(data_dir / "train_sequences.npz")
    x_val, s_val, f_val = load_split(data_dir / "validation_sequences.npz")
    x_test, s_test, f_test = load_split(data_dir / "test_sequences.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMHedgePolicy(n_features=x_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, s_train, f_train, shuffle=True)

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []
    checkpoint_path = models_dir / "lstm_cvar_turnover_best.pt"

    print("=" * 110)
    print("DEEP HEDGING LSTM — PATH CVAR + TURNOVER PENALTY TRAINING")
    print("=" * 110)
    print(f"PyTorch version      : {torch.__version__}")
    print(f"Device               : {device}")
    print(f"Train paths          : {x_train.shape[0]:,}")
    print(f"Validation paths     : {x_val.shape[0]:,}")
    print(f"Synthetic test paths : {x_test.shape[0]:,}")
    print(f"Steps/path           : {x_train.shape[1]}")
    print(f"Hidden size          : {HIDDEN_SIZE}")
    print(f"Hedge range          : [0, {HEDGE_MAX}]")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Tail fraction        : {TAIL_FRACTION:.2%} (worst-loss tail)")
    print(f"lambda_tc            : {LAMBDA_TC:.6f} (normalized turnover penalty)")
    print(f"Max epochs           : {MAX_EPOCHS}")
    print()

    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses: list[float] = []

        for xb, sb, fb in train_loader:
            xb = xb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            h = model(xb)
            loss = objective(sb, fb, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_cvar(model, x_val, s_val, f_val, device)

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features": int(x_train.shape[2]),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "hedge_max": HEDGE_MAX,
                    "tail_fraction": TAIL_FRACTION,
                    "lambda_tc": LAMBDA_TC,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_cvar_scaled": best_val,
                    "torch_version": torch.__version__,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_batch_cvar_scaled_mean": train_loss,
                "validation_full_cvar_scaled": val_loss,
                "best_so_far": bool(improved),
            }
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_batch_CVaR={train_loss:.6f} | "
            f"val_full_CVaR={val_loss:.6f} | "
            f"best={best_val:.6f} @ {best_epoch}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without validation improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    static_h = static_minvar_ratio(s_train, f_train)
    comparison_rows: list[dict] = []
    split_details: dict[str, dict] = {}

    for split_name, x, spot, fut in (
        ("validation", x_val, s_val, f_val),
        ("test", x_test, s_test, f_test),
    ):
        h_lstm = predict_hedges(model, x, device)
        h_zero = np.zeros_like(spot, dtype=np.float32)
        h_one = np.ones_like(spot, dtype=np.float32)
        h_static = np.full_like(spot, static_h, dtype=np.float32)

        rows = [
            summarize_strategy("unhedged", spot, fut, h_zero),
            summarize_strategy("delta_1to1", spot, fut, h_one),
            summarize_strategy("synthetic_train_static_minvar", spot, fut, h_static),
            summarize_strategy("lstm_cvar_turnover", spot, fut, h_lstm),
        ]
        for row in rows:
            row["split"] = split_name
            comparison_rows.append(row)

        split_details[split_name] = {
            "lstm": next(r for r in rows if r["strategy"] == "lstm_cvar_turnover"),
            "hedge_ratio_quantiles": {
                "q01": float(np.quantile(h_lstm, 0.01)),
                "q05": float(np.quantile(h_lstm, 0.05)),
                "q50": float(np.quantile(h_lstm, 0.50)),
                "q95": float(np.quantile(h_lstm, 0.95)),
                "q99": float(np.quantile(h_lstm, 0.99)),
            },
            "mean_hedge_by_step": np.mean(h_lstm.astype(np.float64), axis=0).tolist(),
        }

        if split_name == "validation":
            mean_h = np.mean(h_lstm.astype(np.float64), axis=0)
            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(1, len(mean_h) + 1), mean_h, label="LSTM CVaR+turnover")
            plt.axhline(static_h, label="synthetic train static min-var", linewidth=1.0)
            plt.axhline(1.0, label="1:1 hedge", linewidth=1.0)
            plt.xlabel("Decision step")
            plt.ylabel("Mean hedge ratio")
            plt.title("LSTM CVaR+Turnover Mean Hedge Ratio — Synthetic Validation")
            plt.legend()
            plt.tight_layout()
            fig.savefig(figures_dir / "lstm_cvar_turnover_mean_hedge_by_step.png", dpi=160)
            plt.close(fig)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = tables_dir / "lstm_cvar_turnover_synthetic_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = tables_dir / "lstm_cvar_turnover_training_history.csv"
    history_df.to_csv(history_path, index=False)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_batch_cvar_scaled_mean"], label="train batch CVaR")
    plt.plot(history_df["epoch"], history_df["validation_full_cvar_scaled"], label="validation full CVaR")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled CVaR loss")
    plt.title("LSTM CVaR + Turnover Training Curve")
    plt.legend()
    plt.tight_layout()
    training_fig = figures_dir / "lstm_cvar_turnover_training_curve.png"
    fig.savefig(training_fig, dpi=160)
    plt.close(fig)

    runtime = time.time() - start

    val_rows = comparison_df.loc[comparison_df["split"] == "validation"].set_index("strategy")
    lstm_val = val_rows.loc["lstm_cvar_turnover"]
    delta_val = val_rows.loc["delta_1to1"]
    static_val = val_rows.loc["synthetic_train_static_minvar"]

    sanity = {
        "mean_hedge_between_0.50_and_1.20": bool(0.50 <= lstm_val["hedge_ratio_mean"] <= 1.20),
        "cvar_better_than_unhedged": bool(
            lstm_val["cvar_loss_worst_5pct"] < val_rows.loc["unhedged", "cvar_loss_worst_5pct"]
        ),
        "finite_turnover": bool(np.isfinite(lstm_val["turnover_mean_open_rebalance_close"])),
        "not_saturated_low": bool(split_details["validation"]["hedge_ratio_quantiles"]["q05"] > 0.02),
        "not_saturated_high": bool(split_details["validation"]["hedge_ratio_quantiles"]["q95"] < HEDGE_MAX - 0.02),
    }

    report = {
        "model": "LSTM path-CVaR with normalized turnover penalty",
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": SEED,
        "architecture": {
            "n_features": int(x_train.shape[2]),
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "hedge_output_range": [0.0, HEDGE_MAX],
        },
        "objective": {
            "tail_fraction": TAIL_FRACTION,
            "interpretation": "worst 5% path losses; equivalent to 95% loss-CVaR convention",
            "lambda_tc": LAMBDA_TC,
            "lambda_tc_is_official_market_cost": False,
            "turnover_definition": "open + rebalancing + close",
            "terminal_pnl_proxy": "sum_t(spot_log_return - h_t*fut_log_return) - lambda_tc*turnover",
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "best_validation_cvar_scaled": best_val,
            "epochs_ran": len(history),
            "runtime_seconds": runtime,
        },
        "synthetic_train_static_minvar_ratio": static_h,
        "validation_comparison": {
            "lstm_cvar": float(lstm_val["cvar_loss_worst_5pct"]),
            "delta_cvar": float(delta_val["cvar_loss_worst_5pct"]),
            "static_minvar_cvar": float(static_val["cvar_loss_worst_5pct"]),
        },
        "split_details": split_details,
        "sanity_checks": sanity,
        "all_sanity_checks_pass": bool(all(sanity.values())),
        "real_market_test_used": False,
        "known_limitation": (
            "lambda_tc=0.001 is a normalized turnover regularizer, not verified official VN30F1M cost. "
            "Realistic net-cost evaluation requires a separately documented market-cost model."
        ),
    }

    report_path = metrics_dir / "lstm_cvar_turnover_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print()
    print("=" * 110)
    print("LSTM PATH CVAR + TURNOVER TRAINING COMPLETE")
    print("=" * 110)
    print(f"Best epoch                  : {best_epoch}")
    print(f"Best validation CVaR scaled : {best_val:.6f}")
    print(f"Synthetic train static h*   : {static_h:.6f}")
    print(f"Runtime                     : {runtime:.1f} sec")
    print()
    print(
        comparison_df[
            [
                "split",
                "strategy",
                "net_terminal_mean",
                "net_terminal_std",
                "net_terminal_q05",
                "cvar_loss_worst_5pct",
                "turnover_mean_open_rebalance_close",
                "hedge_ratio_mean",
                "hedge_ratio_std",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("Synthetic validation LSTM hedge quantiles:")
    for key, value in split_details["validation"]["hedge_ratio_quantiles"].items():
        print(f"  {key}: {value:.6f}")
    print()
    for name, passed in sanity.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL SANITY CHECKS PASS : {all(sanity.values())}")
    print()
    print("IMPORTANT: lambda_tc is a normalized turnover penalty, NOT an official VN30F1M fee rate.")
    print(f"Best model       : {checkpoint_path}")
    print(f"Training history : {history_path}")
    print(f"Comparison table : {comparison_path}")
    print(f"Report           : {report_path}")
    print(f"Training figure  : {training_fig}")
    print("REAL MARKET TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
