"""Train a first Deep Hedging LSTM sanity-check baseline with PyTorch.

Purpose
-------
This is intentionally the simplest learned hedge policy before adding CVaR and
transaction costs. The objective is variance/MSE minimization of the synthetic
hedged return so we can verify that the recurrent policy learns a sensible
hedge ratio relative to the traditional 0.76-1.00 baseline range.

Data
----
Uses causal tensors from step 16:
    data/processed/deep_hedging/train_sequences.npz
    data/processed/deep_hedging/validation_sequences.npz
    data/processed/deep_hedging/test_sequences.npz

The synthetic test split is only a Monte Carlo holdout. Real 2025-2026 market
test data is never opened here.

Model
-----
- 1-layer LSTM, hidden size 128
- linear head -> sigmoid -> hedge ratio in [0, 1.5]
- one hedge ratio h_t for every decision time t

Loss
----
For each forward interval:
    r_hedged,t = r_spot,t - h_t * r_futures,t

Training minimizes mean squared hedged return. Since returns are O(1e-2), the
reported optimization loss is multiplied by 1e4 for readability; this scaling
does not change the optimizer optimum.

No transaction-cost penalty and no CVaR objective are used in this step.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20261005
HIDDEN_SIZE = 128
NUM_LAYERS = 1
HEDGE_MAX = 1.5
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 40
PATIENCE = 6
LOSS_SCALE = 1e4
GRAD_CLIP_NORM = 1.0


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

    if x.ndim != 3:
        raise ValueError(f"{path.name}: features must be 3D [paths, steps, features].")
    if spot.shape != x.shape[:2] or fut.shape != x.shape[:2]:
        raise ValueError(f"{path.name}: forward-return shapes do not match feature time dimensions.")
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
        hedge_ratio = HEDGE_MAX * torch.sigmoid(raw)
        return hedge_ratio


def hedged_return(
    spot_return: torch.Tensor,
    fut_return: torch.Tensor,
    hedge_ratio: torch.Tensor,
) -> torch.Tensor:
    return spot_return - hedge_ratio * fut_return


def loss_fn(
    spot_return: torch.Tensor,
    fut_return: torch.Tensor,
    hedge_ratio: torch.Tensor,
) -> torch.Tensor:
    r_h = hedged_return(spot_return, fut_return, hedge_ratio)
    return torch.mean(r_h.square()) * LOSS_SCALE


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
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, spot, fut in loader:
            x = x.to(device, non_blocking=True)
            spot = spot.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            h = model(x)
            loss = loss_fn(spot, fut, h)
            batch_n = x.shape[0]
            total += float(loss.item()) * batch_n
            n += batch_n
    return total / n


def collect_predictions(
    model: nn.Module,
    x: np.ndarray,
    spot: np.ndarray,
    fut: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(x, spot, fut, shuffle=False)
    hedge_chunks: list[np.ndarray] = []
    return_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, sb, fb in loader:
            xb = xb.to(device, non_blocking=True)
            h = model(xb)
            h_cpu = h.cpu().numpy().astype(np.float32)
            r_h = sb.numpy().astype(np.float32) - h_cpu * fb.numpy().astype(np.float32)
            hedge_chunks.append(h_cpu)
            return_chunks.append(r_h)
    return np.concatenate(hedge_chunks, axis=0), np.concatenate(return_chunks, axis=0)


def static_minvar_ratio(spot: np.ndarray, fut: np.ndarray) -> float:
    s = spot.reshape(-1).astype(np.float64)
    f = fut.reshape(-1).astype(np.float64)
    var_f = float(np.var(f, ddof=1))
    if var_f <= 0:
        raise ValueError("Synthetic futures return variance is non-positive.")
    return float(np.cov(s, f, ddof=1)[0, 1] / var_f)


def summarize_strategy(
    name: str,
    spot: np.ndarray,
    fut: np.ndarray,
    hedge: np.ndarray,
) -> dict:
    r = spot.astype(np.float64) - hedge.astype(np.float64) * fut.astype(np.float64)
    u = spot.astype(np.float64)

    flat_r = r.reshape(-1)
    flat_u = u.reshape(-1)
    variance_reduction = 1.0 - np.var(flat_r, ddof=1) / np.var(flat_u, ddof=1)

    path_terminal = np.sum(r, axis=1)
    path_std = float(np.std(path_terminal, ddof=1))

    turnover = np.abs(np.diff(hedge.astype(np.float64), axis=1, prepend=hedge[:, [0]].astype(np.float64)))

    return {
        "strategy": name,
        "daily_mean": float(np.mean(flat_r)),
        "daily_std": float(np.std(flat_r, ddof=1)),
        "variance_reduction": float(variance_reduction),
        "terminal_60d_mean_log_pnl_proxy": float(np.mean(path_terminal)),
        "terminal_60d_std_log_pnl_proxy": path_std,
        "hedge_ratio_mean": float(np.mean(hedge)),
        "hedge_ratio_std": float(np.std(hedge, ddof=1)),
        "hedge_ratio_min": float(np.min(hedge)),
        "hedge_ratio_max": float(np.max(hedge)),
        "mean_daily_turnover_abs_dh": float(np.mean(turnover)),
        "total_mean_path_turnover_abs_dh": float(np.mean(np.sum(turnover, axis=1))),
    }


def main() -> None:
    seed_everything(SEED)

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed" / "deep_hedging"
    results_metrics_dir = root / "results" / "metrics"
    results_tables_dir = root / "results" / "tables"
    results_figures_dir = root / "results" / "figures"
    model_dir = root / "results" / "models"
    for d in (results_metrics_dir, results_tables_dir, results_figures_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    x_train, s_train, f_train = load_split(data_dir / "train_sequences.npz")
    x_val, s_val, f_val = load_split(data_dir / "validation_sequences.npz")
    x_test, s_test, f_test = load_split(data_dir / "test_sequences.npz")

    if x_train.shape[2] != x_val.shape[2] or x_train.shape[2] != x_test.shape[2]:
        raise ValueError("Feature count differs across synthetic splits.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMHedgePolicy(n_features=x_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = make_loader(x_train, s_train, f_train, shuffle=True)
    val_loader = make_loader(x_val, s_val, f_val, shuffle=False)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_state_path = model_dir / "lstm_variance_baseline_best.pt"

    print("=" * 104)
    print("DEEP HEDGING LSTM VARIANCE BASELINE TRAINING")
    print("=" * 104)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")
    print(f"Train paths     : {x_train.shape[0]:,}")
    print(f"Validation paths: {x_val.shape[0]:,}")
    print(f"Synthetic test  : {x_test.shape[0]:,}")
    print(f"Steps/path      : {x_train.shape[1]}")
    print(f"Features        : {x_train.shape[2]}")
    print(f"Hidden size     : {HIDDEN_SIZE}")
    print(f"Hedge range     : [0, {HEDGE_MAX}]")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Max epochs      : {MAX_EPOCHS}")
    print()

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_n = 0

        for x, spot, fut in train_loader:
            x = x.to(device, non_blocking=True)
            spot = spot.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            h = model(x)
            loss = loss_fn(spot, fut, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            batch_n = x.shape[0]
            train_total += float(loss.item()) * batch_n
            train_n += batch_n

        train_loss = train_total / train_n
        val_loss = evaluate_loss(model, val_loader, device)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features": int(x_train.shape[2]),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "hedge_max": HEDGE_MAX,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_loss_scaled": best_val,
                    "torch_version": torch.__version__,
                },
                best_state_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss_scaled": train_loss,
                "validation_loss_scaled": val_loss,
                "best_so_far": bool(improved),
            }
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"best={best_val:.6f} @ {best_epoch}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without validation improvement.")
            break

    checkpoint = torch.load(best_state_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    synthetic_train_minvar = static_minvar_ratio(s_train, f_train)

    split_payload: dict[str, dict] = {}
    comparison_rows: list[dict] = []

    for split_name, x, spot, fut in (
        ("validation", x_val, s_val, f_val),
        ("test", x_test, s_test, f_test),
    ):
        h_lstm, r_lstm = collect_predictions(model, x, spot, fut, device)

        h_zero = np.zeros_like(spot, dtype=np.float32)
        h_one = np.ones_like(spot, dtype=np.float32)
        h_static = np.full_like(spot, synthetic_train_minvar, dtype=np.float32)

        rows = [
            summarize_strategy("unhedged", spot, fut, h_zero),
            summarize_strategy("delta_1to1", spot, fut, h_one),
            summarize_strategy("synthetic_train_static_minvar", spot, fut, h_static),
            summarize_strategy("lstm_variance", spot, fut, h_lstm),
        ]
        for row in rows:
            row["split"] = split_name
            comparison_rows.append(row)

        hedge_by_step = np.mean(h_lstm.astype(np.float64), axis=0)
        split_payload[split_name] = {
            "lstm_metrics": next(r for r in rows if r["strategy"] == "lstm_variance"),
            "mean_hedge_ratio_by_step": hedge_by_step.tolist(),
            "hedge_ratio_quantiles": {
                "q01": float(np.quantile(h_lstm, 0.01)),
                "q05": float(np.quantile(h_lstm, 0.05)),
                "q50": float(np.quantile(h_lstm, 0.50)),
                "q95": float(np.quantile(h_lstm, 0.95)),
                "q99": float(np.quantile(h_lstm, 0.99)),
            },
        }

        if split_name == "validation":
            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(1, len(hedge_by_step) + 1), hedge_by_step, linewidth=1.5)
            plt.axhline(synthetic_train_minvar, linewidth=1.0, label="synthetic train static min-var")
            plt.axhline(1.0, linewidth=1.0, label="1:1 hedge")
            plt.xlabel("Decision step")
            plt.ylabel("Mean hedge ratio")
            plt.title("LSTM Mean Hedge Ratio by Time Step — Synthetic Validation")
            plt.legend()
            plt.tight_layout()
            fig.savefig(results_figures_dir / "lstm_variance_mean_hedge_by_step.png", dpi=160)
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            plt.hist(h_lstm.reshape(-1), bins=60)
            plt.axvline(synthetic_train_minvar, linewidth=1.0, label="synthetic train static min-var")
            plt.axvline(1.0, linewidth=1.0, label="1:1 hedge")
            plt.xlabel("Hedge ratio")
            plt.ylabel("Count")
            plt.title("LSTM Hedge Ratio Distribution — Synthetic Validation")
            plt.legend()
            plt.tight_layout()
            fig.savefig(results_figures_dir / "lstm_variance_hedge_distribution.png", dpi=160)
            plt.close(fig)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = results_tables_dir / "lstm_variance_synthetic_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = results_tables_dir / "lstm_variance_training_history.csv"
    history_df.to_csv(history_path, index=False)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss_scaled"], label="train")
    plt.plot(history_df["epoch"], history_df["validation_loss_scaled"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE loss")
    plt.title("LSTM Variance Baseline Training Curve")
    plt.legend()
    plt.tight_layout()
    training_fig = results_figures_dir / "lstm_variance_training_curve.png"
    fig.savefig(training_fig, dpi=160)
    plt.close(fig)

    runtime_sec = time.time() - start_time
    val_lstm = split_payload["validation"]["lstm_metrics"]
    sanity = {
        "mean_hedge_between_0.60_and_1.20": bool(
            0.60 <= val_lstm["hedge_ratio_mean"] <= 1.20
        ),
        "variance_reduction_at_least_0.85": bool(
            val_lstm["variance_reduction"] >= 0.85
        ),
        "not_saturated_low": bool(
            split_payload["validation"]["hedge_ratio_quantiles"]["q05"] > 0.02
        ),
        "not_saturated_high": bool(
            split_payload["validation"]["hedge_ratio_quantiles"]["q95"] < HEDGE_MAX - 0.02
        ),
    }

    report = {
        "model": "LSTM variance-minimization sanity baseline",
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
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "loss": "mean squared synthetic hedged log-return proxy",
            "loss_scale_for_optimization_and_reporting": LOSS_SCALE,
            "transaction_cost_penalty": 0.0,
            "cvar_objective": False,
            "best_epoch": best_epoch,
            "best_validation_loss_scaled": best_val,
            "epochs_ran": len(history),
            "runtime_seconds": runtime_sec,
        },
        "synthetic_train_static_minvar_ratio": synthetic_train_minvar,
        "synthetic_split_results": split_payload,
        "sanity_checks": sanity,
        "all_sanity_checks_pass": bool(all(sanity.values())),
        "real_market_test_used": False,
        "next_step": (
            "If sanity checks pass, add CVaR and transaction-cost objectives, then evaluate/fine-tune on real train/validation data."
        ),
    }

    report_path = results_metrics_dir / "lstm_variance_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print()
    print("=" * 104)
    print("LSTM VARIANCE BASELINE TRAINING COMPLETE")
    print("=" * 104)
    print(f"Best epoch                  : {best_epoch}")
    print(f"Best validation loss scaled : {best_val:.6f}")
    print(f"Synthetic train static h*   : {synthetic_train_minvar:.6f}")
    print(f"Runtime                     : {runtime_sec:.1f} sec")
    print()
    print(
        comparison_df[
            [
                "split",
                "strategy",
                "daily_std",
                "variance_reduction",
                "terminal_60d_std_log_pnl_proxy",
                "hedge_ratio_mean",
                "hedge_ratio_std",
                "mean_daily_turnover_abs_dh",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("Synthetic validation LSTM hedge quantiles:")
    for key, value in split_payload["validation"]["hedge_ratio_quantiles"].items():
        print(f"  {key}: {value:.6f}")
    print()
    for name, passed in sanity.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL SANITY CHECKS PASS : {all(sanity.values())}")
    print()
    print(f"Best model       : {best_state_path}")
    print(f"Training history : {history_path}")
    print(f"Comparison table : {comparison_path}")
    print(f"Report           : {report_path}")
    print(f"Training figure  : {training_fig}")
    print("REAL MARKET TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
