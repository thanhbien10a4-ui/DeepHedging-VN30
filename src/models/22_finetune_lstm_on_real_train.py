"""Fine-tune the synthetic-pretrained LSTM on REAL TRAIN, select on REAL VALIDATION.

Transfer policy
---------------
- Initialize from step-18 checkpoint trained on frozen synthetic data.
- Clip only neural-network input features to [-5, 5], as selected by step 21.
- Real spot/futures returns used for PnL are never clipped or rescaled.
- Optimize the same path-level CVaR + normalized turnover objective as step 18.
- Use a small learning rate and early stopping to reduce catastrophic forgetting.
- Epoch 0 (the untouched pretrained model) is a valid candidate. Fine-tuning is
  accepted only if REAL VALIDATION CVaR improves.

Leakage policy
--------------
- Fine-tune: REAL TRAIN only (2017-2022 episodes).
- Checkpoint selection: REAL VALIDATION only (2023-2024 episodes).
- REAL TEST 2025-2026 is never opened.

Important limitations
---------------------
Real 60-day episodes overlap with stride 1, so episode-level metrics are useful
for model selection but are not independent-sample statistical inference.
lambda_tc=0.001 remains a normalized turnover regularizer, not an official
VN30F1M fee/tax/slippage specification.
"""

from __future__ import annotations

import copy
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

SEED = 20261007
CLIP_Z = 5.0
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
MAX_EPOCHS = 30
PATIENCE = 6
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


class LSTMHedgePolicy(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, hedge_max: float) -> None:
        super().__init__()
        self.hedge_max = float(hedge_max)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        raw = self.head(hidden).squeeze(-1)
        return self.hedge_max * torch.sigmoid(raw)


def load_sequences(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run step 19 first.")
    with np.load(path) as z:
        x = z["features"].astype(np.float32)
        s = z["spot_forward_log_return"].astype(np.float32)
        f = z["fut_forward_log_return"].astype(np.float32)
    if x.ndim != 3 or s.shape != x.shape[:2] or f.shape != x.shape[:2]:
        raise ValueError(f"{path.name}: inconsistent sequence shapes.")
    if not np.isfinite(x).all() or not np.isfinite(s).all() or not np.isfinite(f).all():
        raise ValueError(f"{path.name}: NaN/inf detected.")
    return np.clip(x, -CLIP_Z, CLIP_Z).astype(np.float32), s, f


def path_turnover_torch(h: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(h[:, 0])
        + torch.sum(torch.abs(h[:, 1:] - h[:, :-1]), dim=1)
        + torch.abs(h[:, -1])
    )


def path_net_pnl_torch(
    spot: torch.Tensor, fut: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gross = torch.sum(spot - h * fut, dim=1)
    turnover = path_turnover_torch(h)
    net = gross - LAMBDA_TC * turnover
    return net, turnover


def empirical_cvar_torch(losses: torch.Tensor) -> torch.Tensor:
    k = max(1, int(math.ceil(TAIL_FRACTION * losses.numel())))
    worst, _ = torch.topk(losses, k=k, largest=True, sorted=False)
    return torch.mean(worst)


def objective(spot: torch.Tensor, fut: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    net, _ = path_net_pnl_torch(spot, fut, h)
    return empirical_cvar_torch(-net) * LOSS_SCALE


def make_loader(
    x: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(s), torch.from_numpy(f))
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    dummy = np.zeros(x.shape[:2], dtype=np.float32)
    loader = make_loader(x, dummy, dummy, shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in loader:
            chunks.append(model(xb.to(device, non_blocking=True)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def cvar_numpy(losses: np.ndarray) -> float:
    x = np.asarray(losses, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(x))))
    idx = np.argpartition(x, len(x) - k)[len(x) - k :]
    return float(np.mean(x[idx]))


def np_turnover(h: np.ndarray) -> np.ndarray:
    h64 = h.astype(np.float64)
    return (
        np.abs(h64[:, 0])
        + np.sum(np.abs(h64[:, 1:] - h64[:, :-1]), axis=1)
        + np.abs(h64[:, -1])
    )


def summarize(name: str, spot: np.ndarray, fut: np.ndarray, h: np.ndarray) -> dict:
    s = spot.astype(np.float64)
    f = fut.astype(np.float64)
    h = h.astype(np.float64)
    daily = s - h * f
    gross = np.sum(daily, axis=1)
    turn = np_turnover(h)
    net = gross - LAMBDA_TC * turn
    losses = -net
    return {
        "strategy": name,
        "gross_daily_std": float(np.std(daily.reshape(-1), ddof=1)),
        "gross_terminal_mean": float(np.mean(gross)),
        "gross_terminal_std": float(np.std(gross, ddof=1)),
        "net_terminal_mean": float(np.mean(net)),
        "net_terminal_std": float(np.std(net, ddof=1)),
        "net_terminal_q01": float(np.quantile(net, 0.01)),
        "net_terminal_q05": float(np.quantile(net, 0.05)),
        "cvar_loss_worst_5pct": cvar_numpy(losses),
        "turnover_mean_open_rebalance_close": float(np.mean(turn)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
        "hedge_q01": float(np.quantile(h, 0.01)),
        "hedge_q05": float(np.quantile(h, 0.05)),
        "hedge_q50": float(np.quantile(h, 0.50)),
        "hedge_q95": float(np.quantile(h, 0.95)),
        "hedge_q99": float(np.quantile(h, 0.99)),
        "fraction_h_lt_0p05": float(np.mean(h < 0.05)),
        "fraction_h_gt_1p45": float(np.mean(h > 1.45)),
    }


def real_train_static_minvar(raw_train_path: Path) -> float:
    df = pd.read_csv(raw_train_path)
    s = np.log(df["spot_close"].to_numpy(dtype=float))
    f = np.log(df["fut_close"].to_numpy(dtype=float))
    sr = np.diff(s)
    fr = np.diff(f)
    var_f = float(np.var(fr, ddof=1))
    return float(np.cov(sr, fr, ddof=1)[0, 1] / var_f)


def full_validation_cvar_scaled(
    model: nn.Module,
    x: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    device: torch.device,
) -> float:
    h = predict(model, x, device)
    turn = np_turnover(h)
    net = np.sum(s.astype(np.float64) - h.astype(np.float64) * f.astype(np.float64), axis=1) - LAMBDA_TC * turn
    return cvar_numpy(-net) * LOSS_SCALE


def main() -> None:
    seed_everything(SEED)

    root = Path(__file__).resolve().parents[2]
    real_dir = root / "data" / "processed" / "deep_hedging_real"
    pretrained_path = root / "results" / "models" / "lstm_cvar_turnover_best.pt"
    raw_train_path = root / "data" / "processed" / "splits" / "train.csv"
    models_dir = root / "results" / "models"
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    for d in (models_dir, tables_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    x_train, s_train, f_train = load_sequences(real_dir / "train_sequences.npz")
    x_val, s_val, f_val = load_sequences(real_dir / "validation_sequences.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    model = LSTMHedgePolicy(
        n_features=int(checkpoint["n_features"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        hedge_max=float(checkpoint["hedge_max"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if x_train.shape[2] != int(checkpoint["n_features"]):
        raise ValueError("Real feature count differs from pretrained checkpoint.")

    pretrained_state = copy.deepcopy(model.state_dict())
    pretrained_val_cvar_scaled = full_validation_cvar_scaled(model, x_val, s_val, f_val, device)

    # Epoch 0 is deliberately eligible as the final model: this prevents forced
    # negative transfer when real fine-tuning does not improve validation CVaR.
    best_val = pretrained_val_cvar_scaled
    best_epoch = 0
    best_state = copy.deepcopy(pretrained_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, s_train, f_train, shuffle=True)
    bad_epochs = 0
    history: list[dict] = [
        {
            "epoch": 0,
            "train_batch_cvar_scaled_mean": np.nan,
            "validation_full_cvar_scaled": pretrained_val_cvar_scaled,
            "best_so_far": True,
        }
    ]

    print("=" * 116)
    print("REAL-DATA FINE-TUNING OF SYNTHETIC-PRETRAINED LSTM — CVAR + TURNOVER")
    print("=" * 116)
    print(f"Device                 : {device}")
    print(f"Real train episodes    : {x_train.shape[0]:,}")
    print(f"Real validation episodes: {x_val.shape[0]:,}")
    print(f"Input clipping         : +/-{CLIP_Z:.1f} z (features only)")
    print(f"Learning rate          : {LEARNING_RATE}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"lambda_tc              : {LAMBDA_TC:.6f} normalized penalty")
    print(f"Tail fraction          : {TAIL_FRACTION:.2%}")
    print(f"Epoch-0 validation CVaR: {pretrained_val_cvar_scaled:.6f} scaled")
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
        val_loss = full_validation_cvar_scaled(model, x_val, s_val, f_val, device)
        improved = val_loss < best_val - 1e-5

        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
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
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | train_batch_CVaR={train_loss:.6f} | "
            f"val_full_CVaR={val_loss:.6f} | best={best_val:.6f} @ {best_epoch}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without validation improvement.")
            break

    model.load_state_dict(best_state)
    runtime = time.time() - start

    best_path = models_dir / "lstm_cvar_turnover_real_finetuned_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": int(checkpoint["n_features"]),
            "hidden_size": int(checkpoint["hidden_size"]),
            "num_layers": int(checkpoint["num_layers"]),
            "hedge_max": float(checkpoint["hedge_max"]),
            "tail_fraction": TAIL_FRACTION,
            "lambda_tc": LAMBDA_TC,
            "clip_z": CLIP_Z,
            "learning_rate": LEARNING_RATE,
            "best_epoch": best_epoch,
            "best_validation_cvar_scaled": best_val,
            "pretrained_validation_cvar_scaled": pretrained_val_cvar_scaled,
            "seed": SEED,
            "torch_version": torch.__version__,
        },
        best_path,
    )

    # Restore the untouched pretrained state for direct zero-shot comparison.
    pretrained_model = LSTMHedgePolicy(
        int(checkpoint["n_features"]),
        int(checkpoint["hidden_size"]),
        int(checkpoint["num_layers"]),
        float(checkpoint["hedge_max"]),
    ).to(device)
    pretrained_model.load_state_dict(pretrained_state)

    static_h_value = real_train_static_minvar(raw_train_path)
    comparison_rows: list[dict] = []
    split_details: dict[str, dict] = {}

    for split_name, x, s, f in (
        ("train", x_train, s_train, f_train),
        ("validation", x_val, s_val, f_val),
    ):
        h_pre = predict(pretrained_model, x, device)
        h_ft = predict(model, x, device)
        h_delta = np.ones_like(s, dtype=np.float32)
        h_static = np.full_like(s, static_h_value, dtype=np.float32)

        rows = [
            summarize("delta_1to1", s, f, h_delta),
            summarize("real_train_static_minvar", s, f, h_static),
            summarize("pretrained_clip5_zero_shot", s, f, h_pre),
            summarize("real_finetuned_lstm", s, f, h_ft),
        ]
        for row in rows:
            row["split"] = split_name
            comparison_rows.append(row)

        split_details[split_name] = {
            "pretrained": next(r for r in rows if r["strategy"] == "pretrained_clip5_zero_shot"),
            "finetuned": next(r for r in rows if r["strategy"] == "real_finetuned_lstm"),
            "mean_abs_hedge_change_pretrained_to_finetuned": float(
                np.mean(np.abs(h_pre.astype(np.float64) - h_ft.astype(np.float64)))
            ),
        }

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = tables_dir / "real_finetune_validation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = tables_dir / "real_finetune_training_history.csv"
    history_df.to_csv(history_path, index=False)

    fig = plt.figure(figsize=(10, 5))
    hist_plot = history_df.dropna(subset=["validation_full_cvar_scaled"])
    plt.plot(hist_plot["epoch"], hist_plot["validation_full_cvar_scaled"], label="real validation CVaR")
    train_plot = history_df.dropna(subset=["train_batch_cvar_scaled_mean"])
    plt.plot(train_plot["epoch"], train_plot["train_batch_cvar_scaled_mean"], label="real train batch CVaR")
    plt.axvline(best_epoch, linewidth=1.0, label=f"best epoch={best_epoch}")
    plt.xlabel("Fine-tune epoch")
    plt.ylabel("Scaled CVaR objective")
    plt.title("Real Fine-tuning of Synthetic-Pretrained LSTM")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "real_finetune_cvar_training_curve.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    val_pre = split_details["validation"]["pretrained"]
    val_ft = split_details["validation"]["finetuned"]
    sanity = {
        "validation_cvar_not_worse_than_pretrained": bool(
            val_ft["cvar_loss_worst_5pct"] <= val_pre["cvar_loss_worst_5pct"] + 1e-12
        ),
        "mean_hedge_between_0.50_and_1.20": bool(0.50 <= val_ft["hedge_ratio_mean"] <= 1.20),
        "no_low_saturation": bool(val_ft["fraction_h_lt_0p05"] < 0.01),
        "no_high_saturation": bool(val_ft["fraction_h_gt_1p45"] < 0.01),
        "finite_turnover": bool(np.isfinite(val_ft["turnover_mean_open_rebalance_close"])),
    }

    report = {
        "model": "synthetic-pretrained LSTM fine-tuned on real VN30/VN30F1M",
        "initial_checkpoint": str(pretrained_path),
        "best_checkpoint": str(best_path),
        "input_policy": "clip synthetic-scaled features to [-5,5]; do not modify returns",
        "real_train_episodes": int(x_train.shape[0]),
        "real_validation_episodes": int(x_val.shape[0]),
        "episodes_overlap_stride1": True,
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "tail_fraction": TAIL_FRACTION,
            "lambda_tc": LAMBDA_TC,
            "best_epoch": best_epoch,
            "epoch0_pretrained_validation_cvar_scaled": pretrained_val_cvar_scaled,
            "best_validation_cvar_scaled": best_val,
            "runtime_seconds": runtime,
        },
        "real_train_static_minvar_ratio": static_h_value,
        "results": split_details,
        "sanity_checks": sanity,
        "all_sanity_checks_pass": bool(all(sanity.values())),
        "fine_tuning_improved_over_epoch0": bool(best_epoch > 0),
        "real_test_used": False,
        "note": (
            "REAL VALIDATION has been used for transfer/input-policy selection and checkpoint selection. "
            "REAL TEST remains sealed for final OOS evaluation."
        ),
    }
    report_path = metrics_dir / "real_finetune_lstm_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print()
    print("=" * 116)
    print("REAL-DATA LSTM FINE-TUNING COMPLETE")
    print("=" * 116)
    print(f"Best epoch                       : {best_epoch}")
    print(f"Epoch-0 pretrained val CVaR     : {pretrained_val_cvar_scaled:.6f} scaled")
    print(f"Best real-validation CVaR       : {best_val:.6f} scaled")
    print(f"Real TRAIN static min-var h*    : {static_h_value:.6f}")
    print(f"Runtime                          : {runtime:.1f} sec")
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
    print(
        "Validation mean |delta h| pretrained -> fine-tuned: "
        f"{split_details['validation']['mean_abs_hedge_change_pretrained_to_finetuned']:.6f}"
    )
    print()
    for name, passed in sanity.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL SANITY CHECKS PASS        : {all(sanity.values())}")
    print(f"FINE-TUNING IMPROVED EPOCH 0  : {best_epoch > 0}")
    print()
    print(f"Best model       : {best_path}")
    print(f"Training history : {history_path}")
    print(f"Comparison table : {comparison_path}")
    print(f"Report           : {report_path}")
    print(f"Training figure  : {figure_path}")
    print("REAL TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
