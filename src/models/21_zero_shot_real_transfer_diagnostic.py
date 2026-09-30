"""Zero-shot diagnostic of the pretrained Deep Hedging LSTM on REAL data.

Purpose
-------
Before fine-tuning, compare the step-18 synthetic-pretrained checkpoint on real
TRAIN/VALIDATION under two neural-network input modes:
1) raw synthetic-scaled real features;
2) the same features clipped to [-5, 5].

Clipping affects only the neural-network INPUT. Real spot/futures returns used to
compute hedging PnL are never clipped, winsorized, rescaled, or otherwise changed.
This lets us test whether bounded input-domain adaptation stabilizes transfer
without deleting real market stress from the hedging objective.

Important
---------
- Uses real TRAIN and real VALIDATION only.
- Real TEST (2025-2026) is never opened.
- Episodes overlap with stride 1, so episode-level risk metrics are diagnostic,
  not independent-sample inference.
- lambda_tc=0.001 is the same normalized turnover regularizer as step 18 and is
  NOT an official VN30F1M transaction-cost rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLIP_Z = 5.0
BATCH_SIZE = 512
TAIL_FRACTION = 0.05
LAMBDA_TC = 0.001
HEDGE_MAX = 1.5


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


def load_real_sequences(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run step 19 first.")
    with np.load(path) as z:
        x = z["features"].astype(np.float32)
        s = z["spot_forward_log_return"].astype(np.float32)
        f = z["fut_forward_log_return"].astype(np.float32)
    if x.ndim != 3 or s.shape != x.shape[:2] or f.shape != x.shape[:2]:
        raise ValueError(f"{path.name}: inconsistent shapes.")
    return x, s, f


def make_loader(x: np.ndarray) -> DataLoader:
    dummy = np.zeros(x.shape[:2], dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(dummy), torch.from_numpy(dummy))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in make_loader(x):
            h = model(xb.to(device))
            chunks.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def turnover(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64)
    return (
        np.abs(h[:, 0])
        + np.sum(np.abs(h[:, 1:] - h[:, :-1]), axis=1)
        + np.abs(h[:, -1])
    )


def cvar(losses: np.ndarray, tail_fraction: float) -> float:
    x = np.asarray(losses, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(tail_fraction * len(x))))
    idx = np.argpartition(x, len(x) - k)[len(x) - k :]
    return float(np.mean(x[idx]))


def summarize(name: str, spot: np.ndarray, fut: np.ndarray, h: np.ndarray) -> dict:
    s = spot.astype(np.float64)
    f = fut.astype(np.float64)
    h = h.astype(np.float64)

    daily = s - h * f
    gross_terminal = np.sum(daily, axis=1)
    path_turnover = turnover(h)
    net_terminal = gross_terminal - LAMBDA_TC * path_turnover
    losses = -net_terminal

    return {
        "strategy": name,
        "gross_daily_std": float(np.std(daily.reshape(-1), ddof=1)),
        "gross_terminal_mean": float(np.mean(gross_terminal)),
        "gross_terminal_std": float(np.std(gross_terminal, ddof=1)),
        "net_terminal_mean": float(np.mean(net_terminal)),
        "net_terminal_std": float(np.std(net_terminal, ddof=1)),
        "net_terminal_q05": float(np.quantile(net_terminal, 0.05)),
        "cvar_loss_worst_5pct": cvar(losses, TAIL_FRACTION),
        "turnover_mean_open_rebalance_close": float(np.mean(path_turnover)),
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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    real_dir = root / "data" / "processed" / "deep_hedging_real"
    model_path = root / "results" / "models" / "lstm_cvar_turnover_best.pt"
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run src/models/18_train_lstm_cvar_turnover.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_features = int(checkpoint["n_features"])
    hidden_size = int(checkpoint["hidden_size"])
    num_layers = int(checkpoint["num_layers"])
    hedge_max = float(checkpoint.get("hedge_max", HEDGE_MAX))

    model = LSTMHedgePolicy(n_features, hidden_size, num_layers, hedge_max).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows: list[dict] = []
    details: dict[str, dict] = {}

    for split in ("train", "validation"):
        x, spot, fut = load_real_sequences(real_dir / f"{split}_sequences.npz")
        if x.shape[2] != n_features:
            raise ValueError(f"{split}: feature count differs from pretrained checkpoint.")

        x_raw = x
        x_clip = np.clip(x, -CLIP_Z, CLIP_Z).astype(np.float32)

        h_raw = predict(model, x_raw, device)
        h_clip = predict(model, x_clip, device)
        h_delta = np.ones_like(spot, dtype=np.float32)

        split_rows = [
            summarize("delta_1to1", spot, fut, h_delta),
            summarize("pretrained_raw_input", spot, fut, h_raw),
            summarize("pretrained_clip5_input", spot, fut, h_clip),
        ]
        for row in split_rows:
            row["split"] = split
            rows.append(row)

        changed_fraction = float(np.mean(np.any(np.abs(x) > CLIP_Z, axis=2)))
        element_fraction = float(np.mean(np.abs(x) > CLIP_Z))
        mean_abs_h_diff = float(np.mean(np.abs(h_raw.astype(np.float64) - h_clip.astype(np.float64))))
        max_abs_h_diff = float(np.max(np.abs(h_raw.astype(np.float64) - h_clip.astype(np.float64))))

        details[split] = {
            "episode_time_fraction_with_any_feature_clipped": changed_fraction,
            "feature_element_fraction_clipped": element_fraction,
            "mean_abs_hedge_difference_raw_vs_clip5": mean_abs_h_diff,
            "max_abs_hedge_difference_raw_vs_clip5": max_abs_h_diff,
            "raw_metrics": next(r for r in split_rows if r["strategy"] == "pretrained_raw_input"),
            "clip5_metrics": next(r for r in split_rows if r["strategy"] == "pretrained_clip5_input"),
        }

    df = pd.DataFrame(rows)
    table_path = tables_dir / "zero_shot_real_transfer_raw_vs_clip5.csv"
    df.to_csv(table_path, index=False)

    val_raw = details["validation"]["raw_metrics"]
    val_clip = details["validation"]["clip5_metrics"]

    recommendation = {
        "clip5_not_worse_cvar": bool(
            val_clip["cvar_loss_worst_5pct"] <= val_raw["cvar_loss_worst_5pct"] * 1.02
        ),
        "clip5_not_higher_turnover": bool(
            val_clip["turnover_mean_open_rebalance_close"]
            <= val_raw["turnover_mean_open_rebalance_close"] * 1.05
        ),
        "clip5_no_output_saturation": bool(
            val_clip["fraction_h_lt_0p05"] < 0.01 and val_clip["fraction_h_gt_1p45"] < 0.01
        ),
    }
    use_clip5_for_finetune = bool(all(recommendation.values()))

    report = {
        "purpose": "zero_shot_pretrained_transfer_diagnostic",
        "checkpoint": str(model_path),
        "device": str(device),
        "clip_threshold_z": CLIP_Z,
        "clip_applied_to": "neural_network_input_only",
        "real_returns_modified": False,
        "lambda_tc": LAMBDA_TC,
        "lambda_tc_is_official_market_fee": False,
        "overlapping_real_episodes": True,
        "results": details,
        "decision_checks": recommendation,
        "recommended_use_clip5_for_finetune": use_clip5_for_finetune,
        "real_test_used": False,
    }
    report_path = metrics_dir / "zero_shot_real_transfer_diagnostic.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 124)
    print("ZERO-SHOT SYNTHETIC->REAL TRANSFER DIAGNOSTIC COMPLETE")
    print("=" * 124)
    print(f"Device              : {device}")
    print(f"Input clip threshold: +/-{CLIP_Z:.1f} z")
    print("Real PnL returns     : unchanged")
    print()
    print(
        df[
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
                "fraction_h_lt_0p05",
                "fraction_h_gt_1p45",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("RAW vs CLIP5 input sensitivity:")
    for split in ("train", "validation"):
        d = details[split]
        print(
            f"{split:10s} | episode-time clipped={d['episode_time_fraction_with_any_feature_clipped']:.4%} | "
            f"elements clipped={d['feature_element_fraction_clipped']:.4%} | "
            f"mean |delta h|={d['mean_abs_hedge_difference_raw_vs_clip5']:.6f} | "
            f"max |delta h|={d['max_abs_hedge_difference_raw_vs_clip5']:.6f}"
        )
    print()
    for name, passed in recommendation.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"RECOMMEND CLIP5 FOR FINE-TUNE : {use_clip5_for_finetune}")
    print()
    print(f"Comparison table : {table_path}")
    print(f"JSON report      : {report_path}")
    print("REAL TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
