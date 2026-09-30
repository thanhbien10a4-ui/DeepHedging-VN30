"""Post-freeze market-regime analysis on REAL TEST.

Purpose
-------
Evaluate the already frozen hedging strategies across causal VN30 market
regimes: bear / sideways / bull. This is descriptive TEST analysis only and MUST
NOT be used to retune the model.

Causal regime definition
------------------------
For each 60-day TEST episode starting at date t, compute the VN30 trailing
60-trading-day log return using ONLY observations strictly before t.

Thresholds are frozen from REAL TRAIN only:
    bear      : trailing_60d_return < train 33.33rd percentile
    sideways  : between train 33.33rd and 66.67th percentiles
    bull      : trailing_60d_return > train 66.67th percentile

Thus TEST outcomes never determine the regime threshold, and no forward episode
return is used to label its starting regime.

Evaluation
----------
Uses economic index-point PnL for one continuous contract-equivalent exposure:
    pnl_t = 100,000 * [(S_{t+1}-S_t) - h_t(F_{t+1}-F_t)]

No fee/slippage is applied here so this step isolates regime sensitivity. Cost
robustness is already analyzed separately in step 24.

Episodes overlap with stride 1, so regime statistics are descriptive rather than
independent-sample inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPISODE_STEPS = 60
REGIME_LOOKBACK = 60
Q_LOW = 1.0 / 3.0
Q_HIGH = 2.0 / 3.0
TAIL_FRACTION = 0.05
BATCH_SIZE = 512


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


def load_model(path: Path, device: torch.device) -> LSTMHedgePolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMHedgePolicy(
        int(ckpt["n_features"]),
        int(ckpt["hidden_size"]),
        int(ckpt["num_layers"]),
        float(ckpt["hedge_max"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    dummy = np.zeros(x.shape[:2], dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(dummy), torch.from_numpy(dummy))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in dl:
            out.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def minvar_ratio(sr: np.ndarray, fr: np.ndarray) -> float:
    var_f = float(np.var(fr, ddof=1))
    if var_f <= 0:
        raise ValueError("Non-positive futures variance")
    return float(np.cov(sr, fr, ddof=1)[0, 1] / var_f)


def build_causal_rolling60(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    starts: np.ndarray,
) -> np.ndarray:
    pre = pd.concat(
        [
            train[["time", "spot_close", "fut_close"]],
            validation[["time", "spot_close", "fut_close"]],
        ],
        ignore_index=True,
    )
    combined = pd.concat(
        [pre, test[["time", "spot_close", "fut_close"]]], ignore_index=True
    ).sort_values("time").reset_index(drop=True)
    combined["sr"] = np.log(combined["spot_close"]).diff()
    combined["fr"] = np.log(combined["fut_close"]).diff()

    first_test_idx = len(pre)
    h_day = np.empty(len(test), dtype=np.float64)
    for j in range(len(test)):
        idx = first_test_idx + j
        hist = combined.iloc[max(0, idx - 60) : idx][["sr", "fr"]].dropna()
        if len(hist) < 20:
            raise ValueError("Insufficient causal history")
        h_day[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())

    h = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        h[i] = h_day[start : start + EPISODE_STEPS]
    return h


def cvar_loss(pnl: np.ndarray) -> float:
    losses = -np.asarray(pnl, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k :]
    return float(np.mean(losses[idx]))


def train_regime_thresholds(train: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    close = train["spot_close"].to_numpy(dtype=np.float64)
    trailing = np.log(close[REGIME_LOOKBACK:] / close[:-REGIME_LOOKBACK])
    low = float(np.quantile(trailing, Q_LOW))
    high = float(np.quantile(trailing, Q_HIGH))
    return low, high, trailing


def label_test_regimes(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    starts: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    pre = pd.concat(
        [train[["time", "spot_close"]], validation[["time", "spot_close"]]],
        ignore_index=True,
    )
    combined = pd.concat([pre, test[["time", "spot_close"]]], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    first_test_idx = len(pre)

    trailing = np.empty(len(starts), dtype=np.float64)
    labels = np.empty(len(starts), dtype=object)

    for i, start in enumerate(starts):
        idx = first_test_idx + int(start)
        if idx - REGIME_LOOKBACK < 0:
            raise ValueError("Insufficient history for causal regime label")
        p0 = float(combined.loc[idx - REGIME_LOOKBACK, "spot_close"])
        p1 = float(combined.loc[idx - 1, "spot_close"])
        r = math.log(p1 / p0)
        trailing[i] = r
        if r < low:
            labels[i] = "bear"
        elif r > high:
            labels[i] = "bull"
        else:
            labels[i] = "sideways"

    return labels, trailing


def build_price_change_episodes(test: pd.DataFrame, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = test["spot_close"].to_numpy(dtype=np.float64)
    f = test["fut_close"].to_numpy(dtype=np.float64)
    ds = np.empty((len(starts), EPISODE_STEPS), dtype=np.float64)
    df = np.empty_like(ds)
    for i, start in enumerate(starts):
        ds[i] = np.diff(s[start : start + EPISODE_STEPS + 1])
        df[i] = np.diff(f[start : start + EPISODE_STEPS + 1])
    return ds, df


def summarize_regime(
    strategy: str,
    regime: str,
    pnl: np.ndarray,
    h: np.ndarray,
    trailing: np.ndarray,
) -> dict:
    return {
        "strategy": strategy,
        "regime": regime,
        "episodes": int(len(pnl)),
        "mean_prior_60d_log_return": float(np.mean(trailing)),
        "mean_prior_60d_simple_return": float(np.mean(np.exp(trailing) - 1.0)),
        "gross_mean_vnd": float(np.mean(pnl)),
        "gross_std_vnd": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan"),
        "gross_q05_vnd": float(np.quantile(pnl, 0.05)),
        "gross_cvar_loss_5pct_vnd": cvar_loss(pnl),
        "positive_pnl_fraction": float(np.mean(pnl > 0)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)) if h.size > 1 else float("nan"),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with open(root / "configs" / "vn30f1m_market_friction.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    multiplier = float(config["market"]["contract_multiplier_vnd_per_index_point"])

    splits = root / "data" / "processed" / "splits"
    train = pd.read_csv(splits / "train.csv")
    validation = pd.read_csv(splits / "validation.csv")
    test = pd.read_csv(splits / "test.csv")
    for frame in (train, validation, test):
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    seq_path = root / "data" / "processed" / "deep_hedging_real" / "test_sequences_final_oos.npz"
    with np.load(seq_path) as z:
        x_test = z["features"].astype(np.float32)
        starts = z["episode_start_index"].astype(int)

    low, high, train_trailing = train_regime_thresholds(train)
    labels, test_trailing = label_test_regimes(train, validation, test, starts, low, high)

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(dtype=np.float64)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(dtype=np.float64)))
    static_h = minvar_ratio(train_sr, train_fr)

    h_unhedged = np.zeros((len(starts), EPISODE_STEPS), dtype=np.float32)
    h_delta = np.ones_like(h_unhedged)
    h_static = np.full_like(h_unhedged, static_h)
    h_rolling = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_model = load_model(root / "results" / "models" / "lstm_cvar_turnover_best.pt", device)
    final_model = load_model(
        root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt", device
    )
    h_pre = predict(pre_model, x_test, device)
    h_final = predict(final_model, x_test, device)

    strategy_map = {
        "unhedged": h_unhedged,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    ds, df = build_price_change_episodes(test, starts)
    rows: list[dict] = []

    for strategy, h in strategy_map.items():
        pnl = multiplier * np.sum(ds - h.astype(np.float64) * df, axis=1)
        for regime in ("bear", "sideways", "bull"):
            mask = labels == regime
            if not np.any(mask):
                continue
            rows.append(
                summarize_regime(
                    strategy,
                    regime,
                    pnl[mask],
                    h[mask],
                    test_trailing[mask],
                )
            )

    result = pd.DataFrame(rows)
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    for d in (tables_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / "market_regime_oos_frozen.csv"
    result.to_csv(table_path, index=False)

    counts = {regime: int(np.sum(labels == regime)) for regime in ("bear", "sideways", "bull")}
    final_vs_delta: dict[str, dict] = {}
    for regime in ("bear", "sideways", "bull"):
        sub = result[result["regime"] == regime]
        final_row = sub[sub["strategy"] == "real_finetuned_lstm"].iloc[0]
        delta_row = sub[sub["strategy"] == "delta_1to1"].iloc[0]
        final_vs_delta[regime] = {
            "cvar_reduction_fraction": float(
                1.0 - final_row["gross_cvar_loss_5pct_vnd"] / delta_row["gross_cvar_loss_5pct_vnd"]
            ),
            "mean_pnl_difference_vnd": float(final_row["gross_mean_vnd"] - delta_row["gross_mean_vnd"]),
        }

    report = {
        "status": "POST_FREEZE_DESCRIPTIVE_REGIME_ANALYSIS_ONLY",
        "model_retuned": False,
        "regime_definition": {
            "lookback_trading_days": REGIME_LOOKBACK,
            "threshold_source": "real TRAIN trailing-60d VN30 log-return distribution",
            "train_q33_log_return": low,
            "train_q67_log_return": high,
            "train_q33_simple_return": math.exp(low) - 1.0,
            "train_q67_simple_return": math.exp(high) - 1.0,
            "causal": True,
            "test_forward_returns_used_for_label": False,
        },
        "test_regime_episode_counts": counts,
        "final_model_vs_delta": final_vs_delta,
        "results": rows,
        "warning": (
            "TEST episodes overlap with stride 1; results are descriptive and must not be used to retune the model."
        ),
    }
    report_path = metrics_dir / "market_regime_oos_frozen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(11, 5))
    regimes = ["bear", "sideways", "bull"]
    strategies = ["delta_1to1", "causal_rolling60_minvar", "synthetic_pretrained_lstm", "real_finetuned_lstm"]
    x = np.arange(len(regimes), dtype=float)
    width = 0.2
    for j, strategy in enumerate(strategies):
        vals = []
        for regime in regimes:
            row = result[(result["strategy"] == strategy) & (result["regime"] == regime)]
            vals.append(float(row.iloc[0]["gross_cvar_loss_5pct_vnd"]))
        plt.bar(x + (j - 1.5) * width, vals, width=width, label=strategy)
    plt.xticks(x, regimes)
    plt.ylabel("Gross 60-day CVaR loss (VND)")
    plt.title("Frozen OOS Hedging Tail Risk by Causal Market Regime")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "market_regime_oos_cvar.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 128)
    print("MARKET REGIME ANALYSIS COMPLETE — FROZEN MODELS, CAUSAL LABELS, NO RETUNING")
    print("=" * 128)
    print(f"Regime lookback                 : {REGIME_LOOKBACK} trading days")
    print(f"TRAIN q33 threshold             : log={low:+.6f} | simple={math.exp(low)-1:+.2%}")
    print(f"TRAIN q67 threshold             : log={high:+.6f} | simple={math.exp(high)-1:+.2%}")
    print(f"TEST episode counts             : bear={counts['bear']}, sideways={counts['sideways']}, bull={counts['bull']}")
    print("Regime at episode start uses only history strictly before that start date.")
    print()

    display = result[
        [
            "regime",
            "strategy",
            "episodes",
            "gross_mean_vnd",
            "gross_std_vnd",
            "gross_q05_vnd",
            "gross_cvar_loss_5pct_vnd",
            "positive_pnl_fraction",
            "hedge_ratio_mean",
        ]
    ].sort_values(["regime", "strategy"])
    print(display.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    print()
    print("FINAL FROZEN MODEL VS DELTA 1:1 BY REGIME")
    for regime, vals in final_vs_delta.items():
        print(
            f"{regime:8s} | CVaR reduction={vals['cvar_reduction_fraction']:+.2%} | "
            f"mean PnL diff={vals['mean_pnl_difference_vnd']:+,.0f} VND"
        )

    print()
    print("IMPORTANT: post-freeze TEST analysis only; DO NOT RETUNE from these regime results.")
    print(f"Table  : {table_path}")
    print(f"Report : {report_path}")
    print(f"Figure : {figure_path}")


if __name__ == "__main__":
    main()
