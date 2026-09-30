"""Create final frozen OOS cumulative-performance and drawdown reporting artifacts.

Post-freeze descriptive TEST reporting only. No model, threshold, feature, or
hyperparameter is changed.

Why path-wise cumulative curves?
--------------------------------
The Deep Hedging policy was trained/evaluated on 60-trading-day episodes with an
explicit time-remaining state. A single stitched chronological wealth curve
would require inventing a deployment rule for resetting/restarting the LSTM and
its episode clock. To avoid that unsupported choice, this script reports the
mean cumulative normalized PnL trajectory WITHIN the existing frozen 60-day OOS
episodes, plus path-wise drawdown distributions.

Economic PnL convention
-----------------------
For one continuous contract-equivalent exposure:
    daily_pnl_vnd = multiplier * [(S_{t+1}-S_t) - h_t(F_{t+1}-F_t)]
Each episode is normalized by its initial VN30 notional:
    initial_notional_vnd = S_start * multiplier
    daily_return_proxy = daily_pnl_vnd / initial_notional_vnd
The cumulative return proxy is additive PnL divided by initial notional, not a
compounded cash-equity return. This is appropriate for linear futures hedge PnL
reporting and avoids the earlier log-return/cumprod inconsistency.

Overlap warning
---------------
The 370 TEST episodes overlap with stride 1. Sharpe/drawdown summaries are
therefore descriptive and are not independent-sample inference.
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
BATCH_SIZE = 512
TRADING_DAYS = 252.0
TAIL_FRACTION = 0.05


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
        y, _ = self.lstm(x)
        return self.hedge_max * torch.sigmoid(self.head(y).squeeze(-1))


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
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in dl:
            chunks.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


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

    first_test = len(pre)
    h_day = np.empty(len(test), dtype=np.float64)
    for j in range(len(test)):
        hist = combined.iloc[max(0, first_test + j - 60) : first_test + j][["sr", "fr"]].dropna()
        if len(hist) < 20:
            raise ValueError("Insufficient causal history for rolling min-var")
        h_day[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())

    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = h_day[start : start + EPISODE_STEPS]
    return out


def price_change_episodes(test: pd.DataFrame, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = test["spot_close"].to_numpy(dtype=np.float64)
    f = test["fut_close"].to_numpy(dtype=np.float64)
    ds = np.empty((len(starts), EPISODE_STEPS), dtype=np.float64)
    df = np.empty_like(ds)
    for i, start in enumerate(starts):
        ds[i] = np.diff(s[start : start + EPISODE_STEPS + 1])
        df[i] = np.diff(f[start : start + EPISODE_STEPS + 1])
    return ds, df


def cvar_loss(x: np.ndarray) -> float:
    losses = -np.asarray(x, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k :]
    return float(np.mean(losses[idx]))


def path_max_drawdown(cumulative_return: np.ndarray) -> np.ndarray:
    # Prepend zero so drawdown from initial capital is represented correctly.
    z = np.concatenate(
        [np.zeros((cumulative_return.shape[0], 1), dtype=np.float64), cumulative_return],
        axis=1,
    )
    running_peak = np.maximum.accumulate(z, axis=1)
    drawdown = z - running_peak
    return np.min(drawdown, axis=1)


def summarize(
    strategy: str,
    h: np.ndarray,
    ds: np.ndarray,
    df: np.ndarray,
    initial_notional: np.ndarray,
    multiplier: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    daily_pnl_vnd = multiplier * (ds - h.astype(np.float64) * df)
    daily_ret = daily_pnl_vnd / initial_notional[:, None]
    cumulative_ret = np.cumsum(daily_ret, axis=1)
    terminal_vnd = np.sum(daily_pnl_vnd, axis=1)
    terminal_ret = cumulative_ret[:, -1]
    mdd = path_max_drawdown(cumulative_ret)

    flat_daily = daily_ret.reshape(-1)
    daily_std = float(np.std(flat_daily, ddof=1))
    pooled_sharpe = (
        float(np.mean(flat_daily) / daily_std * math.sqrt(TRADING_DAYS))
        if daily_std > 0
        else float("nan")
    )

    row = {
        "strategy": strategy,
        "episodes": int(len(terminal_vnd)),
        "terminal_mean_vnd": float(np.mean(terminal_vnd)),
        "terminal_std_vnd": float(np.std(terminal_vnd, ddof=1)),
        "terminal_q05_vnd": float(np.quantile(terminal_vnd, 0.05)),
        "terminal_cvar_loss_5pct_vnd": cvar_loss(terminal_vnd),
        "terminal_mean_return_proxy": float(np.mean(terminal_ret)),
        "terminal_q05_return_proxy": float(np.quantile(terminal_ret, 0.05)),
        "pooled_daily_mean_return_proxy": float(np.mean(flat_daily)),
        "pooled_daily_std_return_proxy": daily_std,
        "pooled_daily_sharpe_zero_rf": pooled_sharpe,
        "mean_episode_max_drawdown": float(np.mean(mdd)),
        "median_episode_max_drawdown": float(np.median(mdd)),
        "q05_episode_max_drawdown": float(np.quantile(mdd, 0.05)),
        "worst_episode_max_drawdown": float(np.min(mdd)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
    }
    return row, cumulative_ret, mdd


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
    if not seq_path.exists():
        raise FileNotFoundError("Run step 23 first")
    with np.load(seq_path) as z:
        x_test = z["features"].astype(np.float32)
        starts = z["episode_start_index"].astype(int)

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(dtype=np.float64)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(dtype=np.float64)))
    static_h = minvar_ratio(train_sr, train_fr)

    shape = (len(starts), EPISODE_STEPS)
    h_zero = np.zeros(shape, dtype=np.float32)
    h_delta = np.ones(shape, dtype=np.float32)
    h_static = np.full(shape, static_h, dtype=np.float32)
    h_rolling = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_pre = predict(
        load_model(root / "results" / "models" / "lstm_cvar_turnover_best.pt", device),
        x_test,
        device,
    )
    h_final = predict(
        load_model(
            root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt",
            device,
        ),
        x_test,
        device,
    )

    strategies = {
        "unhedged": h_zero,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    ds, df = price_change_episodes(test, starts)
    start_spot = test["spot_close"].to_numpy(dtype=np.float64)[starts]
    initial_notional = start_spot * multiplier

    rows: list[dict] = []
    cumulative: dict[str, np.ndarray] = {}
    drawdowns: dict[str, np.ndarray] = {}
    for name, h in strategies.items():
        row, curve, mdd = summarize(name, h, ds, df, initial_notional, multiplier)
        rows.append(row)
        cumulative[name] = curve
        drawdowns[name] = mdd

    result = pd.DataFrame(rows)
    tables = root / "results" / "tables"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"
    for d in (tables, metrics, figures):
        d.mkdir(parents=True, exist_ok=True)

    table_path = tables / "frozen_oos_performance_reporting.csv"
    result.to_csv(table_path, index=False)

    # Mean cumulative normalized PnL path; this is the report's cumulative-performance figure.
    fig = plt.figure(figsize=(11, 6))
    steps = np.arange(1, EPISODE_STEPS + 1)
    for name in (
        "unhedged",
        "delta_1to1",
        "real_train_static_minvar",
        "causal_rolling60_minvar",
        "synthetic_pretrained_lstm",
        "real_finetuned_lstm",
    ):
        mean_curve = np.mean(cumulative[name], axis=0) * 100.0
        plt.plot(steps, mean_curve, label=name)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Trading day within 60-day OOS episode")
    plt.ylabel("Mean cumulative PnL / initial notional (%)")
    plt.title("Frozen OOS Mean Cumulative Hedging Performance")
    plt.legend()
    plt.tight_layout()
    cumulative_path = figures / "frozen_oos_mean_cumulative_performance.png"
    fig.savefig(cumulative_path, dpi=160)
    plt.close(fig)

    # Drawdown distribution figure for the main hedging comparators.
    fig = plt.figure(figsize=(10, 5))
    labels = [
        "delta_1to1",
        "real_train_static_minvar",
        "causal_rolling60_minvar",
        "synthetic_pretrained_lstm",
        "real_finetuned_lstm",
    ]
    data = [drawdowns[name] * 100.0 for name in labels]
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Episode max drawdown (% of initial notional)")
    plt.title("Frozen OOS 60-Day Drawdown Distribution")
    plt.tight_layout()
    drawdown_path = figures / "frozen_oos_drawdown_distribution.png"
    fig.savefig(drawdown_path, dpi=160)
    plt.close(fig)

    report = {
        "status": "POST_FREEZE_FINAL_PERFORMANCE_REPORTING_ONLY",
        "model_retuned": False,
        "real_test_already_unsealed": True,
        "episode_length": EPISODE_STEPS,
        "overlapping_episodes": True,
        "cumulative_curve_definition": (
            "mean within-episode cumulative economic PnL divided by episode initial VN30 notional"
        ),
        "sharpe_definition": (
            "descriptive pooled episode-day normalized PnL mean/std * sqrt(252); overlapping observations"
        ),
        "drawdown_definition": (
            "within-episode peak-to-trough additive normalized PnL drawdown with initial zero included"
        ),
        "results": rows,
        "important_note": (
            "No single stitched chronological LSTM wealth curve is reported because the trained policy has a 60-day episode clock; "
            "stitching would require an additional deployment/reset rule that was not pre-specified."
        ),
    }
    report_path = metrics / "frozen_oos_performance_reporting.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 128)
    print("FINAL CUMULATIVE-PERFORMANCE REPORTING COMPLETE — FROZEN MODELS, NO RETUNING")
    print("=" * 128)
    print(f"Test period : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Episodes    : {len(starts):,} overlapping 60-day paths")
    print(f"Device      : {device}")
    print("Cumulative curves are mean WITHIN-EPISODE paths, not a stitched chronological LSTM wealth curve.")
    print()
    print(
        result[
            [
                "strategy",
                "terminal_mean_vnd",
                "terminal_cvar_loss_5pct_vnd",
                "pooled_daily_sharpe_zero_rf",
                "mean_episode_max_drawdown",
                "q05_episode_max_drawdown",
                "worst_episode_max_drawdown",
                "hedge_ratio_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("IMPORTANT: Sharpe/drawdown are descriptive because episodes overlap with stride 1.")
    print("No log-return cumprod is used; cumulative performance is economic PnL / initial notional.")
    print(f"Table             : {table_path}")
    print(f"Report            : {report_path}")
    print(f"Cumulative figure : {cumulative_path}")
    print(f"Drawdown figure   : {drawdown_path}")


if __name__ == "__main__":
    main()
