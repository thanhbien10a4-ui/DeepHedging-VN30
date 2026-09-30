"""Final out-of-sample evaluation on REAL TEST 2025-2026.

WARNING
-------
Running this script OPENS the previously sealed real TEST split. After running it,
do not tune model architecture, hyperparameters, clipping thresholds, objective,
or simulator settings based on TEST results.

Frozen model
------------
results/models/lstm_cvar_turnover_real_finetuned_best.pt

Evaluation design
-----------------
The Deep Hedging policy was trained on 60-day episodes, so final path-risk
comparison is performed on all overlapping 60-day TEST episodes (stride=1),
matching the training/validation objective. These episodes are strongly
dependent; CVaR and quantiles are descriptive OOS metrics, not independent-
sample inference.

Strategies
----------
- unhedged
- delta_1to1
- real_train_static_minvar (estimated only on real TRAIN)
- causal_rolling60_minvar (uses only returns observed before each test decision)
- synthetic_pretrained_lstm (step 18; frozen zero-shot reference)
- real_finetuned_lstm (step 22; FINAL proposed model)

Costs
-----
Reports both gross path PnL proxy and the same normalized turnover-penalized
metric used during training, lambda_tc=0.001. This is NOT an official VN30F1M
fee/tax/slippage model.

This script writes no model-selection artifact and performs no tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TRADING_DAYS = 252.0
EPISODE_STEPS = 60
STRIDE = 1
CLIP_Z = 5.0
BATCH_SIZE = 512
TAIL_FRACTION = 0.05
LAMBDA_TC = 0.001

FEATURE_NAMES = [
    "lag_spot_log_return",
    "lag_fut_log_return",
    "basis_t",
    "sqrt_variance_t",
    "log_spot_relative_t",
    "log_futures_relative_t",
    "time_remaining",
]


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gk_annualized_variance(df: pd.DataFrame) -> np.ndarray:
    log_hl = np.log(df["spot_high"].to_numpy(float) / df["spot_low"].to_numpy(float))
    log_co = np.log(df["spot_close"].to_numpy(float) / df["spot_open"].to_numpy(float))
    gk_daily = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return np.maximum(gk_daily, 0.0) * TRADING_DAYS


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if list(meta["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Feature schema mismatch versus frozen synthetic scaler.")
    mean = np.array([meta["scaler_mean"][k] for k in FEATURE_NAMES], dtype=float)
    std = np.array([meta["scaler_std"][k] for k in FEATURE_NAMES], dtype=float)
    return mean, std


def validate_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {
        "time", "spot_open", "spot_high", "spot_low", "spot_close", "fut_close"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing {sorted(missing)}")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="raise")
    out = out.sort_values("time").reset_index(drop=True)
    if out["time"].duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    return out


def build_test_episodes(
    test: pd.DataFrame,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spot = test["spot_close"].to_numpy(float)
    fut = test["fut_close"].to_numpy(float)
    basis = (fut - spot) / spot
    sqrt_var = np.sqrt(gk_annualized_variance(test))

    starts = np.arange(0, len(test) - EPISODE_STEPS, STRIDE, dtype=int)
    n_ep = len(starts)
    x = np.empty((n_ep, EPISODE_STEPS, len(FEATURE_NAMES)), dtype=np.float32)
    sr = np.empty((n_ep, EPISODE_STEPS), dtype=np.float32)
    fr = np.empty((n_ep, EPISODE_STEPS), dtype=np.float32)
    t_rem = (EPISODE_STEPS - np.arange(EPISODE_STEPS, dtype=float)) / EPISODE_STEPS

    for j, start in enumerate(starts):
        end = start + EPISODE_STEPS
        s = spot[start : end + 1]
        f = fut[start : end + 1]
        sret = np.diff(np.log(s))
        fret = np.diff(np.log(f))

        lag_s = np.zeros(EPISODE_STEPS, dtype=float)
        lag_f = np.zeros(EPISODE_STEPS, dtype=float)
        lag_s[1:] = sret[:-1]
        lag_f[1:] = fret[:-1]

        raw = np.column_stack(
            [
                lag_s,
                lag_f,
                basis[start:end],
                sqrt_var[start:end],
                np.log(s[:-1] / s[0]),
                np.log(f[:-1] / f[0]),
                t_rem,
            ]
        )
        z = (raw - scaler_mean.reshape(1, -1)) / scaler_std.reshape(1, -1)
        x[j] = np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)
        sr[j] = sret.astype(np.float32)
        fr[j] = fret.astype(np.float32)

    return x, sr, fr, starts


def minvar_ratio(spot_ret: np.ndarray, fut_ret: np.ndarray) -> float:
    var_f = float(np.var(fut_ret, ddof=1))
    if var_f <= 0:
        raise ValueError("Non-positive futures variance.")
    return float(np.cov(spot_ret, fut_ret, ddof=1)[0, 1] / var_f)


def build_causal_rolling60_hedges(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    starts: np.ndarray,
) -> np.ndarray:
    pretest = pd.concat(
        [
            train[["time", "spot_close", "fut_close"]],
            validation[["time", "spot_close", "fut_close"]],
        ],
        ignore_index=True,
    )
    test_hist = test[["time", "spot_close", "fut_close"]].copy()
    combined = pd.concat([pretest, test_hist], ignore_index=True).sort_values("time").reset_index(drop=True)
    combined["sr"] = np.log(combined["spot_close"]).diff()
    combined["fr"] = np.log(combined["fut_close"]).diff()

    first_test_idx = len(pretest)
    per_test_day_h = np.empty(len(test), dtype=float)
    for j in range(len(test)):
        idx = first_test_idx + j
        hist = combined.iloc[max(0, idx - 60) : idx]
        valid = hist[["sr", "fr"]].dropna()
        if len(valid) < 20:
            raise ValueError("Insufficient causal history for rolling min-var.")
        per_test_day_h[j] = minvar_ratio(valid["sr"].to_numpy(), valid["fr"].to_numpy())

    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = per_test_day_h[start : start + EPISODE_STEPS].astype(np.float32)
    return out


def load_model(checkpoint_path: Path, device: torch.device) -> LSTMHedgePolicy:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def turnover(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64)
    return (
        np.abs(h[:, 0])
        + np.sum(np.abs(h[:, 1:] - h[:, :-1]), axis=1)
        + np.abs(h[:, -1])
    )


def cvar(losses: np.ndarray) -> float:
    x = np.asarray(losses, dtype=float).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(x))))
    idx = np.argpartition(x, len(x) - k)[len(x) - k :]
    return float(np.mean(x[idx]))


def summarize(name: str, spot: np.ndarray, fut: np.ndarray, h: np.ndarray) -> dict:
    s = spot.astype(float)
    f = fut.astype(float)
    h = h.astype(float)
    daily = s - h * f
    gross = np.sum(daily, axis=1)
    turn = turnover(h)
    net = gross - LAMBDA_TC * turn
    return {
        "strategy": name,
        "gross_daily_std": float(np.std(daily.reshape(-1), ddof=1)),
        "gross_terminal_mean": float(np.mean(gross)),
        "gross_terminal_std": float(np.std(gross, ddof=1)),
        "gross_terminal_q05": float(np.quantile(gross, 0.05)),
        "gross_cvar_loss_worst_5pct": cvar(-gross),
        "net_terminal_mean_proxy": float(np.mean(net)),
        "net_terminal_std_proxy": float(np.std(net, ddof=1)),
        "net_terminal_q05_proxy": float(np.quantile(net, 0.05)),
        "net_cvar_loss_worst_5pct_proxy": cvar(-net),
        "turnover_mean_open_rebalance_close": float(np.mean(turn)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
        "hedge_q01": float(np.quantile(h, 0.01)),
        "hedge_q50": float(np.quantile(h, 0.50)),
        "hedge_q99": float(np.quantile(h, 0.99)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    splits = root / "data" / "processed" / "splits"
    scaler_path = root / "data" / "processed" / "deep_hedging" / "sequence_metadata.json"
    pretrained_path = root / "results" / "models" / "lstm_cvar_turnover_best.pt"
    final_model_path = root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt"
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    processed_real = root / "data" / "processed" / "deep_hedging_real"
    for d in (tables_dir, metrics_dir, figures_dir, processed_real):
        d.mkdir(parents=True, exist_ok=True)

    # THIS IS THE FIRST INTENTIONAL POST-FREEZE OPEN OF REAL TEST.
    train = validate_split(pd.read_csv(splits / "train.csv"), "train")
    validation = validate_split(pd.read_csv(splits / "validation.csv"), "validation")
    test = validate_split(pd.read_csv(splits / "test.csv"), "test")

    scaler_mean, scaler_std = load_scaler(scaler_path)
    x_test, s_test, f_test, starts = build_test_episodes(test, scaler_mean, scaler_std)
    np.savez_compressed(
        processed_real / "test_sequences_final_oos.npz",
        features=x_test,
        spot_forward_log_return=s_test,
        fut_forward_log_return=f_test,
        episode_start_index=starts.astype(np.int32),
    )

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(float)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(float)))
    static_h = minvar_ratio(train_sr, train_fr)

    h_unhedged = np.zeros_like(s_test, dtype=np.float32)
    h_delta = np.ones_like(s_test, dtype=np.float32)
    h_static = np.full_like(s_test, static_h, dtype=np.float32)
    h_rolling = build_causal_rolling60_hedges(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = load_model(pretrained_path, device)
    final_model = load_model(final_model_path, device)
    h_pre = predict(pretrained_model, x_test, device)
    h_final = predict(final_model, x_test, device)

    strategy_map = {
        "unhedged": h_unhedged,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }
    rows = [summarize(name, s_test, f_test, h) for name, h in strategy_map.items()]
    result = pd.DataFrame(rows)
    table_path = tables_dir / "final_oos_real_test_comparison.csv"
    result.to_csv(table_path, index=False)

    improvement = {}
    final_row = next(r for r in rows if r["strategy"] == "real_finetuned_lstm")
    for baseline in ("delta_1to1", "real_train_static_minvar", "causal_rolling60_minvar"):
        b = next(r for r in rows if r["strategy"] == baseline)
        improvement[baseline] = {
            "gross_cvar_reduction_fraction": float(
                1.0 - final_row["gross_cvar_loss_worst_5pct"] / b["gross_cvar_loss_worst_5pct"]
            ),
            "net_proxy_cvar_reduction_fraction": float(
                1.0 - final_row["net_cvar_loss_worst_5pct_proxy"] / b["net_cvar_loss_worst_5pct_proxy"]
            ),
        }

    report = {
        "status": "FINAL_REAL_TEST_OPENED_AND_EVALUATED",
        "test_opened_at_local_runtime": datetime.now().isoformat(timespec="seconds"),
        "test_period": {
            "rows": int(len(test)),
            "date_min": test["time"].min().strftime("%Y-%m-%d"),
            "date_max": test["time"].max().strftime("%Y-%m-%d"),
            "overlapping_60d_episodes": int(len(starts)),
        },
        "frozen_model_sha256": sha256_file(final_model_path),
        "pretrained_model_sha256": sha256_file(pretrained_path),
        "input_clip_z": CLIP_Z,
        "tail_fraction": TAIL_FRACTION,
        "lambda_tc_normalized": LAMBDA_TC,
        "official_transaction_cost_model_used": False,
        "real_train_static_minvar_ratio": static_h,
        "results": rows,
        "final_model_improvement_vs_baselines": improvement,
        "methodological_warning": (
            "Do not tune any model choice based on this real TEST result. Overlapping episodes are dependent. "
            "The normalized turnover penalty is not an official VN30F1M fee model."
        ),
    }
    report_path = metrics_dir / "final_oos_real_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(10, 5))
    names = result["strategy"].tolist()
    vals = result["gross_cvar_loss_worst_5pct"].to_numpy()
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Gross 60-day CVaR loss (worst 5%)")
    plt.title("FINAL OOS Real TEST: 60-day Tail Risk")
    plt.tight_layout()
    fig_path = figures_dir / "final_oos_real_test_cvar.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print("=" * 124)
    print("FINAL OUT-OF-SAMPLE REAL TEST EVALUATION COMPLETE — TEST IS NOW UNSEALED")
    print("=" * 124)
    print(f"Test period            : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Test rows              : {len(test):,}")
    print(f"60-day episodes        : {len(starts):,} (overlapping, stride=1)")
    print(f"Device                 : {device}")
    print(f"Static min-var h*      : {static_h:.6f} (TRAIN only)")
    print(f"Normalized lambda_tc   : {LAMBDA_TC:.6f} (NOT official market cost)")
    print()
    print(
        result[
            [
                "strategy",
                "gross_terminal_mean",
                "gross_terminal_std",
                "gross_terminal_q05",
                "gross_cvar_loss_worst_5pct",
                "net_terminal_mean_proxy",
                "net_cvar_loss_worst_5pct_proxy",
                "turnover_mean_open_rebalance_close",
                "hedge_ratio_mean",
                "hedge_ratio_std",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("FINAL MODEL CVaR REDUCTION VS BASELINES:")
    for baseline, vals in improvement.items():
        print(
            f"{baseline:26s} | gross={vals['gross_cvar_reduction_fraction']:+.2%} | "
            f"net-proxy={vals['net_proxy_cvar_reduction_fraction']:+.2%}"
        )
    print()
    print("IMPORTANT: REAL TEST IS NOW UNSEALED. DO NOT RETUNE THE MODEL FROM THESE RESULTS.")
    print("The lambda_tc metric is a normalized penalty, not an official VN30F1M transaction-cost model.")
    print(f"Table  : {table_path}")
    print(f"Report : {report_path}")
    print(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()
