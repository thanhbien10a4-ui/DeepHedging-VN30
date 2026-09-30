"""Graceful-degradation stress analysis for frozen OOS hedging strategies.

This is post-freeze descriptive TEST analysis only. The real TEST was already
opened in step 23. No model or hyperparameter is changed by this script.

Stress philosophy
-----------------
We perturb future daily spot/futures LOG returns after the observed TEST states.
Hedge decisions remain the frozen decisions generated from the original causal
TEST state sequences. Therefore this is an open-loop forward-shock robustness
test: it measures how much performance degrades when the future market relation
changes unexpectedly after decisions are formed.

Stress magnitudes are fixed ex ante and use TRAIN volatility only where a scale
parameter is needed. TEST data are not used to calibrate stress sizes.

Scenarios
---------
1. baseline
2. joint_vol_x1p25        : both spot and futures forward returns x1.25
3. joint_vol_x1p50        : both spot and futures forward returns x1.50
4. fut_noise_0p5_train_sd : add zero-mean idiosyncratic futures noise with
                            sigma = 0.5 * TRAIN futures daily std
5. fut_noise_1p0_train_sd : sigma = 1.0 * TRAIN futures daily std
6. downside_x1p50         : negative spot/futures returns amplified by 1.50
7. combined_stress        : joint vol x1.25 + futures noise 0.5 TRAIN std

Noise is generated once on the unique TEST daily timeline with a fixed seed and
then sliced into overlapping 60-day episodes, so repeated dates receive the
same shock across overlapping episodes.

Metrics use normalized log-PnL proxy:
    path_pnl = sum_t [r_spot,t - h_t * r_fut,t]

This isolates graceful degradation of hedge effectiveness from the market-cost
layer already studied in steps 24-26.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPISODE_STEPS = 60
TAIL_FRACTION = 0.05
BATCH_SIZE = 512
SEED = 20261008

SCENARIOS = [
    {"name": "baseline", "kind": "baseline"},
    {"name": "joint_vol_x1p25", "kind": "joint_vol", "factor": 1.25},
    {"name": "joint_vol_x1p50", "kind": "joint_vol", "factor": 1.50},
    {"name": "fut_noise_0p5_train_sd", "kind": "fut_noise", "noise_scale": 0.50},
    {"name": "fut_noise_1p0_train_sd", "kind": "fut_noise", "noise_scale": 1.00},
    {"name": "downside_x1p50", "kind": "downside", "factor": 1.50},
    {"name": "combined_stress", "kind": "combined", "vol_factor": 1.25, "noise_scale": 0.50},
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


def build_causal_rolling60(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    pre = pd.concat(
        [train[["time", "spot_close", "fut_close"]], validation[["time", "spot_close", "fut_close"]]],
        ignore_index=True,
    )
    combined = pd.concat([pre, test[["time", "spot_close", "fut_close"]]], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    combined["sr"] = np.log(combined["spot_close"]).diff()
    combined["fr"] = np.log(combined["fut_close"]).diff()
    first_test = len(pre)
    day_h = np.empty(len(test), dtype=float)
    for j in range(len(test)):
        hist = combined.iloc[max(0, first_test + j - 60) : first_test + j][["sr", "fr"]].dropna()
        if len(hist) < 20:
            raise ValueError("Insufficient causal history")
        day_h[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())
    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = day_h[start : start + EPISODE_STEPS]
    return out


def cvar_loss(pnl: np.ndarray) -> float:
    losses = -np.asarray(pnl, dtype=float).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k :]
    return float(np.mean(losses[idx]))


def unique_daily_log_returns(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = np.log(test["spot_close"].to_numpy(dtype=float))
    f = np.log(test["fut_close"].to_numpy(dtype=float))
    return np.diff(s), np.diff(f)


def apply_scenario(
    sr: np.ndarray,
    fr: np.ndarray,
    scenario: dict,
    train_fut_std: float,
    base_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kind = scenario["kind"]
    s = sr.astype(float).copy()
    f = fr.astype(float).copy()

    if kind == "baseline":
        return s, f
    if kind == "joint_vol":
        factor = float(scenario["factor"])
        return s * factor, f * factor
    if kind == "fut_noise":
        sigma = float(scenario["noise_scale"]) * train_fut_std
        return s, f + base_noise * sigma
    if kind == "downside":
        factor = float(scenario["factor"])
        s = np.where(s < 0.0, s * factor, s)
        f = np.where(f < 0.0, f * factor, f)
        return s, f
    if kind == "combined":
        vf = float(scenario["vol_factor"])
        sigma = float(scenario["noise_scale"]) * train_fut_std
        return s * vf, f * vf + base_noise * sigma
    raise ValueError(f"Unknown scenario kind: {kind}")


def slice_episodes(daily: np.ndarray, starts: np.ndarray) -> np.ndarray:
    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float64)
    for i, start in enumerate(starts):
        out[i] = daily[start : start + EPISODE_STEPS]
    return out


def summarize(strategy: str, scenario: str, spot: np.ndarray, fut: np.ndarray, h: np.ndarray) -> dict:
    pnl = np.sum(spot - h.astype(float) * fut, axis=1)
    flat_s = spot.reshape(-1)
    flat_f = fut.reshape(-1)
    corr = float(np.corrcoef(flat_s, flat_f)[0, 1])
    return {
        "scenario": scenario,
        "strategy": strategy,
        "terminal_mean": float(np.mean(pnl)),
        "terminal_std": float(np.std(pnl, ddof=1)),
        "terminal_q05": float(np.quantile(pnl, 0.05)),
        "cvar_loss_5pct": cvar_loss(pnl),
        "positive_pnl_fraction": float(np.mean(pnl > 0)),
        "spot_daily_std": float(np.std(flat_s, ddof=1)),
        "fut_daily_std": float(np.std(flat_f, ddof=1)),
        "spot_fut_corr": corr,
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
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

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(dtype=float)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(dtype=float)))
    train_fut_std = float(np.std(train_fr, ddof=1))
    static_h = minvar_ratio(train_sr, train_fr)

    shape = (len(starts), EPISODE_STEPS)
    h_zero = np.zeros(shape, dtype=np.float32)
    h_delta = np.ones(shape, dtype=np.float32)
    h_static = np.full(shape, static_h, dtype=np.float32)
    h_rolling = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_pre = predict(load_model(root / "results" / "models" / "lstm_cvar_turnover_best.pt", device), x_test, device)
    h_final = predict(load_model(root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt", device), x_test, device)

    strategies = {
        "unhedged": h_zero,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    sr_daily, fr_daily = unique_daily_log_returns(test)
    rng = np.random.default_rng(SEED)
    base_noise = rng.standard_normal(len(fr_daily))

    rows: list[dict] = []
    scenario_stats: dict[str, dict] = {}

    for scenario in SCENARIOS:
        s_daily, f_daily = apply_scenario(sr_daily, fr_daily, scenario, train_fut_std, base_noise)
        s_ep = slice_episodes(s_daily, starts)
        f_ep = slice_episodes(f_daily, starts)

        scenario_stats[scenario["name"]] = {
            "spot_daily_std": float(np.std(s_daily, ddof=1)),
            "fut_daily_std": float(np.std(f_daily, ddof=1)),
            "spot_fut_corr": float(np.corrcoef(s_daily, f_daily)[0, 1]),
        }

        for strategy, h in strategies.items():
            rows.append(summarize(strategy, scenario["name"], s_ep, f_ep, h))

    result = pd.DataFrame(rows)
    tables = root / "results" / "tables"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"
    for d in (tables, metrics, figures):
        d.mkdir(parents=True, exist_ok=True)

    table_path = tables / "graceful_degradation_frozen.csv"
    result.to_csv(table_path, index=False)

    baseline_final = float(
        result[(result["scenario"] == "baseline") & (result["strategy"] == "real_finetuned_lstm")]
        .iloc[0]["cvar_loss_5pct"]
    )

    graceful_rows: list[dict] = []
    for scenario in [s["name"] for s in SCENARIOS]:
        sub = result[result["scenario"] == scenario]
        final = sub[sub["strategy"] == "real_finetuned_lstm"].iloc[0]
        delta = sub[sub["strategy"] == "delta_1to1"].iloc[0]
        graceful_rows.append(
            {
                "scenario": scenario,
                "final_cvar": float(final["cvar_loss_5pct"]),
                "degradation_vs_final_baseline_fraction": float(
                    final["cvar_loss_5pct"] / baseline_final - 1.0
                ),
                "final_cvar_reduction_vs_delta_fraction": float(
                    1.0 - final["cvar_loss_5pct"] / delta["cvar_loss_5pct"]
                ),
                "final_terminal_mean": float(final["terminal_mean"]),
                "delta_terminal_mean": float(delta["terminal_mean"]),
                "spot_fut_corr": float(final["spot_fut_corr"]),
            }
        )

    graceful_df = pd.DataFrame(graceful_rows)
    summary_path = tables / "graceful_degradation_final_summary.csv"
    graceful_df.to_csv(summary_path, index=False)

    report = {
        "status": "POST_FREEZE_GRACEFUL_DEGRADATION_STRESS_ONLY",
        "model_retuned": False,
        "real_test_already_unsealed": True,
        "stress_calibration_source": "fixed factors; TRAIN futures std for idiosyncratic-noise scale",
        "seed": SEED,
        "open_loop_forward_shock_design": True,
        "policy_actions_recomputed_under_stressed_state": False,
        "scenario_definitions": SCENARIOS,
        "train_futures_daily_log_return_std": train_fut_std,
        "scenario_realized_market_stats": scenario_stats,
        "final_model_summary": graceful_rows,
        "limitation": (
            "This is an open-loop forward-shock stress test: frozen hedge actions come from original causal TEST states. "
            "It measures resilience to unexpected future-distribution changes, not full endogenous policy adaptation under a re-simulated stressed state path."
        ),
        "warning": "Do not retune the model from these TEST stress results.",
    }
    report_path = metrics / "graceful_degradation_frozen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(11, 5))
    x = np.arange(len(graceful_df))
    plt.plot(x, graceful_df["degradation_vs_final_baseline_fraction"] * 100.0, marker="o")
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, graceful_df["scenario"], rotation=35, ha="right")
    plt.ylabel("Final-model CVaR degradation vs baseline (%)")
    plt.title("Frozen OOS Graceful Degradation Under Market Stress")
    plt.tight_layout()
    fig_path = figures / "graceful_degradation_frozen.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print("=" * 132)
    print("GRACEFUL-DEGRADATION STRESS ANALYSIS COMPLETE — FROZEN MODELS, NO RETUNING")
    print("=" * 132)
    print(f"Test period        : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Episodes           : {len(starts):,} overlapping 60-day paths")
    print(f"Train futures std  : {train_fut_std:.6f} daily log-return")
    print(f"Noise seed         : {SEED}")
    print("Design             : open-loop forward shock; hedge actions stay frozen from original causal TEST states")
    print()

    print("REALIZED STRESS MARKET STATISTICS")
    for name, st in scenario_stats.items():
        print(
            f"{name:28s} | spot std={st['spot_daily_std']:.5f} | "
            f"fut std={st['fut_daily_std']:.5f} | corr={st['spot_fut_corr']:.4f}"
        )

    print()
    print("FINAL FROZEN MODEL — GRACEFUL DEGRADATION SUMMARY")
    for _, row in graceful_df.iterrows():
        print(
            f"{row['scenario']:28s} | CVaR={row['final_cvar']:.6f} | "
            f"degradation={row['degradation_vs_final_baseline_fraction']:+.2%} | "
            f"vs delta CVaR reduction={row['final_cvar_reduction_vs_delta_fraction']:+.2%} | "
            f"corr={row['spot_fut_corr']:.4f}"
        )

    print()
    print("IMPORTANT: post-freeze TEST stress analysis only; DO NOT RETUNE from these results.")
    print("Limitation: actions are not recursively recomputed from stressed states; this is an unexpected-forward-shock robustness test.")
    print(f"Full table : {table_path}")
    print(f"Summary    : {summary_path}")
    print(f"Report     : {report_path}")
    print(f"Figure     : {fig_path}")


if __name__ == "__main__":
    main()
