"""Evaluate traditional VN30/VN30F1M hedge baselines on VALIDATION only.

Objective
---------
Hedge a linear long VN30 spot exposure with short VN30F1M futures exposure.
For this linear exposure, the 1:1 hedge is the natural ``delta=1`` baseline.
This is not an option-delta model; no nonlinear option payoff is assumed here.

Baselines
---------
1) Unhedged: h_t = 0
2) Delta / naive 1:1 hedge: h_t = 1
3) Static minimum-variance hedge ratio estimated on TRAIN only
4) Rolling 60-day minimum-variance hedge ratio, computed causally from past data

Hedged normalized daily PnL/return:
    r_hedged,t = r_spot,t - h_t * r_futures,t

Leakage policy
--------------
- Static minimum-variance ratio: TRAIN only.
- Rolling ratio on validation day t: previous 60 observed returns only.
- Real TEST split is never opened.

Transaction costs
-----------------
This step reports GROSS results only. Official VN30 futures fees/taxes/slippage
are intentionally not hardcoded yet. Net-of-cost baselines will be added only
after the market-cost specification is documented from authoritative sources.

Outputs
-------
results/tables/baseline_validation_metrics.csv
results/tables/baseline_validation_timeseries.csv
data/processed/baseline_validation_report.json
results/figures/baseline_validation_cumulative_returns.png
results/figures/baseline_validation_hedge_ratios.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
ROLLING_WINDOW = 60


def min_variance_ratio(spot_ret: np.ndarray, fut_ret: np.ndarray) -> float:
    spot_ret = np.asarray(spot_ret, dtype=float)
    fut_ret = np.asarray(fut_ret, dtype=float)
    valid = np.isfinite(spot_ret) & np.isfinite(fut_ret)
    s = spot_ret[valid]
    f = fut_ret[valid]
    if len(s) < 2:
        raise ValueError("Not enough observations to estimate minimum-variance hedge ratio.")
    var_f = float(np.var(f, ddof=1))
    if var_f <= 0:
        raise ValueError("Futures return variance is non-positive.")
    cov_sf = float(np.cov(s, f, ddof=1)[0, 1])
    return cov_sf / var_f


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0
    return float(np.min(drawdown))


def metrics_for_strategy(
    name: str,
    returns: np.ndarray,
    unhedged_returns: np.ndarray,
    hedge_ratio: np.ndarray,
) -> dict:
    r = np.asarray(returns, dtype=float)
    u = np.asarray(unhedged_returns, dtype=float)
    h = np.asarray(hedge_ratio, dtype=float)

    mean_daily = float(np.mean(r))
    std_daily = float(np.std(r, ddof=1))
    annualized_vol = std_daily * np.sqrt(TRADING_DAYS)
    sharpe = float(mean_daily / std_daily * np.sqrt(TRADING_DAYS)) if std_daily > 0 else float("nan")
    cumulative_return = float(np.prod(1.0 + r) - 1.0)
    max_dd = max_drawdown_from_returns(r)

    var_u = float(np.var(u, ddof=1))
    var_h = float(np.var(r, ddof=1))
    hedge_effectiveness = float(1.0 - var_h / var_u) if var_u > 0 else float("nan")

    turnover = np.abs(np.diff(h, prepend=h[0]))

    return {
        "strategy": name,
        "n_days": int(len(r)),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "annualized_mean_arithmetic": float(mean_daily * TRADING_DAYS),
        "annualized_volatility": float(annualized_vol),
        "sharpe_zero_rf": sharpe,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_dd,
        "hedge_effectiveness_variance_reduction": hedge_effectiveness,
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)) if len(h) > 1 else 0.0,
        "hedge_ratio_min": float(np.min(h)),
        "hedge_ratio_max": float(np.max(h)),
        "mean_daily_turnover_abs_dh": float(np.mean(turnover)),
        "total_turnover_abs_dh": float(np.sum(turnover)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    splits_dir = root / "data" / "processed" / "splits"
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    validation_path = splits_dir / "validation.csv"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run step 09 first.")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    train["time"] = pd.to_datetime(train["time"], errors="raise")
    validation["time"] = pd.to_datetime(validation["time"], errors="raise")

    train = train.sort_values("time").reset_index(drop=True)
    validation = validation.sort_values("time").reset_index(drop=True)

    # TRAIN returns for static minimum-variance estimation.
    train_spot_ret = np.log(train["spot_close"]).diff()
    train_fut_ret = np.log(train["fut_close"]).diff()
    static_h = min_variance_ratio(train_spot_ret.to_numpy(), train_fut_ret.to_numpy())

    # Build a continuous TRAIN+VALIDATION history so the first validation return
    # correctly uses the last TRAIN close, while all rolling estimates remain causal.
    train_hist = train[["time", "spot_close", "fut_close"]].copy()
    train_hist["split"] = "train"
    val_hist = validation[["time", "spot_close", "fut_close"]].copy()
    val_hist["split"] = "validation"
    combined = pd.concat([train_hist, val_hist], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    combined["spot_log_return"] = np.log(combined["spot_close"]).diff()
    combined["fut_log_return"] = np.log(combined["fut_close"]).diff()

    validation_indices = combined.index[combined["split"] == "validation"].to_numpy()
    rolling_h = np.empty(len(validation_indices), dtype=float)

    for j, idx in enumerate(validation_indices):
        start = idx - ROLLING_WINDOW
        if start < 0:
            raise ValueError("Insufficient pre-validation history for rolling hedge ratio.")
        hist = combined.iloc[start:idx]
        rolling_h[j] = min_variance_ratio(
            hist["spot_log_return"].to_numpy(),
            hist["fut_log_return"].to_numpy(),
        )

    val_rows = combined.loc[validation_indices].copy().reset_index(drop=True)
    spot_ret = val_rows["spot_log_return"].to_numpy(dtype=float)
    fut_ret = val_rows["fut_log_return"].to_numpy(dtype=float)

    if not np.isfinite(spot_ret).all() or not np.isfinite(fut_ret).all():
        raise ValueError("Validation returns contain NaN/inf values.")

    h_unhedged = np.zeros_like(spot_ret)
    h_delta = np.ones_like(spot_ret)
    h_static = np.full_like(spot_ret, static_h)
    h_rolling = rolling_h

    strategy_returns = {
        "unhedged": spot_ret,
        "delta_1to1": spot_ret - h_delta * fut_ret,
        "minvar_static_train": spot_ret - h_static * fut_ret,
        "minvar_rolling60_causal": spot_ret - h_rolling * fut_ret,
    }
    strategy_h = {
        "unhedged": h_unhedged,
        "delta_1to1": h_delta,
        "minvar_static_train": h_static,
        "minvar_rolling60_causal": h_rolling,
    }

    metrics_rows = [
        metrics_for_strategy(name, strategy_returns[name], spot_ret, strategy_h[name])
        for name in strategy_returns
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = tables_dir / "baseline_validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    ts = pd.DataFrame({
        "time": val_rows["time"],
        "spot_log_return": spot_ret,
        "fut_log_return": fut_ret,
        "h_unhedged": h_unhedged,
        "h_delta_1to1": h_delta,
        "h_minvar_static_train": h_static,
        "h_minvar_rolling60_causal": h_rolling,
    })

    for name, r in strategy_returns.items():
        ts[f"return_{name}"] = r
        ts[f"wealth_{name}"] = np.cumprod(1.0 + r)

    ts_path = tables_dir / "baseline_validation_timeseries.csv"
    ts.to_csv(ts_path, index=False, date_format="%Y-%m-%d")

    report = {
        "evaluation_scope": "real_validation_only",
        "real_test_used": False,
        "exposure_definition": "long linear VN30 spot exposure hedged with short VN30F1M futures",
        "delta_baseline_definition": "1:1 hedge ratio because the hedged exposure is linear with delta=1",
        "return_definition": "r_spot - h_t * r_futures using close-to-close log returns as normalized gross hedge PnL proxy",
        "transaction_costs_included": False,
        "transaction_cost_note": (
            "Gross baseline only. Official VN30 futures fees/taxes/slippage are not hardcoded in this step."
        ),
        "static_minvar_ratio_train_only": float(static_h),
        "rolling_window": ROLLING_WINDOW,
        "validation_date_min": val_rows["time"].min().strftime("%Y-%m-%d"),
        "validation_date_max": val_rows["time"].max().strftime("%Y-%m-%d"),
        "metrics": metrics_rows,
    }

    report_path = processed_dir / "baseline_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(12, 5))
    for name in strategy_returns:
        plt.plot(ts["time"], ts[f"wealth_{name}"], label=name, linewidth=1.0)
    plt.axhline(1.0, linewidth=0.8)
    plt.title("Traditional Hedging Baselines on Real Validation Data — Gross")
    plt.xlabel("Date")
    plt.ylabel("Normalized wealth")
    plt.legend()
    plt.tight_layout()
    cumulative_fig = figures_dir / "baseline_validation_cumulative_returns.png"
    fig.savefig(cumulative_fig, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(ts["time"], h_rolling, label="rolling 60d min-var", linewidth=1.0)
    plt.axhline(static_h, label="static train min-var", linewidth=1.0)
    plt.axhline(1.0, label="delta 1:1", linewidth=1.0)
    plt.title("Baseline Hedge Ratios on Real Validation Data")
    plt.xlabel("Date")
    plt.ylabel("Hedge ratio h")
    plt.legend()
    plt.tight_layout()
    hedge_fig = figures_dir / "baseline_validation_hedge_ratios.png"
    fig.savefig(hedge_fig, dpi=160)
    plt.close(fig)

    print("=" * 112)
    print("TRADITIONAL HEDGING BASELINE VALIDATION COMPLETE — GROSS, VALIDATION ONLY")
    print("=" * 112)
    print(f"Validation rows            : {len(val_rows):,}")
    print(f"Validation period          : {val_rows['time'].min().date()} -> {val_rows['time'].max().date()}")
    print(f"Static min-var h (TRAIN)   : {static_h:.6f}")
    print(f"Rolling min-var window     : {ROLLING_WINDOW} trading days")
    print()
    print(
        metrics_df[
            [
                "strategy",
                "cumulative_return",
                "annualized_volatility",
                "sharpe_zero_rf",
                "max_drawdown",
                "hedge_effectiveness_variance_reduction",
                "hedge_ratio_mean",
                "hedge_ratio_std",
                "total_turnover_abs_dh",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("IMPORTANT: transaction costs are NOT included yet.")
    print("REAL TEST DATA WAS NOT READ.")
    print(f"Metrics table   : {metrics_path}")
    print(f"Timeseries      : {ts_path}")
    print(f"JSON report     : {report_path}")
    print(f"Cumulative fig  : {cumulative_fig}")
    print(f"Hedge-ratio fig : {hedge_fig}")


if __name__ == "__main__":
    main()
