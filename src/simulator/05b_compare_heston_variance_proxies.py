"""Compare Heston physical-measure estimates across smoothed GK variance proxies.

Purpose
-------
The raw daily Garman-Klass variance proxy is noisy. This script compares
rolling-mean versions of the same daily GK variance proxy over 1, 5, 10 and
20 trading days before we choose parameters for synthetic-data simulation.

This is still physical-measure proxy estimation, NOT risk-neutral option
calibration.

Outputs
-------
results/tables/heston_proxy_comparison.csv
data/processed/heston_proxy_comparison.json
results/figures/heston_proxy_comparison_volatility.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS
VAR_FLOOR = 1e-8
WINDOWS = (1, 5, 10, 20)


def garman_klass_daily_variance(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["spot_high"] / df["spot_low"])
    log_co = np.log(df["spot_close"] / df["spot_open"])
    var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    return var.clip(lower=0.0)


def conditional_mean(v_t: np.ndarray, kappa: float, theta: float) -> np.ndarray:
    e = np.exp(-kappa * DT)
    return theta + (v_t - theta) * e


def cir_variance_coefficient(v_t: np.ndarray, kappa: float, theta: float) -> np.ndarray:
    e = np.exp(-kappa * DT)
    one_minus_e = 1.0 - e
    c1 = v_t * e * one_minus_e / kappa
    c2 = theta * (one_minus_e**2) / (2.0 * kappa)
    return np.maximum(c1 + c2, VAR_FLOOR)


def estimate_one(v_series: pd.Series, returns: pd.Series, dates: pd.Series, window: int) -> dict:
    tmp = pd.DataFrame({"time": dates, "r": returns, "v": v_series}).copy()
    tmp["v_start"] = tmp["v"].shift(1)
    tmp = tmp.dropna().reset_index(drop=True)

    v_t = tmp["v_start"].to_numpy(dtype=float)
    v_next = tmp["v"].to_numpy(dtype=float)
    r = tmp["r"].to_numpy(dtype=float)

    corr = np.corrcoef(v_t, v_next)[0, 1]
    if np.isfinite(corr) and 0.0 < corr < 0.999999:
        kappa0_raw = max(-np.log(corr) / DT, 0.1)
    else:
        kappa0_raw = 2.0

    theta0_raw = float(np.mean(v_t))
    lower = np.array([1e-4, 1e-8], dtype=float)
    upper = np.array([1000.0, 4.0], dtype=float)
    x0 = np.array([kappa0_raw, max(theta0_raw, 1e-6)], dtype=float)
    x0 = np.minimum(np.maximum(x0, lower + np.array([1e-8, 1e-12])), upper - np.array([1e-8, 1e-12]))

    def residuals(params: np.ndarray) -> np.ndarray:
        kappa, theta = params
        pred = conditional_mean(v_t, kappa, theta)
        scale = np.sqrt(np.maximum(v_t, VAR_FLOOR))
        return (v_next - pred) / scale

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=20_000,
    )

    kappa, theta = map(float, result.x)
    pred = conditional_mean(v_t, kappa, theta)
    var_resid = v_next - pred
    c = cir_variance_coefficient(v_t, kappa, theta)
    xi2 = float(np.mean((var_resid**2) / c))
    xi = float(np.sqrt(max(xi2, 0.0)))

    mu = float(np.mean(r) / DT + 0.5 * np.mean(v_t))

    eps_s = (r - (mu - 0.5 * v_t) * DT) / np.sqrt(np.maximum(v_t * DT, VAR_FLOOR))
    if xi > 0:
        eps_v = var_resid / np.sqrt(np.maximum((xi**2) * c, VAR_FLOOR))
        valid = np.isfinite(eps_s) & np.isfinite(eps_v)
        rho = float(np.corrcoef(eps_s[valid], eps_v[valid])[0, 1])
    else:
        rho = float("nan")

    half_life_days = float(np.log(2.0) / kappa * TRADING_DAYS)
    feller_lhs = float(2.0 * kappa * theta)
    feller_rhs = float(xi**2)
    feller_ratio = float(feller_lhs / feller_rhs) if feller_rhs > 0 else float("inf")

    return {
        "window": int(window),
        "rows_used": int(len(tmp)),
        "date_min": tmp["time"].min().strftime("%Y-%m-%d"),
        "date_max": tmp["time"].max().strftime("%Y-%m-%d"),
        "lag1_variance_corr": float(corr),
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "long_run_volatility": float(np.sqrt(theta)),
        "latest_volatility_proxy": float(np.sqrt(tmp["v"].iloc[-1])),
        "half_life_trading_days": half_life_days,
        "feller_ratio": feller_ratio,
        "feller_satisfied": bool(feller_lhs > feller_rhs),
        "variance_rmse": float(np.sqrt(np.mean(var_resid**2))),
        "variance_mae": float(np.mean(np.abs(var_resid))),
        "optimizer_success": bool(result.success),
        "optimizer_cost": float(result.cost),
        "kappa_at_upper_bound": bool(kappa >= 0.999 * upper[0]),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "aligned_1d.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run 01_validate_and_align.py first.")

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)
    df["spot_log_return"] = np.log(df["spot_close"]).diff()
    df["gk_daily_variance"] = garman_klass_daily_variance(df)

    results: list[dict] = []
    proxy_columns: dict[int, pd.Series] = {}

    for window in WINDOWS:
        if window == 1:
            daily_var = df["gk_daily_variance"]
        else:
            daily_var = df["gk_daily_variance"].rolling(window=window, min_periods=window).mean()

        v_proxy = (daily_var * TRADING_DAYS).clip(lower=VAR_FLOOR)
        proxy_columns[window] = v_proxy
        results.append(estimate_one(v_proxy, df["spot_log_return"], df["time"], window))

    comparison = pd.DataFrame(results)
    comparison_path = tables_dir / "heston_proxy_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    json_path = processed_dir / "heston_proxy_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "estimation_type": "physical_measure_proxy_comparison",
                "warning": (
                    "This compares smoothed Garman-Klass variance proxies. It does not constitute "
                    "risk-neutral Heston calibration from option prices."
                ),
                "windows": list(WINDOWS),
                "results": results,
                "selection_policy": (
                    "Do not select a window only because the Feller condition holds. Prefer a proxy "
                    "that reduces raw daily measurement noise while retaining economically meaningful "
                    "variance persistence. Review kappa, xi, half-life, rho and stability jointly."
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    fig = plt.figure(figsize=(12, 5))
    for window in WINDOWS:
        vol = np.sqrt(proxy_columns[window])
        label = "GK daily" if window == 1 else f"GK rolling {window}d"
        plt.plot(df["time"], vol, linewidth=0.8, label=label)
    plt.title("VN30 Garman-Klass Volatility Proxy: Smoothing Comparison")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "heston_proxy_comparison_volatility.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 100)
    print("HESTON VARIANCE-PROXY COMPARISON COMPLETE")
    print("=" * 100)
    print(
        comparison[
            [
                "window",
                "lag1_variance_corr",
                "kappa",
                "theta",
                "xi",
                "rho",
                "long_run_volatility",
                "half_life_trading_days",
                "feller_ratio",
                "feller_satisfied",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print(f"Table  : {comparison_path}")
    print(f"JSON   : {json_path}")
    print(f"Figure : {figure_path}")
    print()
    print("Do not choose the simulation parameters yet; review this comparison first.")


if __name__ == "__main__":
    main()
