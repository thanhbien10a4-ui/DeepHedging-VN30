"""Estimate a mean-reverting VN30F1M basis process for joint simulation.

Why this step exists
--------------------
Heston models the VN30 spot dynamics, but Deep Hedging will trade VN30F1M.
Therefore synthetic training data needs a futures process too. We model the
relative basis

    b_t = (F_t - S_t) / S_t

with a simple AR(1) / discrete OU approximation:

    b_{t+1} = c + phi * b_t + eps_{t+1}

and compare estimates on:
1) the full aligned sample;
2) a filtered sample that excludes dates previously flagged as likely
   roll-related by 04_rollover_outlier_diagnostics.py.

Outputs
-------
data/processed/basis_process_params.json
results/tables/basis_process_comparison.csv
results/figures/basis_ar1_fit.png

Important
---------
This script does not alter raw data. Excluding likely roll-related dates is
only for parameter estimation; the original observations remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def fit_ar1(df: pd.DataFrame, label: str) -> dict:
    x = df["basis_lag1"].to_numpy(dtype=float)
    y = df["basis_pct"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, phi = map(float, beta)

    fitted = c + phi * x
    resid = y - fitted
    resid_std = float(np.std(resid, ddof=2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y - np.mean(y)) ** 2))

    if abs(1.0 - phi) > 1e-12:
        long_run_mean = float(c / (1.0 - phi))
    else:
        long_run_mean = float("nan")

    stationary = bool(abs(phi) < 1.0)
    if 0.0 < phi < 1.0:
        half_life_days = float(np.log(0.5) / np.log(phi))
        kappa_year = float(-np.log(phi) * TRADING_DAYS)
    else:
        half_life_days = float("nan")
        kappa_year = float("nan")

    return {
        "sample": label,
        "rows_used": int(len(df)),
        "c": c,
        "phi": phi,
        "long_run_mean_basis_pct": long_run_mean,
        "residual_std_daily": resid_std,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r2,
        "stationary_abs_phi_lt_1": stationary,
        "half_life_trading_days": half_life_days,
        "ou_kappa_per_year_equivalent": kappa_year,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = processed_dir / "aligned_1d.csv"
    flagged_path = tables_dir / "rollover_flagged_dates.csv"

    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing {aligned_path}. Run 01_validate_and_align.py first.")
    if not flagged_path.exists():
        raise FileNotFoundError(
            f"Missing {flagged_path}. Run 04_rollover_outlier_diagnostics.py (v2) first."
        )

    df = pd.read_csv(aligned_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)
    df["basis_pct"] = (df["fut_close"] - df["spot_close"]) / df["spot_close"]
    df["basis_lag1"] = df["basis_pct"].shift(1)

    flagged = pd.read_csv(flagged_path)
    flagged["time"] = pd.to_datetime(flagged["time"], errors="raise")

    if "likely_roll_related" not in flagged.columns:
        raise ValueError(
            "rollover_flagged_dates.csv is from the old diagnostic version. "
            "Re-run 04_rollover_outlier_diagnostics.py after git pull."
        )

    roll_dates = set(flagged.loc[flagged["likely_roll_related"].astype(bool), "time"])

    full = df.dropna(subset=["basis_pct", "basis_lag1"]).copy()

    # Exclude a transition whenever either endpoint touches a likely roll-related date.
    # This prevents a roll discontinuity from contaminating the AR(1) residual variance.
    filtered = full.loc[
        (~full["time"].isin(roll_dates))
        & (~(full["time"] - pd.Timedelta(days=1)).isin(roll_dates))
    ].copy()

    # A safer exact previous-observation check for non-calendar trading gaps.
    full["prev_time"] = df["time"].shift(1).loc[full.index].to_numpy()
    filtered = full.loc[
        (~full["time"].isin(roll_dates))
        & (~full["prev_time"].isin(roll_dates))
    ].copy()

    full_result = fit_ar1(full, "full")
    filtered_result = fit_ar1(filtered, "filtered_excluding_likely_roll_transitions")

    comparison = pd.DataFrame([full_result, filtered_result])
    comparison_path = tables_dir / "basis_process_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    output = {
        "model": "AR(1) discrete OU approximation for relative basis",
        "basis_definition": "(VN30F1M_close - VN30_close) / VN30_close",
        "roll_dates_excluded_count": int(len(roll_dates)),
        "results": {
            "full": full_result,
            "filtered": filtered_result,
        },
        "recommended_for_simulation": "filtered",
        "simulation_equation": "b_next = c + phi*b_current + epsilon, epsilon~N(0,residual_std_daily^2)",
        "warning": (
            "Gaussian basis innovations are a baseline simplification. Tail/jump behavior and explicit "
            "expiry-roll shocks should be tested separately in robustness experiments."
        ),
    }

    json_path = processed_dir / "basis_process_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

    # Plot actual basis and one-step fitted conditional mean using filtered parameters.
    c = filtered_result["c"]
    phi = filtered_result["phi"]
    plot_df = full.copy()
    plot_df["fitted_basis"] = c + phi * plot_df["basis_lag1"]

    fig = plt.figure(figsize=(12, 5))
    plt.plot(plot_df["time"], plot_df["basis_pct"], linewidth=0.8, label="Observed basis_pct")
    plt.plot(plot_df["time"], plot_df["fitted_basis"], linewidth=0.8, label="AR(1) conditional mean")
    plt.title("VN30F1M Relative Basis and AR(1) Conditional Mean")
    plt.xlabel("Date")
    plt.ylabel("Basis as fraction of VN30 spot")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "basis_ar1_fit.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 92)
    print("BASIS PROCESS ESTIMATION COMPLETE")
    print("=" * 92)
    print(
        comparison[
            [
                "sample",
                "rows_used",
                "phi",
                "long_run_mean_basis_pct",
                "residual_std_daily",
                "half_life_trading_days",
                "r_squared",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print(f"Likely roll dates excluded : {len(roll_dates)}")
    print(f"Parameters                 : {json_path}")
    print(f"Comparison table           : {comparison_path}")
    print(f"Figure                     : {figure_path}")
    print()
    print("Recommended baseline for joint simulation: filtered AR(1) basis process.")


if __name__ == "__main__":
    main()
