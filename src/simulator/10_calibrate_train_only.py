"""Leakage-safe simulator calibration using TRAIN split only.

This script re-estimates all simulator ingredients after chronological splitting.
Nothing from validation/test is used here.

Inputs
------
data/processed/splits/train.csv

Outputs
-------
data/processed/calibration/train_heston_candidates.json
data/processed/calibration/train_basis_params.json
results/tables/train_heston_candidates.csv
results/tables/train_basis_process_comparison.csv

Policy
------
- Heston: compare GK variance proxies with windows 1, 5, 10, 20 days.
- Basis: fit AR(1) on full train and on train with likely roll-related transitions removed.
- Do NOT select the final Heston window here. Selection/tuning belongs to validation.
"""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS
VAR_FLOOR = 1e-8
WINDOWS = (1, 5, 10, 20)
ROBUST_Z_THRESHOLD = 4.0
MARKET_SHOCK_Z_THRESHOLD = 3.0
EXPIRY_WINDOW_CALENDAR_DAYS = 3


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


def estimate_heston_candidate(
    v_series: pd.Series,
    returns: pd.Series,
    dates: pd.Series,
    window: int,
) -> dict:
    tmp = pd.DataFrame({"time": dates, "r": returns, "v": v_series}).copy()
    tmp["v_start"] = tmp["v"].shift(1)
    tmp = tmp.dropna().reset_index(drop=True)

    v_t = tmp["v_start"].to_numpy(dtype=float)
    v_next = tmp["v"].to_numpy(dtype=float)
    r = tmp["r"].to_numpy(dtype=float)

    lag1_corr = float(np.corrcoef(v_t, v_next)[0, 1])
    if np.isfinite(lag1_corr) and 0.0 < lag1_corr < 0.999999:
        kappa0 = max(-np.log(lag1_corr) / DT, 0.1)
    else:
        kappa0 = 2.0

    theta0 = float(np.mean(v_t))
    lower = np.array([1e-4, 1e-8], dtype=float)
    upper = np.array([1000.0, 4.0], dtype=float)
    x0 = np.array([kappa0, max(theta0, 1e-6)], dtype=float)
    x0 = np.minimum(
        np.maximum(x0, lower + np.array([1e-8, 1e-12])),
        upper - np.array([1e-8, 1e-12]),
    )

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
        rho = 0.0

    feller_lhs = float(2.0 * kappa * theta)
    feller_rhs = float(xi**2)

    return {
        "window": int(window),
        "rows_used": int(len(tmp)),
        "date_min": tmp["time"].min().strftime("%Y-%m-%d"),
        "date_max": tmp["time"].max().strftime("%Y-%m-%d"),
        "lag1_variance_corr": lag1_corr,
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "long_run_volatility": float(np.sqrt(theta)),
        "latest_volatility_proxy": float(np.sqrt(tmp["v"].iloc[-1])),
        "half_life_trading_days": float(np.log(2.0) / kappa * TRADING_DAYS),
        "feller_ratio": float(feller_lhs / feller_rhs) if feller_rhs > 0 else float("inf"),
        "feller_satisfied": bool(feller_lhs > feller_rhs),
        "variance_rmse": float(np.sqrt(np.mean(var_resid**2))),
        "optimizer_success": bool(result.success),
    }


def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    median = x.median()
    mad = (x - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - median) / scale


def third_thursday(year: int, month: int) -> date:
    cal = calendar.monthcalendar(year, month)
    thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0]
    return date(year, month, thursdays[2])


def fit_basis_ar1(df: pd.DataFrame, label: str) -> dict:
    x = df["basis_lag1"].to_numpy(dtype=float)
    y = df["basis_pct"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, phi = map(float, beta)

    fitted = c + phi * x
    resid = y - fitted
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))

    long_run_mean = float(c / (1.0 - phi)) if abs(1.0 - phi) > 1e-12 else 0.0
    if 0.0 < phi < 1.0:
        half_life = float(np.log(0.5) / np.log(phi))
    else:
        half_life = float("nan")

    return {
        "sample": label,
        "rows_used": int(len(df)),
        "c": c,
        "phi": phi,
        "long_run_mean_basis_pct": long_run_mean,
        "residual_std_daily": float(np.std(resid, ddof=2)),
        "half_life_trading_days": half_life,
        "r_squared": float(1.0 - sse / sst),
        "stationary_abs_phi_lt_1": bool(abs(phi) < 1.0),
    }


def estimate_basis_train_only(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = train.copy()
    df["spot_log_return"] = np.log(df["spot_close"]).diff()
    df["fut_log_return"] = np.log(df["fut_close"]).diff()
    df["basis_pct"] = (df["fut_close"] - df["spot_close"]) / df["spot_close"]
    df["basis_lag1"] = df["basis_pct"].shift(1)
    df["basis_velocity_1"] = df["basis_pct"].diff()

    for col in ("spot_log_return", "fut_log_return", "basis_velocity_1"):
        df[f"rz_{col}"] = robust_zscore(df[col])

    expiry_dates = [third_thursday(ts.year, ts.month) for ts in df["time"]]
    df["third_thursday"] = pd.to_datetime(expiry_dates)
    df["days_from_third_thursday"] = (
        df["time"].dt.normalize() - df["third_thursday"]
    ).dt.days
    df["near_expiry_calendar"] = (
        df["days_from_third_thursday"].abs() <= EXPIRY_WINDOW_CALENDAR_DAYS
    )

    same_sign = np.sign(df["spot_log_return"]) == np.sign(df["fut_log_return"])
    df["market_shock"] = (
        (df["rz_spot_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & (df["rz_fut_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & same_sign
    )
    df["basis_jump"] = df["rz_basis_velocity_1"].abs() >= ROBUST_Z_THRESHOLD
    df["likely_roll_related"] = (
        df["basis_jump"] & df["near_expiry_calendar"] & (~df["market_shock"])
    )

    roll_dates = set(df.loc[df["likely_roll_related"], "time"])

    full = df.dropna(subset=["basis_pct", "basis_lag1"]).copy()
    full["prev_time"] = df["time"].shift(1).loc[full.index].to_numpy()
    filtered = full.loc[
        (~full["time"].isin(roll_dates))
        & (~full["prev_time"].isin(roll_dates))
    ].copy()

    full_result = fit_basis_ar1(full, "train_full")
    filtered_result = fit_basis_ar1(filtered, "train_filtered_roll_transitions")

    payload = {
        "calibration_scope": "train_only",
        "roll_detection_scope": "train_only",
        "likely_roll_dates_count": int(len(roll_dates)),
        "likely_roll_dates": sorted(ts.strftime("%Y-%m-%d") for ts in roll_dates),
        "results": {
            "full": full_result,
            "filtered": filtered_result,
        },
        "candidate_for_validation": "filtered",
    }
    comparison = pd.DataFrame([full_result, filtered_result])
    return payload, comparison


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    train_path = processed_dir / "splits" / "train.csv"
    calibration_dir = processed_dir / "calibration"
    tables_dir = repo_root / "results" / "tables"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run 09_prepare_chronological_splits.py first."
        )

    train = pd.read_csv(train_path)
    train["time"] = pd.to_datetime(train["time"], format="%Y-%m-%d", errors="raise")
    train = train.sort_values("time").reset_index(drop=True)
    train["spot_log_return"] = np.log(train["spot_close"]).diff()
    train["gk_daily_variance"] = garman_klass_daily_variance(train)

    heston_results: list[dict] = []
    for window in WINDOWS:
        if window == 1:
            daily_var = train["gk_daily_variance"]
        else:
            daily_var = train["gk_daily_variance"].rolling(window, min_periods=window).mean()
        v_proxy = (daily_var * TRADING_DAYS).clip(lower=VAR_FLOOR)
        heston_results.append(
            estimate_heston_candidate(v_proxy, train["spot_log_return"], train["time"], window)
        )

    heston_table = pd.DataFrame(heston_results)
    heston_table_path = tables_dir / "train_heston_candidates.csv"
    heston_table.to_csv(heston_table_path, index=False)

    heston_payload = {
        "calibration_scope": "train_only",
        "train_date_min": train["time"].min().strftime("%Y-%m-%d"),
        "train_date_max": train["time"].max().strftime("%Y-%m-%d"),
        "windows": list(WINDOWS),
        "results": heston_results,
        "selection_rule": (
            "No final window selected on train. Use validation only to compare simulator candidates. "
            "Test remains untouched."
        ),
    }
    heston_json_path = calibration_dir / "train_heston_candidates.json"
    with open(heston_json_path, "w", encoding="utf-8") as f:
        json.dump(heston_payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    basis_payload, basis_table = estimate_basis_train_only(train)
    basis_table_path = tables_dir / "train_basis_process_comparison.csv"
    basis_table.to_csv(basis_table_path, index=False)
    basis_json_path = calibration_dir / "train_basis_params.json"
    with open(basis_json_path, "w", encoding="utf-8") as f:
        json.dump(basis_payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 104)
    print("TRAIN-ONLY SIMULATOR CALIBRATION COMPLETE")
    print("=" * 104)
    print(f"Train rows : {len(train):,} | {train['time'].min().date()} -> {train['time'].max().date()}")
    print()
    print("Heston candidates (TRAIN ONLY):")
    print(
        heston_table[
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
    print("Basis candidates (TRAIN ONLY):")
    print(
        basis_table[
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
    print(f"Train-only likely roll dates : {basis_payload['likely_roll_dates_count']}")
    print(f"Heston JSON                 : {heston_json_path}")
    print(f"Basis JSON                  : {basis_json_path}")
    print()
    print("Next: evaluate candidate simulators on VALIDATION only. Do not use TEST.")


if __name__ == "__main__":
    main()
