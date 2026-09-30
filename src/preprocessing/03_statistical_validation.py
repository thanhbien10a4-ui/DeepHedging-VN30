"""Statistical validation and EDA for engineered VN30/VN30F1M features.

Input:
    data/processed/features_1d_model_ready.csv

Outputs (generated, gitignored):
    data/processed/statistical_validation.json
    results/tables/feature_descriptive_stats.csv
    results/tables/feature_correlation.csv
    results/figures/spot_fut_returns.png
    results/figures/rolling_volatility.png
    results/figures/basis_pct.png

Purpose:
- Validate stationarity and distributional properties of key series.
- Quantify serial dependence and spot/futures dependence.
- Detect highly redundant engineered features before modeling.
- Produce auditable EDA artifacts for the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

KEY_SERIES = [
    "spot_log_return",
    "fut_log_return",
    "basis_pct",
    "basis_velocity_1",
    "spot_realized_vol_20",
    "fut_realized_vol_20",
    "return_corr_20",
    "minvar_hedge_ratio_60",
]


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return float(x)


def adf_result(series: pd.Series) -> dict:
    x = series.dropna().astype(float)
    stat, pvalue, usedlag, nobs, critical_values, icbest = adfuller(x, autolag="AIC")
    return {
        "statistic": safe_float(stat),
        "pvalue": safe_float(pvalue),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {k: safe_float(v) for k, v in critical_values.items()},
        "icbest": safe_float(icbest),
    }


def kpss_result(series: pd.Series) -> dict:
    x = series.dropna().astype(float)
    stat, pvalue, lags, critical_values = kpss(x, regression="c", nlags="auto")
    return {
        "statistic": safe_float(stat),
        "pvalue": safe_float(pvalue),
        "lags": int(lags),
        "critical_values": {k: safe_float(v) for k, v in critical_values.items()},
    }


def distribution_result(series: pd.Series) -> dict:
    x = series.dropna().astype(float)
    jb = jarque_bera(x)
    return {
        "mean": safe_float(x.mean()),
        "std": safe_float(x.std(ddof=1)),
        "skewness": safe_float(skew(x, bias=False)),
        "excess_kurtosis": safe_float(kurtosis(x, fisher=True, bias=False)),
        "jarque_bera_stat": safe_float(jb.statistic),
        "jarque_bera_pvalue": safe_float(jb.pvalue),
    }


def ljung_box_result(series: pd.Series, lags=(5, 10, 20)) -> dict:
    x = series.dropna().astype(float)
    result = acorr_ljungbox(x, lags=list(lags), return_df=True)
    return {
        str(int(lag)): {
            "stat": safe_float(result.loc[lag, "lb_stat"]),
            "pvalue": safe_float(result.loc[lag, "lb_pvalue"]),
        }
        for lag in result.index
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "features_1d_model_ready.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run 02_feature_engineering.py first."
        )

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    feature_cols = [
        c
        for c in df.columns
        if c not in {"time", "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume",
                     "fut_open", "fut_high", "fut_low", "fut_close", "fut_volume"}
    ]

    missing_key = [c for c in KEY_SERIES if c not in df.columns]
    if missing_key:
        raise ValueError(f"Missing key series: {missing_key}")

    desc = df[feature_cols].describe().T
    desc["skewness"] = df[feature_cols].skew()
    desc["excess_kurtosis"] = df[feature_cols].kurt()
    desc.to_csv(tables_dir / "feature_descriptive_stats.csv")

    corr = df[feature_cols].corr(method="pearson")
    corr.to_csv(tables_dir / "feature_correlation.csv")

    # Highly redundant pairs by absolute Pearson correlation.
    redundant_pairs = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1 :]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) >= 0.90:
                redundant_pairs.append(
                    {"feature_a": a, "feature_b": b, "pearson_corr": safe_float(rho)}
                )
    redundant_pairs.sort(key=lambda x: abs(x["pearson_corr"]), reverse=True)

    tests = {}
    for col in KEY_SERIES:
        series = df[col]
        tests[col] = {
            "adf": adf_result(series),
            "kpss": kpss_result(series),
            "distribution": distribution_result(series),
            "ljung_box": ljung_box_result(series),
        }

    spot_fut_corr = safe_float(df["spot_log_return"].corr(df["fut_log_return"]))

    # Volatility clustering proxy: autocorrelation of squared returns.
    sq_acf = {
        "spot_squared_return_acf_1": safe_float(df["spot_log_return"].pow(2).autocorr(lag=1)),
        "spot_squared_return_acf_5": safe_float(df["spot_log_return"].pow(2).autocorr(lag=5)),
        "fut_squared_return_acf_1": safe_float(df["fut_log_return"].pow(2).autocorr(lag=1)),
        "fut_squared_return_acf_5": safe_float(df["fut_log_return"].pow(2).autocorr(lag=5)),
    }

    report = {
        "rows": int(len(df)),
        "date_min": df["time"].min().strftime("%Y-%m-%d"),
        "date_max": df["time"].max().strftime("%Y-%m-%d"),
        "feature_count": int(len(feature_cols)),
        "spot_futures_return_correlation": spot_fut_corr,
        "squared_return_autocorrelation": sq_acf,
        "highly_correlated_feature_pairs_abs_ge_0_90": redundant_pairs,
        "tests": tests,
        "interpretation_policy": {
            "adf": "Reject unit-root null when p < 0.05.",
            "kpss": "Reject stationarity null when p < 0.05.",
            "jarque_bera": "Reject normality when p < 0.05.",
            "ljung_box": "Evidence of serial dependence when p < 0.05.",
        },
    }

    with open(processed_dir / "statistical_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Plot 1: daily returns
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["spot_log_return"], label="VN30 spot")
    plt.plot(df["time"], df["fut_log_return"], label="VN30F1M", alpha=0.75)
    plt.axhline(0, linewidth=0.8)
    plt.title("Daily log returns: VN30 vs VN30F1M")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "spot_fut_returns.png", dpi=160)
    plt.close()

    # Plot 2: realized volatility
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["spot_realized_vol_20"], label="VN30 spot RV20")
    plt.plot(df["time"], df["fut_realized_vol_20"], label="VN30F1M RV20", alpha=0.8)
    plt.title("20-day annualized realized volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "rolling_volatility.png", dpi=160)
    plt.close()

    # Plot 3: futures basis
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["basis_pct"])
    plt.axhline(0, linewidth=0.8)
    plt.title("VN30F1M basis relative to VN30 spot")
    plt.xlabel("Date")
    plt.ylabel("Basis / spot")
    plt.tight_layout()
    plt.savefig(figures_dir / "basis_pct.png", dpi=160)
    plt.close()

    print("=" * 72)
    print("STATISTICAL VALIDATION COMPLETE")
    print("=" * 72)
    print(f"Rows                         : {len(df):,}")
    print(f"Features analyzed            : {len(feature_cols)}")
    print(f"Spot/Futures return corr     : {spot_fut_corr:.6f}")
    print(f"Highly correlated pairs >=.9 : {len(redundant_pairs)}")
    print(f"Report                       : {processed_dir / 'statistical_validation.json'}")
    print(f"Tables                       : {tables_dir}")
    print(f"Figures                      : {figures_dir}")


if __name__ == "__main__":
    main()
