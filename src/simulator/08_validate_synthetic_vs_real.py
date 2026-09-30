"""Validate synthetic VN30/VN30F1M paths against the real aligned sample.

The goal is not to make every sample moment identical. We want to know whether
synthetic data preserves the market characteristics that matter for hedging:
- spot/futures return location and scale;
- skewness and fat tails;
- volatility clustering via squared-return autocorrelation;
- spot/futures return correlation;
- relative-basis distribution and persistence.

Inputs
------
data/processed/aligned_1d.csv
data/synthetic/train.npz
data/synthetic/validation.npz
data/synthetic/test.npz

Outputs
-------
data/processed/synthetic_validation.json
results/tables/synthetic_vs_real_metrics.csv
results/figures/synthetic_vs_real_spot_returns.png
results/figures/synthetic_vs_real_basis.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)


def acf_at_lag(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_series(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.quantile(x, QUANTILES)
    out = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)),
        f"{prefix}_skew": float(skew(x, bias=False)),
        f"{prefix}_excess_kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        f"{prefix}_q01": float(q[0]),
        f"{prefix}_q05": float(q[1]),
        f"{prefix}_q50": float(q[2]),
        f"{prefix}_q95": float(q[3]),
        f"{prefix}_q99": float(q[4]),
    }
    return out


def summarize_return_series(x: np.ndarray, prefix: str) -> dict[str, float]:
    out = summarize_series(x, prefix)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sq = x * x
    out[f"{prefix}_squared_acf_1"] = acf_at_lag(sq, 1)
    out[f"{prefix}_squared_acf_5"] = acf_at_lag(sq, 5)
    return out


def summarize_real(df: pd.DataFrame) -> dict[str, float]:
    spot_ret = np.log(df["spot_close"].to_numpy(dtype=float))[1:] - np.log(
        df["spot_close"].to_numpy(dtype=float)
    )[:-1]
    fut_ret = np.log(df["fut_close"].to_numpy(dtype=float))[1:] - np.log(
        df["fut_close"].to_numpy(dtype=float)
    )[:-1]
    basis = ((df["fut_close"] - df["spot_close"]) / df["spot_close"]).to_numpy(dtype=float)

    metrics: dict[str, float] = {}
    metrics.update(summarize_return_series(spot_ret, "spot_return"))
    metrics.update(summarize_return_series(fut_ret, "futures_return"))
    metrics.update(summarize_series(basis, "basis"))
    metrics["spot_futures_return_corr"] = float(np.corrcoef(spot_ret, fut_ret)[0, 1])
    metrics["basis_acf_1"] = acf_at_lag(basis, 1)
    return metrics


def summarize_synthetic(npz_path: Path) -> dict[str, float]:
    with np.load(npz_path) as z:
        spot_ret_2d = z["spot_log_return"].astype(np.float64)
        fut_ret_2d = z["fut_log_return"].astype(np.float64)
        basis_2d = z["basis"].astype(np.float64)

    # For distributional moments, pool all generated daily observations.
    spot_ret = spot_ret_2d.ravel()
    fut_ret = fut_ret_2d.ravel()
    # Exclude t=0 because it is the same deterministic initial basis on every path.
    basis = basis_2d[:, 1:].ravel()

    metrics: dict[str, float] = {}
    metrics.update(summarize_series(spot_ret, "spot_return"))
    metrics.update(summarize_series(fut_ret, "futures_return"))
    metrics.update(summarize_series(basis, "basis"))
    metrics["spot_futures_return_corr"] = float(np.corrcoef(spot_ret, fut_ret)[0, 1])

    # Time-series dependence must be measured within paths rather than across
    # flattened path boundaries. Average path-level ACFs.
    spot_sq_acf1 = []
    spot_sq_acf5 = []
    fut_sq_acf1 = []
    fut_sq_acf5 = []
    basis_acf1 = []

    for i in range(spot_ret_2d.shape[0]):
        sr = spot_ret_2d[i]
        fr = fut_ret_2d[i]
        bb = basis_2d[i]
        spot_sq_acf1.append(acf_at_lag(sr * sr, 1))
        spot_sq_acf5.append(acf_at_lag(sr * sr, 5))
        fut_sq_acf1.append(acf_at_lag(fr * fr, 1))
        fut_sq_acf5.append(acf_at_lag(fr * fr, 5))
        basis_acf1.append(acf_at_lag(bb, 1))

    metrics["spot_return_squared_acf_1"] = float(np.nanmean(spot_sq_acf1))
    metrics["spot_return_squared_acf_5"] = float(np.nanmean(spot_sq_acf5))
    metrics["futures_return_squared_acf_1"] = float(np.nanmean(fut_sq_acf1))
    metrics["futures_return_squared_acf_5"] = float(np.nanmean(fut_sq_acf5))
    metrics["basis_acf_1"] = float(np.nanmean(basis_acf1))
    return metrics


def comparison_rows(real: dict[str, float], synthetic: dict[str, float], split: str) -> list[dict]:
    rows = []
    for metric, real_value in real.items():
        syn_value = synthetic.get(metric)
        if syn_value is None:
            continue
        abs_diff = float(syn_value - real_value)
        if np.isfinite(real_value) and abs(real_value) > 1e-12:
            relative_error = float(abs_diff / abs(real_value))
        else:
            relative_error = float("nan")
        rows.append(
            {
                "split": split,
                "metric": metric,
                "real": float(real_value),
                "synthetic": float(syn_value),
                "difference": abs_diff,
                "relative_error": relative_error,
            }
        )
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    synthetic_dir = repo_root / "data" / "synthetic"
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_path = processed_dir / "aligned_1d.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing {real_path}.")

    real_df = pd.read_csv(real_path)
    real_metrics = summarize_real(real_df)

    splits = ("train", "validation", "test")
    synthetic_metrics: dict[str, dict[str, float]] = {}
    all_rows: list[dict] = []

    for split in splits:
        path = synthetic_dir / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run 07_generate_joint_synthetic.py first.")
        metrics = summarize_synthetic(path)
        synthetic_metrics[split] = metrics
        all_rows.extend(comparison_rows(real_metrics, metrics, split))

    comparison = pd.DataFrame(all_rows)
    table_path = tables_dir / "synthetic_vs_real_metrics.csv"
    comparison.to_csv(table_path, index=False)

    # Compact decision diagnostics based on the training split.
    train = synthetic_metrics["train"]
    diagnostics = {
        "spot_vol_ratio_syn_to_real": float(train["spot_return_std"] / real_metrics["spot_return_std"]),
        "futures_vol_ratio_syn_to_real": float(train["futures_return_std"] / real_metrics["futures_return_std"]),
        "spot_futures_corr_difference": float(
            train["spot_futures_return_corr"] - real_metrics["spot_futures_return_corr"]
        ),
        "basis_mean_difference": float(train["basis_mean"] - real_metrics["basis_mean"]),
        "basis_std_ratio_syn_to_real": float(train["basis_std"] / real_metrics["basis_std"]),
        "spot_skew_difference": float(train["spot_return_skew"] - real_metrics["spot_return_skew"]),
        "spot_kurtosis_difference": float(
            train["spot_return_excess_kurtosis"] - real_metrics["spot_return_excess_kurtosis"]
        ),
        "spot_squared_acf1_difference": float(
            train["spot_return_squared_acf_1"] - real_metrics["spot_return_squared_acf_1"]
        ),
    }

    # Heuristic, deliberately transparent gates. Passing means "reasonable first
    # simulator", not "statistically identical market".
    gates = {
        "spot_vol_ratio_between_0.90_and_1.10": bool(
            0.90 <= diagnostics["spot_vol_ratio_syn_to_real"] <= 1.10
        ),
        "futures_vol_ratio_between_0.90_and_1.10": bool(
            0.90 <= diagnostics["futures_vol_ratio_syn_to_real"] <= 1.10
        ),
        "return_corr_abs_diff_le_0.05": bool(
            abs(diagnostics["spot_futures_corr_difference"]) <= 0.05
        ),
        "basis_std_ratio_between_0.80_and_1.20": bool(
            0.80 <= diagnostics["basis_std_ratio_syn_to_real"] <= 1.20
        ),
        "spot_squared_acf1_abs_diff_le_0.10": bool(
            abs(diagnostics["spot_squared_acf1_difference"]) <= 0.10
        ),
    }
    all_core_gates_pass = bool(all(gates.values()))

    payload = {
        "real_metrics": real_metrics,
        "synthetic_metrics": synthetic_metrics,
        "training_split_diagnostics": diagnostics,
        "core_validation_gates": gates,
        "all_core_gates_pass": all_core_gates_pass,
        "interpretation": (
            "These gates are engineering diagnostics, not formal goodness-of-fit tests. "
            "Tail-shape mismatches should still be reviewed even when core gates pass."
        ),
    }

    json_path = processed_dir / "synthetic_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    # Plot real vs synthetic train spot-return histograms.
    with np.load(synthetic_dir / "train.npz") as z:
        syn_spot_ret = z["spot_log_return"].astype(np.float64).ravel()
        syn_basis = z["basis"].astype(np.float64)[:, 1:].ravel()

    real_spot_ret = np.diff(np.log(real_df["spot_close"].to_numpy(dtype=float)))
    real_basis = ((real_df["fut_close"] - real_df["spot_close"]) / real_df["spot_close"]).to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 5))
    lo = min(np.quantile(real_spot_ret, 0.005), np.quantile(syn_spot_ret, 0.005))
    hi = max(np.quantile(real_spot_ret, 0.995), np.quantile(syn_spot_ret, 0.995))
    bins = np.linspace(lo, hi, 80)
    plt.hist(real_spot_ret, bins=bins, density=True, alpha=0.5, label="Real VN30")
    plt.hist(syn_spot_ret, bins=bins, density=True, alpha=0.5, label="Synthetic train")
    plt.title("Real vs Synthetic VN30 Daily Log Returns")
    plt.xlabel("Daily log return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    spot_fig = figures_dir / "synthetic_vs_real_spot_returns.png"
    fig.savefig(spot_fig, dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    lo = min(np.quantile(real_basis, 0.005), np.quantile(syn_basis, 0.005))
    hi = max(np.quantile(real_basis, 0.995), np.quantile(syn_basis, 0.995))
    bins = np.linspace(lo, hi, 80)
    plt.hist(real_basis, bins=bins, density=True, alpha=0.5, label="Real basis")
    plt.hist(syn_basis, bins=bins, density=True, alpha=0.5, label="Synthetic train")
    plt.title("Real vs Synthetic VN30F1M Relative Basis")
    plt.xlabel("Basis fraction")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    basis_fig = figures_dir / "synthetic_vs_real_basis.png"
    fig.savefig(basis_fig, dpi=160)
    plt.close(fig)

    print("=" * 94)
    print("SYNTHETIC VS REAL VALIDATION COMPLETE")
    print("=" * 94)
    print(f"Real spot std daily        : {real_metrics['spot_return_std']:.6f}")
    print(f"Synthetic spot std daily   : {train['spot_return_std']:.6f}")
    print(f"Spot vol ratio syn/real    : {diagnostics['spot_vol_ratio_syn_to_real']:.4f}")
    print(f"Real futures std daily     : {real_metrics['futures_return_std']:.6f}")
    print(f"Synthetic futures std      : {train['futures_return_std']:.6f}")
    print(f"Futures vol ratio syn/real : {diagnostics['futures_vol_ratio_syn_to_real']:.4f}")
    print(f"Real corr(S,F)             : {real_metrics['spot_futures_return_corr']:.6f}")
    print(f"Synthetic corr(S,F)        : {train['spot_futures_return_corr']:.6f}")
    print(f"Real basis mean            : {real_metrics['basis_mean']:.6f}")
    print(f"Synthetic basis mean       : {train['basis_mean']:.6f}")
    print(f"Real basis std             : {real_metrics['basis_std']:.6f}")
    print(f"Synthetic basis std        : {train['basis_std']:.6f}")
    print(f"Real spot skew             : {real_metrics['spot_return_skew']:.4f}")
    print(f"Synthetic spot skew        : {train['spot_return_skew']:.4f}")
    print(f"Real spot excess kurtosis  : {real_metrics['spot_return_excess_kurtosis']:.4f}")
    print(f"Synthetic excess kurtosis  : {train['spot_return_excess_kurtosis']:.4f}")
    print(f"Real spot squared ACF(1)   : {real_metrics['spot_return_squared_acf_1']:.4f}")
    print(f"Synthetic squared ACF(1)   : {train['spot_return_squared_acf_1']:.4f}")
    print()
    for name, passed in gates.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print()
    print(f"ALL CORE GATES PASS        : {all_core_gates_pass}")
    print(f"Metrics table              : {table_path}")
    print(f"JSON report                : {json_path}")
    print(f"Spot-return figure         : {spot_fig}")
    print(f"Basis figure               : {basis_fig}")


if __name__ == "__main__":
    main()
