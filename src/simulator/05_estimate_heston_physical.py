"""Approximate physical-measure Heston parameter estimation from VN30 spot data.

This is NOT risk-neutral calibration from option prices.
With only VN30 OHLCV and VN30F1M, the latent variance process is approximated
using a Garman-Klass daily variance proxy constructed from VN30 spot OHLC.

Model (physical measure):
    dS_t / S_t = mu dt + sqrt(v_t) dW^S_t
    dv_t       = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t
    corr(dW^S_t, dW^v_t) = rho

Outputs:
    data/processed/heston_physical_params.json
    data/processed/heston_variance_proxy.csv
    results/figures/heston_variance_proxy.png

Notes:
- Parameters are annualized.
- v_t, theta and v0 are annualized variance units.
- kappa is per year.
- xi is volatility-of-variance in annualized units.
- rho is the empirical correlation between standardized price/variance shocks.
- The estimate is approximate because variance is latent and proxied from daily OHLC.
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


def garman_klass_daily_variance(df: pd.DataFrame) -> pd.Series:
    """Daily (not annualized) Garman-Klass variance estimate."""
    log_hl = np.log(df["spot_high"] / df["spot_low"])
    log_co = np.log(df["spot_close"] / df["spot_open"])
    var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    return var.clip(lower=0.0)


def conditional_mean(v_t: np.ndarray, kappa: float, theta: float) -> np.ndarray:
    e = np.exp(-kappa * DT)
    return theta + (v_t - theta) * e


def cir_variance_coefficient(v_t: np.ndarray, kappa: float, theta: float) -> np.ndarray:
    """Return coefficient C where Var(v_{t+dt}|v_t) = xi^2 * C."""
    e = np.exp(-kappa * DT)
    one_minus_e = 1.0 - e
    c1 = v_t * e * one_minus_e / kappa
    c2 = theta * (one_minus_e**2) / (2.0 * kappa)
    return np.maximum(c1 + c2, VAR_FLOOR)


def estimate_kappa_theta(v_t: np.ndarray, v_next: np.ndarray) -> tuple[float, float, dict]:
    """Fit CIR conditional mean with bounded nonlinear least squares."""
    theta0_raw = float(np.nanmean(v_t))

    # Rough persistence-based initial guess.
    # With a noisy daily variance proxy the implied annualized kappa can easily
    # exceed 100, so use a wider upper bound and always clip x0 inside bounds.
    corr = np.corrcoef(v_t, v_next)[0, 1]
    if np.isfinite(corr) and 0.0 < corr < 0.999999:
        kappa0_raw = max(-np.log(corr) / DT, 0.1)
    else:
        kappa0_raw = 2.0

    lower = np.array([1e-4, 1e-8], dtype=float)
    upper = np.array([1000.0, 4.0], dtype=float)

    x0_raw = np.array([kappa0_raw, max(theta0_raw, 1e-6)], dtype=float)
    # scipy.least_squares requires the starting point to be strictly feasible.
    eps = np.array([1e-8, 1e-12], dtype=float)
    x0 = np.minimum(np.maximum(x0_raw, lower + eps), upper - eps)

    def residuals(params: np.ndarray) -> np.ndarray:
        kappa, theta = params
        mean = conditional_mean(v_t, kappa, theta)
        # Approximate heteroskedastic weighting by current variance level.
        scale = np.sqrt(np.maximum(v_t, VAR_FLOOR))
        return (v_next - mean) / scale

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=20_000,
    )

    kappa, theta = map(float, result.x)
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "lag1_variance_proxy_corr": float(corr) if np.isfinite(corr) else None,
        "initial_kappa_raw": float(kappa0_raw),
        "initial_theta_raw": float(theta0_raw),
        "initial_kappa_used": float(x0[0]),
        "initial_theta_used": float(x0[1]),
        "kappa_upper_bound": float(upper[0]),
        "theta_upper_bound": float(upper[1]),
    }
    return kappa, theta, diagnostics


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    figures_dir = repo_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "aligned_1d.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run 01_validate_and_align.py first."
        )

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    # Spot returns only. Futures rollover diagnostics do not alter this estimator.
    df["spot_log_return"] = np.log(df["spot_close"]).diff()

    # Daily GK variance -> annualized variance proxy for v_t.
    df["gk_daily_variance"] = garman_klass_daily_variance(df)
    df["v_proxy"] = (df["gk_daily_variance"] * TRADING_DAYS).clip(lower=VAR_FLOOR)

    # Same interval alignment:
    # return_t is from t-1 to t, so v_start is v_{t-1}; dv is v_t-v_{t-1}.
    df["v_start"] = df["v_proxy"].shift(1)
    df["dv"] = df["v_proxy"] - df["v_start"]

    est = df[["time", "spot_log_return", "v_proxy", "v_start", "dv"]].dropna().copy()

    v_t = est["v_start"].to_numpy(dtype=float)
    v_next = est["v_proxy"].to_numpy(dtype=float)

    kappa, theta, opt_diag = estimate_kappa_theta(v_t, v_next)

    mean_v_next = conditional_mean(v_t, kappa, theta)
    var_resid = v_next - mean_v_next
    c = cir_variance_coefficient(v_t, kappa, theta)

    # Estimate xi^2 from standardized conditional variance residuals.
    xi2 = float(np.mean((var_resid**2) / c))
    xi = float(np.sqrt(max(xi2, 0.0)))

    # Annualized physical drift. Under log-price Heston:
    # E[r_t] approx (mu - 0.5 v_t) dt.
    r = est["spot_log_return"].to_numpy(dtype=float)
    mu = float(np.mean(r) / DT + 0.5 * np.mean(v_t))

    # Standardized contemporaneous shocks for rho.
    eps_s_denom = np.sqrt(np.maximum(v_t * DT, VAR_FLOOR))
    eps_s = (r - (mu - 0.5 * v_t) * DT) / eps_s_denom

    if xi > 0:
        cond_var = np.maximum((xi**2) * c, VAR_FLOOR)
        eps_v = var_resid / np.sqrt(cond_var)
        valid = np.isfinite(eps_s) & np.isfinite(eps_v)
        rho = float(np.corrcoef(eps_s[valid], eps_v[valid])[0, 1])
    else:
        rho = float("nan")

    v0_latest = float(est["v_proxy"].iloc[-1])
    long_run_vol = float(np.sqrt(theta))
    latest_vol = float(np.sqrt(v0_latest))
    half_life_years = float(np.log(2.0) / kappa)
    half_life_trading_days = float(half_life_years * TRADING_DAYS)

    feller_lhs = float(2.0 * kappa * theta)
    feller_rhs = float(xi**2)
    feller_ratio = float(feller_lhs / feller_rhs) if feller_rhs > 0 else float("inf")
    feller_satisfied = bool(feller_lhs > feller_rhs)

    # Diagnostics on variance proxy fit.
    rmse = float(np.sqrt(np.mean(var_resid**2)))
    mae = float(np.mean(np.abs(var_resid)))
    persistence = float(np.exp(-kappa * DT))

    output = {
        "estimation_type": "physical_measure_proxy_estimation",
        "warning": (
            "This is not risk-neutral Heston calibration from option prices. "
            "Variance is latent and approximated with daily Garman-Klass variance from VN30 spot OHLC."
        ),
        "sample": {
            "rows_used": int(len(est)),
            "date_min": est["time"].min().strftime("%Y-%m-%d"),
            "date_max": est["time"].max().strftime("%Y-%m-%d"),
            "dt_years": DT,
            "trading_days_per_year": int(TRADING_DAYS),
        },
        "parameters": {
            "mu": mu,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "v0_latest": v0_latest,
        },
        "derived": {
            "long_run_volatility": long_run_vol,
            "latest_volatility_proxy": latest_vol,
            "variance_persistence_1d": persistence,
            "variance_half_life_years": half_life_years,
            "variance_half_life_trading_days": half_life_trading_days,
            "feller_lhs_2kappa_theta": feller_lhs,
            "feller_rhs_xi_squared": feller_rhs,
            "feller_ratio": feller_ratio,
            "feller_satisfied": feller_satisfied,
        },
        "fit_diagnostics": {
            "variance_conditional_mean_rmse": rmse,
            "variance_conditional_mean_mae": mae,
            "variance_proxy_mean": float(np.mean(v_t)),
            "variance_proxy_median": float(np.median(v_t)),
            "variance_proxy_std": float(np.std(v_t, ddof=1)),
            "optimizer": opt_diag,
        },
        "simulation_guidance": {
            "preferred_scheme": (
                "full_truncation_euler_or_qe_if_feller_violated"
                if not feller_satisfied
                else "full_truncation_euler_or_qe"
            ),
            "do_not_use_futures_roll_flags_to_delete_spot_returns": True,
            "basis_process_should_be_modelled_separately": True,
        },
    }

    params_path = processed_dir / "heston_physical_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

    proxy_cols = [
        "time",
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_close",
        "spot_log_return",
        "gk_daily_variance",
        "v_proxy",
    ]
    proxy_path = processed_dir / "heston_variance_proxy.csv"
    df[proxy_cols].to_csv(proxy_path, index=False, date_format="%Y-%m-%d")

    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], np.sqrt(df["v_proxy"]), linewidth=0.8)
    plt.axhline(long_run_vol, linestyle="--", linewidth=1.0, label="sqrt(theta)")
    plt.title("VN30 Garman-Klass Volatility Proxy and Estimated Heston Long-Run Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "heston_variance_proxy.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 72)
    print("PHYSICAL HESTON PARAMETER ESTIMATION COMPLETE")
    print("=" * 72)
    print(f"Rows used               : {len(est):,}")
    print(f"mu                      : {mu:.6f}")
    print(f"kappa                   : {kappa:.6f}")
    print(f"theta                   : {theta:.6f}")
    print(f"xi                      : {xi:.6f}")
    print(f"rho                     : {rho:.6f}")
    print(f"v0 latest               : {v0_latest:.6f}")
    print(f"Long-run volatility     : {long_run_vol:.4%}")
    print(f"Latest volatility proxy : {latest_vol:.4%}")
    print(f"Variance half-life      : {half_life_trading_days:.2f} trading days")
    print(f"Feller ratio            : {feller_ratio:.6f}")
    print(f"Feller satisfied        : {feller_satisfied}")
    print(f"Raw kappa start         : {opt_diag['initial_kappa_raw']:.6f}")
    print(f"Used kappa start        : {opt_diag['initial_kappa_used']:.6f}")
    print(f"Parameters              : {params_path}")
    print(f"Variance proxy          : {proxy_path}")
    print(f"Figure                  : {figure_path}")
    print()
    print("IMPORTANT: These are physical-measure proxy estimates, not option-implied risk-neutral parameters.")


if __name__ == "__main__":
    main()
