"""Validation-only tuning of the TRAIN-calibrated simulator.

Leakage policy
--------------
- Base Heston/basis parameters come from TRAIN only.
- Tuning targets come from VALIDATION only.
- TEST is never opened.

Why this step exists
--------------------
The untuned candidates miss validation volatility, basis dispersion and skew.
We therefore make only a small, transparent set of adjustments:
1) variance-scale correction derived from validation spot-vol mismatch;
2) basis innovation-scale correction derived from validation basis-std mismatch;
3) a small grid over vol-of-vol scale, rho scale and intraday substeps.

The variance rescaling is internally consistent:
    theta' = a * theta
    v0'    = a * v0
    xi'    = sqrt(a) * xi * xi_scale
so the variance level is changed without arbitrarily altering units.

Intraday substeps still simulate the same Heston SDE; they simply reduce the
one-step Euler discretization bias and allow leverage (rho<0) to feed back
within a trading day.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

TRADING_DAYS = 252.0
SEED = 20261002
N_PATHS = 1200
N_DAYS = 120
XI_SCALES = (0.50, 0.75, 1.00)
RHO_SCALES = (1.0, 2.0, 3.0)
SUBSTEPS = (1, 4, 8)


def acf1_squared_2d(r: np.ndarray) -> float:
    return float(np.corrcoef((r[:, :-1] ** 2).ravel(), (r[:, 1:] ** 2).ravel())[0, 1])


def acf1_2d(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])


def validation_metrics(df: pd.DataFrame) -> dict[str, float]:
    s = df["spot_close"].to_numpy(dtype=float)
    f = df["fut_close"].to_numpy(dtype=float)
    sr = np.diff(np.log(s))
    fr = np.diff(np.log(f))
    basis = (f - s) / s
    return {
        "spot_std": float(np.std(sr, ddof=1)),
        "futures_std": float(np.std(fr, ddof=1)),
        "spot_skew": float(skew(sr, bias=False)),
        "spot_excess_kurtosis": float(kurtosis(sr, fisher=True, bias=False)),
        "spot_squared_acf1": float(np.corrcoef(sr[:-1] ** 2, sr[1:] ** 2)[0, 1]),
        "spot_futures_corr": float(np.corrcoef(sr, fr)[0, 1]),
        "basis_mean": float(np.mean(basis)),
        "basis_std": float(np.std(basis, ddof=1)),
        "basis_acf1": float(np.corrcoef(basis[:-1], basis[1:])[0, 1]),
    }


def initial_variance_from_train(train: pd.DataFrame, window: int) -> float:
    log_hl = np.log(train["spot_high"] / train["spot_low"])
    log_co = np.log(train["spot_close"] / train["spot_open"])
    gk = (0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)).clip(lower=0.0)
    if window == 1:
        daily = gk
    else:
        daily = gk.rolling(window, min_periods=window).mean()
    return float(max(daily.iloc[-1] * TRADING_DAYS, 1e-8))


def simulate(
    h: dict,
    basis_params: dict,
    s0: float,
    b0: float,
    v0_base: float,
    variance_scale: float,
    basis_sigma_scale: float,
    xi_scale: float,
    rho_scale: float,
    substeps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    mu = float(h["mu"])
    kappa = float(h["kappa"])
    theta = float(h["theta"]) * variance_scale
    xi = float(h["xi"]) * np.sqrt(variance_scale) * xi_scale
    rho = float(np.clip(float(h["rho"]) * rho_scale, -0.95, 0.95))
    v0 = float(v0_base * variance_scale)

    c = float(basis_params["c"])
    phi = float(basis_params["phi"])
    sigma_b = float(basis_params["residual_std_daily"]) * basis_sigma_scale

    spot = np.empty((N_PATHS, N_DAYS + 1), dtype=np.float64)
    var = np.empty((N_PATHS, N_DAYS + 1), dtype=np.float64)
    basis = np.empty((N_PATHS, N_DAYS + 1), dtype=np.float64)
    fut = np.empty((N_PATHS, N_DAYS + 1), dtype=np.float64)

    spot[:, 0] = s0
    var[:, 0] = v0
    basis[:, 0] = b0
    fut[:, 0] = s0 * (1.0 + b0)

    dt_sub = 1.0 / (TRADING_DAYS * substeps)
    sqrt_dt_sub = np.sqrt(dt_sub)
    rho_scale_ind = np.sqrt(max(1.0 - rho * rho, 0.0))

    for day in range(N_DAYS):
        s = spot[:, day].copy()
        v = var[:, day].copy()

        for _ in range(substeps):
            z_s = rng.standard_normal(N_PATHS)
            z_i = rng.standard_normal(N_PATHS)
            z_v = rho * z_s + rho_scale_ind * z_i

            v_pos = np.maximum(v, 0.0)
            v_next = (
                v
                + kappa * (theta - v_pos) * dt_sub
                + xi * np.sqrt(v_pos) * sqrt_dt_sub * z_v
            )
            v_next = np.maximum(v_next, 0.0)

            s = s * np.exp(
                (mu - 0.5 * v_pos) * dt_sub
                + np.sqrt(v_pos) * sqrt_dt_sub * z_s
            )
            v = v_next

        z_b = rng.standard_normal(N_PATHS)
        b_next = c + phi * basis[:, day] + sigma_b * z_b
        b_next = np.maximum(b_next, -0.95)

        spot[:, day + 1] = s
        var[:, day + 1] = v
        basis[:, day + 1] = b_next
        fut[:, day + 1] = s * (1.0 + b_next)

    sr = np.diff(np.log(spot), axis=1)
    fr = np.diff(np.log(fut), axis=1)
    return {"spot_return": sr, "futures_return": fr, "basis": basis[:, 1:]}


def metrics(sim: dict[str, np.ndarray]) -> dict[str, float]:
    sr2d = sim["spot_return"]
    fr2d = sim["futures_return"]
    b2d = sim["basis"]
    sr = sr2d.ravel()
    fr = fr2d.ravel()
    b = b2d.ravel()
    return {
        "spot_std": float(np.std(sr, ddof=1)),
        "futures_std": float(np.std(fr, ddof=1)),
        "spot_skew": float(skew(sr, bias=False)),
        "spot_excess_kurtosis": float(kurtosis(sr, fisher=True, bias=False)),
        "spot_squared_acf1": acf1_squared_2d(sr2d),
        "spot_futures_corr": float(np.corrcoef(sr, fr)[0, 1]),
        "basis_mean": float(np.mean(b)),
        "basis_std": float(np.std(b, ddof=1)),
        "basis_acf1": acf1_2d(b2d),
    }


def score(target: dict[str, float], syn: dict[str, float]) -> float:
    parts = {
        "spot_vol": abs(syn["spot_std"] / target["spot_std"] - 1.0),
        "futures_vol": abs(syn["futures_std"] / target["futures_std"] - 1.0),
        "corr": abs(syn["spot_futures_corr"] - target["spot_futures_corr"]) / 0.05,
        "basis_std": abs(syn["basis_std"] / target["basis_std"] - 1.0),
        "basis_mean": abs(syn["basis_mean"] - target["basis_mean"]) / max(target["basis_std"], 1e-8),
        "skew": abs(syn["spot_skew"] - target["spot_skew"]) / 0.50,
        "kurt": abs(syn["spot_excess_kurtosis"] - target["spot_excess_kurtosis"]) / 2.0,
        "sq_acf1": abs(syn["spot_squared_acf1"] - target["spot_squared_acf1"]) / 0.10,
        "basis_acf1": abs(syn["basis_acf1"] - target["basis_acf1"]) / 0.10,
    }
    weights = {
        "spot_vol": 2.0,
        "futures_vol": 2.0,
        "corr": 1.0,
        "basis_std": 1.0,
        "basis_mean": 0.5,
        "skew": 1.5,
        "kurt": 1.0,
        "sq_acf1": 1.5,
        "basis_acf1": 0.5,
    }
    return float(sum(weights[k] * parts[k] for k in parts) / sum(weights.values()))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    calibration = processed / "calibration"
    splits = processed / "splits"
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    with open(calibration / "train_heston_candidates.json", "r", encoding="utf-8") as f:
        heston_payload = json.load(f)
    with open(calibration / "train_basis_params.json", "r", encoding="utf-8") as f:
        basis_payload = json.load(f)
    with open(calibration / "validation_simulator_selection.json", "r", encoding="utf-8") as f:
        untuned_payload = json.load(f)

    train = pd.read_csv(splits / "train.csv")
    validation = pd.read_csv(splits / "validation.csv")

    target = validation_metrics(validation)
    basis_params = basis_payload["results"]["filtered"]

    s0 = float(train["spot_close"].iloc[-1])
    f0 = float(train["fut_close"].iloc[-1])
    b0 = float((f0 - s0) / s0)

    untuned_by_window = {int(r["window"]): r for r in untuned_payload["ranking"]}

    rows: list[dict] = []
    candidate_details: dict[str, dict] = {}

    for h in heston_payload["results"]:
        window = int(h["window"])
        untuned = untuned_by_window[window]

        # Validation-derived level corrections. These are hyperparameter tuning,
        # not re-estimation of the structural TRAIN parameters.
        variance_scale = float(np.clip(1.0 / (untuned["spot_vol_ratio"] ** 2), 0.30, 1.50))
        basis_sigma_scale = float(np.clip(1.0 / untuned["basis_std_ratio"], 0.20, 1.50))
        v0_base = initial_variance_from_train(train, window)

        for xi_scale in XI_SCALES:
            for rho_scale in RHO_SCALES:
                for substeps in SUBSTEPS:
                    config_seed = SEED + window * 1000 + int(xi_scale * 100) * 10 + int(rho_scale * 10) + substeps
                    sim = simulate(
                        h=h,
                        basis_params=basis_params,
                        s0=s0,
                        b0=b0,
                        v0_base=v0_base,
                        variance_scale=variance_scale,
                        basis_sigma_scale=basis_sigma_scale,
                        xi_scale=xi_scale,
                        rho_scale=rho_scale,
                        substeps=substeps,
                        seed=config_seed,
                    )
                    syn = metrics(sim)
                    sc = score(target, syn)

                    row = {
                        "window": window,
                        "score": sc,
                        "variance_scale": variance_scale,
                        "basis_sigma_scale": basis_sigma_scale,
                        "xi_scale": float(xi_scale),
                        "rho_scale": float(rho_scale),
                        "substeps": int(substeps),
                        "rho_effective": float(np.clip(float(h["rho"]) * rho_scale, -0.95, 0.95)),
                        "spot_vol_ratio": syn["spot_std"] / target["spot_std"],
                        "futures_vol_ratio": syn["futures_std"] / target["futures_std"],
                        "spot_skew_syn": syn["spot_skew"],
                        "spot_kurt_syn": syn["spot_excess_kurtosis"],
                        "spot_sq_acf1_syn": syn["spot_squared_acf1"],
                        "corr_syn": syn["spot_futures_corr"],
                        "basis_std_ratio": syn["basis_std"] / target["basis_std"],
                        "basis_mean_syn": syn["basis_mean"],
                        "basis_acf1_syn": syn["basis_acf1"],
                    }
                    rows.append(row)

    result = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    best = result.iloc[0]

    gates = {
        "spot_vol_ratio_0.90_1.10": bool(0.90 <= best["spot_vol_ratio"] <= 1.10),
        "futures_vol_ratio_0.90_1.10": bool(0.90 <= best["futures_vol_ratio"] <= 1.10),
        "corr_abs_diff_le_0.05": bool(abs(best["corr_syn"] - target["spot_futures_corr"]) <= 0.05),
        "basis_std_ratio_0.80_1.20": bool(0.80 <= best["basis_std_ratio"] <= 1.20),
        "spot_sq_acf1_abs_diff_le_0.10": bool(
            abs(best["spot_sq_acf1_syn"] - target["spot_squared_acf1"]) <= 0.10
        ),
        "spot_skew_abs_diff_le_0.30": bool(abs(best["spot_skew_syn"] - target["spot_skew"]) <= 0.30),
        "spot_kurt_abs_diff_le_1.50": bool(abs(best["spot_kurt_syn"] - target["spot_excess_kurtosis"]) <= 1.50),
    }

    needs_tail_extension = bool(
        (not gates["spot_skew_abs_diff_le_0.30"])
        or (not gates["spot_kurt_abs_diff_le_1.50"])
    )

    table_path = tables / "validation_simulator_tuning_grid.csv"
    result.to_csv(table_path, index=False)

    best_config = {
        "window": int(best["window"]),
        "variance_scale": float(best["variance_scale"]),
        "basis_sigma_scale": float(best["basis_sigma_scale"]),
        "xi_scale": float(best["xi_scale"]),
        "rho_scale": float(best["rho_scale"]),
        "rho_effective": float(best["rho_effective"]),
        "substeps": int(best["substeps"]),
    }

    report = {
        "leakage_policy": {
            "base_parameters": "train_only",
            "tuning": "validation_only",
            "test_used": False,
        },
        "tuning_grid": {
            "xi_scales": list(XI_SCALES),
            "rho_scales": list(RHO_SCALES),
            "substeps": list(SUBSTEPS),
            "n_paths": N_PATHS,
            "n_days": N_DAYS,
        },
        "validation_target": target,
        "best_config": best_config,
        "best_metrics": {k: float(best[k]) for k in result.columns if k not in {"window", "substeps"}},
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "needs_tail_extension": needs_tail_extension,
        "top10": result.head(10).to_dict(orient="records"),
        "next_step": (
            "freeze tuned Heston+basis simulator"
            if all(gates.values())
            else (
                "standard Heston still misses skew/tails; evaluate a documented Heston jump/skew extension on validation only"
                if needs_tail_extension
                else "refine a small validation-only tuning grid"
            )
        ),
    }

    report_path = calibration / "validation_simulator_tuning.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 132)
    print("VALIDATION-ONLY SIMULATOR TUNING COMPLETE")
    print("=" * 132)
    print("Top 10 configurations:")
    print(
        result.head(10)[
            [
                "window",
                "score",
                "variance_scale",
                "basis_sigma_scale",
                "xi_scale",
                "rho_scale",
                "rho_effective",
                "substeps",
                "spot_vol_ratio",
                "futures_vol_ratio",
                "spot_skew_syn",
                "spot_kurt_syn",
                "spot_sq_acf1_syn",
                "corr_syn",
                "basis_std_ratio",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print()
    print("VALIDATION TARGET:")
    print(f"spot skew         : {target['spot_skew']:.6f}")
    print(f"spot excess kurt  : {target['spot_excess_kurtosis']:.6f}")
    print(f"spot squared ACF1 : {target['spot_squared_acf1']:.6f}")
    print(f"corr(S,F)         : {target['spot_futures_corr']:.6f}")
    print()
    print("Best config:")
    for k, v in best_config.items():
        print(f"  {k:20s}: {v}")
    print()
    for name, passed in gates.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL GATES PASS       : {all(gates.values())}")
    print(f"NEEDS TAIL EXTENSION : {needs_tail_extension}")
    print(f"Grid table           : {table_path}")
    print(f"Report               : {report_path}")
    print()
    print("TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
