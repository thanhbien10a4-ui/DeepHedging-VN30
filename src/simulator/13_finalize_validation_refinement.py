"""Final validation-only refinement around the best standard-Heston family.

This is deliberately the LAST parameter search on validation before freezing the
simulator. Its purpose is to correct the remaining under-volatility / excessive
volatility-clustering trade-off found in step 12 without endlessly overfitting
validation.

Leakage policy
--------------
- Structural Heston and basis parameters: TRAIN only.
- Hyperparameter refinement: VALIDATION only.
- TEST is never opened.

Search family
-------------
The step-12 best family used the TRAIN-calibrated 1-day GK Heston candidate.
We keep that structural candidate fixed and refine only:
- variance_scale around the step-12 best value;
- xi_scale below the previous lower bound to reduce squared-return ACF;
- rho_scale around the strong negative-leverage region;
- 4/8 intraday substeps.

The basis innovation scale is frozen at the step-12 best value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

TRADING_DAYS = 252.0
SEED = 20261003
N_PATHS = 1400
N_DAYS = 120

VARIANCE_MULTIPLIERS = (1.20, 1.35, 1.50, 1.65, 1.80)
XI_SCALES = (0.20, 0.30, 0.40, 0.50)
RHO_SCALES = (3.0, 3.5, 4.0)
SUBSTEPS = (4, 8)


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
    daily = gk if window == 1 else gk.rolling(window, min_periods=window).mean()
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

    dt = 1.0 / (TRADING_DAYS * substeps)
    sqrt_dt = np.sqrt(dt)
    rho_ind = np.sqrt(max(1.0 - rho * rho, 0.0))

    for day in range(N_DAYS):
        s = spot[:, day].copy()
        v = var[:, day].copy()

        for _ in range(substeps):
            z_s = rng.standard_normal(N_PATHS)
            z_i = rng.standard_normal(N_PATHS)
            z_v = rho * z_s + rho_ind * z_i

            v_pos = np.maximum(v, 0.0)
            v_next = (
                v
                + kappa * (theta - v_pos) * dt
                + xi * np.sqrt(v_pos) * sqrt_dt * z_v
            )
            v_next = np.maximum(v_next, 0.0)

            s = s * np.exp(
                (mu - 0.5 * v_pos) * dt
                + np.sqrt(v_pos) * sqrt_dt * z_s
            )
            v = v_next

        z_b = rng.standard_normal(N_PATHS)
        b_next = c + phi * basis[:, day] + sigma_b * z_b
        b_next = np.maximum(b_next, -0.95)

        spot[:, day + 1] = s
        var[:, day + 1] = v
        basis[:, day + 1] = b_next
        fut[:, day + 1] = s * (1.0 + b_next)

    return {
        "spot_return": np.diff(np.log(spot), axis=1),
        "futures_return": np.diff(np.log(fut), axis=1),
        "basis": basis[:, 1:],
    }


def summarize(sim: dict[str, np.ndarray]) -> dict[str, float]:
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
        "skew": abs(syn["spot_skew"] - target["spot_skew"]) / 0.30,
        "kurt": abs(syn["spot_excess_kurtosis"] - target["spot_excess_kurtosis"]) / 1.50,
        "sq_acf1": abs(syn["spot_squared_acf1"] - target["spot_squared_acf1"]) / 0.10,
        "basis_acf1": abs(syn["basis_acf1"] - target["basis_acf1"]) / 0.10,
    }
    weights = {
        "spot_vol": 2.5,
        "futures_vol": 2.5,
        "corr": 1.0,
        "basis_std": 1.0,
        "basis_mean": 0.5,
        "skew": 1.5,
        "kurt": 1.0,
        "sq_acf1": 2.0,
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
    with open(calibration / "validation_simulator_tuning.json", "r", encoding="utf-8") as f:
        previous = json.load(f)

    train = pd.read_csv(splits / "train.csv")
    validation = pd.read_csv(splits / "validation.csv")
    target = validation_metrics(validation)

    previous_best = previous["best_config"]
    base_window = int(previous_best["window"])
    if base_window != 1:
        raise ValueError(
            f"Expected step-12 best family to be window=1, got {base_window}. "
            "Review step 12 before running final refinement."
        )

    h_matches = [h for h in heston_payload["results"] if int(h["window"]) == base_window]
    if len(h_matches) != 1:
        raise ValueError("Could not identify the unique TRAIN-calibrated 1-day Heston candidate.")
    h = h_matches[0]
    basis_params = basis_payload["results"]["filtered"]

    base_variance_scale = float(previous_best["variance_scale"])
    basis_sigma_scale = float(previous_best["basis_sigma_scale"])

    s0 = float(train["spot_close"].iloc[-1])
    f0 = float(train["fut_close"].iloc[-1])
    b0 = float((f0 - s0) / s0)
    v0_base = initial_variance_from_train(train, base_window)

    rows: list[dict] = []

    for mult in VARIANCE_MULTIPLIERS:
        variance_scale = base_variance_scale * mult
        for xi_scale in XI_SCALES:
            for rho_scale in RHO_SCALES:
                for substeps in SUBSTEPS:
                    seed = (
                        SEED
                        + int(mult * 1000)
                        + int(xi_scale * 100) * 10
                        + int(rho_scale * 10)
                        + substeps
                    )
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
                        seed=seed,
                    )
                    syn = summarize(sim)
                    rows.append(
                        {
                            "window": base_window,
                            "score": score(target, syn),
                            "variance_multiplier": mult,
                            "variance_scale": variance_scale,
                            "basis_sigma_scale": basis_sigma_scale,
                            "xi_scale": xi_scale,
                            "rho_scale": rho_scale,
                            "rho_effective": float(np.clip(float(h["rho"]) * rho_scale, -0.95, 0.95)),
                            "substeps": substeps,
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
                    )

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

    table_path = tables / "validation_final_refinement_grid.csv"
    result.to_csv(table_path, index=False)

    frozen_config = {
        "window": int(best["window"]),
        "variance_scale": float(best["variance_scale"]),
        "basis_sigma_scale": float(best["basis_sigma_scale"]),
        "xi_scale": float(best["xi_scale"]),
        "rho_scale": float(best["rho_scale"]),
        "rho_effective": float(best["rho_effective"]),
        "substeps": int(best["substeps"]),
    }

    report = {
        "status": "final_validation_refinement_complete",
        "policy": {
            "structural_parameters": "train_only",
            "selection_and_tuning": "validation_only",
            "test_used": False,
            "further_validation_parameter_search_after_this_step": False,
        },
        "validation_target": target,
        "search_grid": {
            "variance_multipliers": list(VARIANCE_MULTIPLIERS),
            "xi_scales": list(XI_SCALES),
            "rho_scales": list(RHO_SCALES),
            "substeps": list(SUBSTEPS),
            "basis_sigma_scale_frozen": basis_sigma_scale,
            "n_paths": N_PATHS,
            "n_days": N_DAYS,
        },
        "frozen_config": frozen_config,
        "frozen_metrics": {k: float(best[k]) for k in result.columns if k not in {"window", "substeps"}},
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "top10": result.head(10).to_dict(orient="records"),
        "interpretation": (
            "This is the final validation tuning pass. Freeze this standard-Heston configuration. "
            "Any remaining mismatch must be documented or addressed by a separately named model extension, "
            "not by continuing to search validation hyperparameters."
        ),
    }

    report_path = calibration / "frozen_simulator_config.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 136)
    print("FINAL VALIDATION REFINEMENT COMPLETE — SIMULATOR CONFIG FROZEN")
    print("=" * 136)
    print("Top 10 configurations:")
    print(
        result.head(10)[
            [
                "score",
                "variance_scale",
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
    print("Frozen config:")
    for k, v in frozen_config.items():
        print(f"  {k:20s}: {v}")
    print()
    for name, passed in gates.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL GATES PASS : {all(gates.values())}")
    print(f"Grid table     : {table_path}")
    print(f"Frozen report  : {report_path}")
    print()
    print("IMPORTANT: no more validation hyperparameter search after this step.")
    print("TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
