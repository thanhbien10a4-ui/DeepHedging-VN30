"""Select among TRAIN-calibrated simulator candidates using VALIDATION only.

Leakage policy
--------------
- Heston and basis parameters come only from TRAIN calibration outputs.
- Candidate selection uses only data/processed/splits/validation.csv.
- data/processed/splits/test.csv is never opened by this script.

This step compares the 1/5/10/20-day GK Heston candidates without tuning them.
If every candidate is materially misspecified, a later validation-only tuning
step can adjust a small number of simulator hyperparameters. TEST remains sealed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS
SEED = 20261001
N_PATHS = 4000


def acf1_squared_2d(r: np.ndarray) -> float:
    x = r[:, :-1] ** 2
    y = r[:, 1:] ** 2
    x = x.ravel()
    y = y.ravel()
    return float(np.corrcoef(x, y)[0, 1])


def acf1_2d(x: np.ndarray) -> float:
    a = x[:, :-1].ravel()
    b = x[:, 1:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def real_metrics(validation: pd.DataFrame) -> dict[str, float]:
    s = validation["spot_close"].to_numpy(dtype=float)
    f = validation["fut_close"].to_numpy(dtype=float)
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
    v0: float,
    n_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    mu = float(h["mu"])
    kappa = float(h["kappa"])
    theta = float(h["theta"])
    xi = float(h["xi"])
    rho = float(h["rho"])

    c = float(basis_params["c"])
    phi = float(basis_params["phi"])
    sigma_b = float(basis_params["residual_std_daily"])

    spot = np.empty((N_PATHS, n_steps + 1), dtype=np.float64)
    var = np.empty((N_PATHS, n_steps + 1), dtype=np.float64)
    basis = np.empty((N_PATHS, n_steps + 1), dtype=np.float64)
    fut = np.empty((N_PATHS, n_steps + 1), dtype=np.float64)

    spot[:, 0] = s0
    var[:, 0] = v0
    basis[:, 0] = b0
    fut[:, 0] = s0 * (1.0 + b0)

    sqrt_dt = np.sqrt(DT)
    rho_scale = np.sqrt(max(1.0 - rho * rho, 0.0))

    for t in range(n_steps):
        z_s = rng.standard_normal(N_PATHS)
        z_i = rng.standard_normal(N_PATHS)
        z_v = rho * z_s + rho_scale * z_i
        z_b = rng.standard_normal(N_PATHS)

        s_t = spot[:, t]
        v_t = var[:, t]
        v_pos = np.maximum(v_t, 0.0)

        v_next = v_t + kappa * (theta - v_pos) * DT + xi * np.sqrt(v_pos) * sqrt_dt * z_v
        v_next = np.maximum(v_next, 0.0)

        s_next = s_t * np.exp(
            (mu - 0.5 * v_pos) * DT + np.sqrt(v_pos) * sqrt_dt * z_s
        )
        b_next = c + phi * basis[:, t] + sigma_b * z_b
        b_next = np.maximum(b_next, -0.95)
        f_next = s_next * (1.0 + b_next)

        spot[:, t + 1] = s_next
        var[:, t + 1] = v_next
        basis[:, t + 1] = b_next
        fut[:, t + 1] = f_next

    sr = np.diff(np.log(spot), axis=1)
    fr = np.diff(np.log(fut), axis=1)
    return {"spot_return": sr, "futures_return": fr, "basis": basis[:, 1:]}


def synthetic_metrics(sim: dict[str, np.ndarray]) -> dict[str, float]:
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


def score_candidate(real: dict[str, float], syn: dict[str, float]) -> tuple[float, dict[str, float]]:
    # Transparent normalized errors. Lower is better.
    components = {
        "spot_vol": abs(syn["spot_std"] / real["spot_std"] - 1.0),
        "futures_vol": abs(syn["futures_std"] / real["futures_std"] - 1.0),
        "corr": abs(syn["spot_futures_corr"] - real["spot_futures_corr"]) / 0.05,
        "basis_std": abs(syn["basis_std"] / real["basis_std"] - 1.0),
        "basis_mean": abs(syn["basis_mean"] - real["basis_mean"]) / max(real["basis_std"], 1e-8),
        "skew": abs(syn["spot_skew"] - real["spot_skew"]) / 0.50,
        "kurtosis": abs(syn["spot_excess_kurtosis"] - real["spot_excess_kurtosis"]) / 2.0,
        "sq_acf1": abs(syn["spot_squared_acf1"] - real["spot_squared_acf1"]) / 0.10,
        "basis_acf1": abs(syn["basis_acf1"] - real["basis_acf1"]) / 0.10,
    }
    weights = {
        "spot_vol": 2.0,
        "futures_vol": 2.0,
        "corr": 1.0,
        "basis_std": 1.0,
        "basis_mean": 0.5,
        "skew": 1.5,
        "kurtosis": 1.5,
        "sq_acf1": 2.0,
        "basis_acf1": 0.5,
    }
    total = sum(weights[k] * components[k] for k in components) / sum(weights.values())
    return float(total), components


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    calibration = processed / "calibration"
    splits = processed / "splits"
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    heston_path = calibration / "train_heston_candidates.json"
    basis_path = calibration / "train_basis_params.json"
    train_path = splits / "train.csv"
    validation_path = splits / "validation.csv"

    for p in (heston_path, basis_path, train_path, validation_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    with open(heston_path, "r", encoding="utf-8") as f:
        heston_payload = json.load(f)
    with open(basis_path, "r", encoding="utf-8") as f:
        basis_payload = json.load(f)

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    train["time"] = pd.to_datetime(train["time"])
    validation["time"] = pd.to_datetime(validation["time"])

    basis_params = basis_payload["results"]["filtered"]
    target = real_metrics(validation)

    s0 = float(train["spot_close"].iloc[-1])
    f0 = float(train["fut_close"].iloc[-1])
    b0 = float((f0 - s0) / s0)
    n_steps = len(validation)

    rows: list[dict] = []
    detailed: dict[str, dict] = {}

    for h in heston_payload["results"]:
        window = int(h["window"])
        v0 = initial_variance_from_train(train, window)
        sim = simulate(h, basis_params, s0, b0, v0, n_steps, SEED)
        syn = synthetic_metrics(sim)
        score, components = score_candidate(target, syn)

        row = {
            "window": window,
            "score": score,
            "spot_vol_ratio": syn["spot_std"] / target["spot_std"],
            "futures_vol_ratio": syn["futures_std"] / target["futures_std"],
            "spot_skew_real": target["spot_skew"],
            "spot_skew_syn": syn["spot_skew"],
            "spot_kurt_real": target["spot_excess_kurtosis"],
            "spot_kurt_syn": syn["spot_excess_kurtosis"],
            "spot_sq_acf1_real": target["spot_squared_acf1"],
            "spot_sq_acf1_syn": syn["spot_squared_acf1"],
            "corr_real": target["spot_futures_corr"],
            "corr_syn": syn["spot_futures_corr"],
            "basis_std_ratio": syn["basis_std"] / target["basis_std"],
            "basis_mean_real": target["basis_mean"],
            "basis_mean_syn": syn["basis_mean"],
            "basis_acf1_real": target["basis_acf1"],
            "basis_acf1_syn": syn["basis_acf1"],
        }
        rows.append(row)
        detailed[str(window)] = {
            "train_calibrated_heston": h,
            "v0_from_train": v0,
            "synthetic_metrics_on_candidate": syn,
            "validation_target": target,
            "score_components": components,
            "score": score,
        }

    result = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    best_window = int(result.iloc[0]["window"])

    # Core gates are deliberately strict enough to detect material misspecification.
    best = result.iloc[0]
    gates = {
        "spot_vol_ratio_0.90_1.10": bool(0.90 <= best["spot_vol_ratio"] <= 1.10),
        "futures_vol_ratio_0.90_1.10": bool(0.90 <= best["futures_vol_ratio"] <= 1.10),
        "corr_abs_diff_le_0.05": bool(abs(best["corr_syn"] - best["corr_real"]) <= 0.05),
        "basis_std_ratio_0.80_1.20": bool(0.80 <= best["basis_std_ratio"] <= 1.20),
        "spot_sq_acf1_abs_diff_le_0.10": bool(
            abs(best["spot_sq_acf1_syn"] - best["spot_sq_acf1_real"]) <= 0.10
        ),
    }

    table_path = tables / "validation_simulator_candidate_comparison.csv"
    result.to_csv(table_path, index=False)

    report = {
        "leakage_policy": {
            "calibration": "train_only",
            "selection": "validation_only",
            "test_used": False,
        },
        "n_validation_rows": int(len(validation)),
        "n_monte_carlo_paths_per_candidate": N_PATHS,
        "common_random_seed": SEED,
        "ranking": result.to_dict(orient="records"),
        "best_untuned_window": best_window,
        "best_core_gates": gates,
        "all_best_core_gates_pass": bool(all(gates.values())),
        "details": detailed,
        "next_step": (
            "freeze candidate if gates and tail diagnostics are satisfactory; otherwise perform a small "
            "validation-only tuning grid without opening test data"
        ),
    }
    report_path = calibration / "validation_simulator_selection.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 126)
    print("VALIDATION-ONLY SIMULATOR CANDIDATE SELECTION COMPLETE")
    print("=" * 126)
    print(
        result[
            [
                "window",
                "score",
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
    print(f"spot std          : {target['spot_std']:.6f}")
    print(f"futures std       : {target['futures_std']:.6f}")
    print(f"spot skew         : {target['spot_skew']:.6f}")
    print(f"spot excess kurt  : {target['spot_excess_kurtosis']:.6f}")
    print(f"spot squared ACF1 : {target['spot_squared_acf1']:.6f}")
    print(f"corr(S,F)         : {target['spot_futures_corr']:.6f}")
    print(f"basis std         : {target['basis_std']:.6f}")
    print()
    print(f"Best untuned window : {best_window}d")
    for name, passed in gates.items():
        print(f"{'PASS' if passed else 'FAIL':4s} | {name}")
    print(f"ALL CORE GATES PASS : {all(gates.values())}")
    print(f"Table               : {table_path}")
    print(f"Report              : {report_path}")
    print()
    print("TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
