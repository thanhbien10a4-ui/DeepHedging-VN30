"""Generate joint synthetic VN30 spot + VN30F1M futures paths.

Spot dynamics
-------------
Heston stochastic volatility with the selected baseline parameters from the
10-day smoothed Garman-Klass proxy comparison.

Basis dynamics
--------------
Filtered AR(1) relative-basis model estimated in 06_estimate_basis_process.py.
Synthetic futures are reconstructed as:

    F_t = S_t * (1 + b_t)

Numerical scheme
----------------
Full-truncation Euler is used for the Heston variance process to keep variance
non-negative and remain robust when the Feller condition is only marginally
satisfied.

Outputs
-------
data/synthetic/train.npz
data/synthetic/validation.npz
data/synthetic/test.npz
data/synthetic/synthetic_metadata.json
results/figures/synthetic_sample_paths.png

The generated NPZ files are intentionally gitignored and can be reproduced
from committed code and parameter files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS
BASELINE_WINDOW = 10
SEED = 20260930

# Moderate defaults for a first reproducible training set.
# Can be increased later after validating the simulator statistics.
N_STEPS = 60
N_TRAIN = 20_000
N_VALIDATION = 4_000
N_TEST = 4_000


def load_selected_heston_params(processed_dir: Path) -> dict:
    path = processed_dir / "heston_proxy_comparison.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 05b_compare_heston_variance_proxies.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    matches = [row for row in payload["results"] if int(row["window"]) == BASELINE_WINDOW]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Heston result for window={BASELINE_WINDOW}.")
    return matches[0]


def load_basis_params(processed_dir: Path) -> dict:
    path = processed_dir / "basis_process_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 06_estimate_basis_process.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"]["filtered"]


def load_initial_state(processed_dir: Path, raw_dir: Path) -> tuple[float, float, float]:
    # Latest real spot close.
    import pandas as pd

    spot_path = raw_dir / "vn30_1d.csv"
    fut_path = raw_dir / "vn30f1m_1d.csv"
    if not spot_path.exists() or not fut_path.exists():
        raise FileNotFoundError("Missing raw VN30/VN30F1M CSV files.")

    spot = pd.read_csv(spot_path)
    fut = pd.read_csv(fut_path)
    s0 = float(spot["close"].iloc[-1])
    f0 = float(fut["close"].iloc[-1])
    b0 = float((f0 - s0) / s0)

    # Latest 10-day smoothed GK variance proxy used consistently with baseline.
    proxy_path = processed_dir / "heston_proxy_comparison.json"
    # v0 is not stored in the comparison JSON, so rebuild from the committed rule.
    aligned_path = processed_dir / "aligned_1d.csv"
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing {aligned_path}.")

    aligned = pd.read_csv(aligned_path)
    log_hl = np.log(aligned["spot_high"] / aligned["spot_low"])
    log_co = np.log(aligned["spot_close"] / aligned["spot_open"])
    gk_daily = (0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)).clip(lower=0.0)
    v10 = (gk_daily.rolling(BASELINE_WINDOW).mean() * TRADING_DAYS).iloc[-1]
    v0 = float(max(v10, 1e-8))
    return s0, v0, b0


def simulate_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    s0: float,
    v0: float,
    b0: float,
    heston: dict,
    basis: dict,
) -> dict[str, np.ndarray]:
    mu = float(heston["mu"])
    kappa = float(heston["kappa"])
    theta = float(heston["theta"])
    xi = float(heston["xi"])
    rho = float(heston["rho"])

    c = float(basis["c"])
    phi = float(basis["phi"])
    sigma_b = float(basis["residual_std_daily"])

    spot = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    variance = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    basis_arr = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    futures = np.empty((n_paths, n_steps + 1), dtype=np.float32)

    spot[:, 0] = s0
    variance[:, 0] = v0
    basis_arr[:, 0] = b0
    futures[:, 0] = s0 * (1.0 + b0)

    sqrt_dt = np.sqrt(DT)
    sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 0.0))

    for t in range(n_steps):
        z_s = rng.standard_normal(n_paths)
        z_ind = rng.standard_normal(n_paths)
        z_v = rho * z_s + sqrt_one_minus_rho2 * z_ind
        z_b = rng.standard_normal(n_paths)

        s_t = spot[:, t].astype(np.float64)
        v_t = variance[:, t].astype(np.float64)
        v_pos = np.maximum(v_t, 0.0)

        # Full-truncation Euler variance step.
        v_next = (
            v_t
            + kappa * (theta - v_pos) * DT
            + xi * np.sqrt(v_pos) * sqrt_dt * z_v
        )
        v_next = np.maximum(v_next, 0.0)

        # Log-Euler spot step using current non-negative variance.
        log_increment = (mu - 0.5 * v_pos) * DT + np.sqrt(v_pos) * sqrt_dt * z_s
        s_next = s_t * np.exp(log_increment)

        b_t = basis_arr[:, t].astype(np.float64)
        b_next = c + phi * b_t + sigma_b * z_b

        # Prevent pathological F<=0 states from extremely rare Gaussian basis draws.
        b_next = np.maximum(b_next, -0.95)
        f_next = s_next * (1.0 + b_next)

        spot[:, t + 1] = s_next.astype(np.float32)
        variance[:, t + 1] = v_next.astype(np.float32)
        basis_arr[:, t + 1] = b_next.astype(np.float32)
        futures[:, t + 1] = f_next.astype(np.float32)

    spot_log_return = np.diff(np.log(spot.astype(np.float64)), axis=1).astype(np.float32)
    fut_log_return = np.diff(np.log(futures.astype(np.float64)), axis=1).astype(np.float32)

    return {
        "spot": spot,
        "variance": variance,
        "basis": basis_arr,
        "futures": futures,
        "spot_log_return": spot_log_return,
        "fut_log_return": fut_log_return,
    }


def save_split(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def summarize_split(arrays: dict[str, np.ndarray]) -> dict:
    spot_ret = arrays["spot_log_return"].astype(np.float64).ravel()
    fut_ret = arrays["fut_log_return"].astype(np.float64).ravel()
    basis_vals = arrays["basis"].astype(np.float64).ravel()
    var_vals = arrays["variance"].astype(np.float64).ravel()

    return {
        "n_paths": int(arrays["spot"].shape[0]),
        "n_steps": int(arrays["spot"].shape[1] - 1),
        "spot_return_mean_daily": float(np.mean(spot_ret)),
        "spot_return_std_daily": float(np.std(spot_ret, ddof=1)),
        "futures_return_mean_daily": float(np.mean(fut_ret)),
        "futures_return_std_daily": float(np.std(fut_ret, ddof=1)),
        "spot_futures_return_corr": float(np.corrcoef(spot_ret, fut_ret)[0, 1]),
        "basis_mean": float(np.mean(basis_vals)),
        "basis_std": float(np.std(basis_vals, ddof=1)),
        "variance_mean": float(np.mean(var_vals)),
        "variance_zero_fraction": float(np.mean(var_vals <= 0.0)),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    raw_dir = repo_root / "data" / "raw"
    synthetic_dir = repo_root / "data" / "synthetic"
    figures_dir = repo_root / "results" / "figures"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    heston = load_selected_heston_params(processed_dir)
    basis = load_basis_params(processed_dir)
    s0, v0, b0 = load_initial_state(processed_dir, raw_dir)

    rng = np.random.default_rng(SEED)

    splits = {
        "train": N_TRAIN,
        "validation": N_VALIDATION,
        "test": N_TEST,
    }

    metadata = {
        "seed": SEED,
        "dt": DT,
        "trading_days_per_year": int(TRADING_DAYS),
        "n_steps": N_STEPS,
        "initial_state": {"spot": s0, "variance": v0, "basis": b0},
        "heston_source": f"10-day GK rolling proxy (window={BASELINE_WINDOW})",
        "heston_parameters": {
            "mu": float(heston["mu"]),
            "kappa": float(heston["kappa"]),
            "theta": float(heston["theta"]),
            "xi": float(heston["xi"]),
            "rho": float(heston["rho"]),
        },
        "basis_parameters": {
            "c": float(basis["c"]),
            "phi": float(basis["phi"]),
            "residual_std_daily": float(basis["residual_std_daily"]),
            "long_run_mean_basis_pct": float(basis["long_run_mean_basis_pct"]),
        },
        "numerical_scheme": "full_truncation_euler_for_variance_log_euler_for_spot",
        "basis_scheme": "gaussian_AR1_baseline",
        "splits": {},
    }

    sample_train = None
    for name, n_paths in splits.items():
        arrays = simulate_paths(
            rng=rng,
            n_paths=n_paths,
            n_steps=N_STEPS,
            s0=s0,
            v0=v0,
            b0=b0,
            heston=heston,
            basis=basis,
        )
        save_split(synthetic_dir / f"{name}.npz", arrays)
        metadata["splits"][name] = summarize_split(arrays)
        if name == "train":
            sample_train = arrays

    metadata_path = synthetic_dir / "synthetic_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Plot only a few paths to keep the figure readable.
    assert sample_train is not None
    fig = plt.figure(figsize=(12, 5))
    n_plot = 12
    for i in range(n_plot):
        plt.plot(sample_train["spot"][i], linewidth=0.8)
    plt.title("Synthetic VN30 Spot Paths — Heston Baseline")
    plt.xlabel("Trading day")
    plt.ylabel("VN30 level")
    plt.tight_layout()
    figure_path = figures_dir / "synthetic_sample_paths.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 84)
    print("JOINT SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 84)
    print(f"Seed                : {SEED}")
    print(f"Horizon             : {N_STEPS} trading days")
    print(f"Train paths         : {N_TRAIN:,}")
    print(f"Validation paths    : {N_VALIDATION:,}")
    print(f"Test paths          : {N_TEST:,}")
    print(f"S0                  : {s0:.6f}")
    print(f"v0                  : {v0:.6f}")
    print(f"basis0              : {b0:.6f}")
    print(f"Heston kappa        : {float(heston['kappa']):.6f}")
    print(f"Heston theta        : {float(heston['theta']):.6f}")
    print(f"Heston xi           : {float(heston['xi']):.6f}")
    print(f"Heston rho          : {float(heston['rho']):.6f}")
    print(f"Basis phi           : {float(basis['phi']):.6f}")
    print(f"Basis residual std  : {float(basis['residual_std_daily']):.6f}")
    print()
    for name in splits:
        stats = metadata["splits"][name]
        print(
            f"{name:10s} | corr(S,F)={stats['spot_futures_return_corr']:.4f} "
            f"| spot vol(daily)={stats['spot_return_std_daily']:.5f} "
            f"| basis mean={stats['basis_mean']:.5f}"
        )
    print()
    print(f"Synthetic data dir  : {synthetic_dir}")
    print(f"Metadata            : {metadata_path}")
    print(f"Figure              : {figure_path}")
    print()
    print("Next step: validate synthetic moments against the real VN30/VN30F1M sample before training.")


if __name__ == "__main__":
    main()
