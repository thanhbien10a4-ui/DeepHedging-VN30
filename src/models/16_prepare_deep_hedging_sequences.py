"""Prepare causal sequence tensors for Deep Hedging LSTM/GRU training.

This step is framework-agnostic. It converts the FINAL frozen synthetic paths
from step 14 into decision-time state sequences and forward hedge returns.

Causal timing
-------------
At decision time t the policy may observe only information available at t:
- previous spot return r^S_{t-1,t} (0 at t=0),
- previous futures return r^F_{t-1,t} (0 at t=0),
- current relative basis b_t,
- current instantaneous volatility proxy sqrt(v_t),
- log spot level relative to path start,
- log futures level relative to path start,
- normalized time remaining.

The action h_t is then applied to the NEXT interval t -> t+1. The training
loss can therefore use:
    hedge_return_t = spot_forward_return_t - h_t * fut_forward_return_t

No future return is present in the state tensor.

Scaling
-------
Feature mean/std are fitted on SYNTHETIC TRAIN only, then frozen and applied to
synthetic validation/test. The real 2025-2026 OOS market test is not opened.

Outputs
-------
data/processed/deep_hedging/train_sequences.npz
data/processed/deep_hedging/validation_sequences.npz
data/processed/deep_hedging/test_sequences.npz
data/processed/deep_hedging/sequence_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "lag_spot_log_return",
    "lag_fut_log_return",
    "basis_t",
    "sqrt_variance_t",
    "log_spot_relative_t",
    "log_futures_relative_t",
    "time_remaining",
]


def load_frozen_split(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/simulator/14_generate_frozen_synthetic.py first."
        )
    with np.load(path) as z:
        required = {
            "spot",
            "variance",
            "basis",
            "futures",
            "spot_log_return",
            "fut_log_return",
        }
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        return {name: z[name].astype(np.float32) for name in required}


def build_raw_sequences(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    spot = arrays["spot"]
    fut = arrays["futures"]
    variance = arrays["variance"]
    basis = arrays["basis"]
    spot_forward = arrays["spot_log_return"]
    fut_forward = arrays["fut_log_return"]

    n_paths, n_plus_one = spot.shape
    n_steps = n_plus_one - 1

    expected_shape = (n_paths, n_steps + 1)
    for name in ("futures", "variance", "basis"):
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name}: expected shape {expected_shape}, got {arrays[name].shape}"
            )
    if spot_forward.shape != (n_paths, n_steps):
        raise ValueError("spot_log_return shape is inconsistent with spot paths.")
    if fut_forward.shape != (n_paths, n_steps):
        raise ValueError("fut_log_return shape is inconsistent with futures paths.")

    lag_spot = np.zeros((n_paths, n_steps), dtype=np.float32)
    lag_fut = np.zeros((n_paths, n_steps), dtype=np.float32)
    if n_steps > 1:
        lag_spot[:, 1:] = spot_forward[:, :-1]
        lag_fut[:, 1:] = fut_forward[:, :-1]

    basis_t = basis[:, :-1]
    sqrt_variance_t = np.sqrt(np.maximum(variance[:, :-1], 0.0)).astype(np.float32)

    spot0 = spot[:, [0]].astype(np.float64)
    fut0 = fut[:, [0]].astype(np.float64)
    log_spot_relative = np.log(spot[:, :-1].astype(np.float64) / spot0).astype(np.float32)
    log_fut_relative = np.log(fut[:, :-1].astype(np.float64) / fut0).astype(np.float32)

    # t=0 -> 1.0 remaining, final decision t=n_steps-1 -> 1/n_steps remaining.
    time_remaining_1d = (
        (n_steps - np.arange(n_steps, dtype=np.float32)) / float(n_steps)
    )
    time_remaining = np.broadcast_to(time_remaining_1d, (n_paths, n_steps)).copy()

    features = np.stack(
        [
            lag_spot,
            lag_fut,
            basis_t,
            sqrt_variance_t,
            log_spot_relative,
            log_fut_relative,
            time_remaining,
        ],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Feature tensor contains NaN/inf values.")
    if not np.isfinite(spot_forward).all() or not np.isfinite(fut_forward).all():
        raise ValueError("Forward-return tensors contain NaN/inf values.")

    return {
        "features_raw": features,
        "spot_forward_log_return": spot_forward.astype(np.float32),
        "fut_forward_log_return": fut_forward.astype(np.float32),
    }


def fit_train_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0, ddof=0)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError("Scaler statistics contain NaN/inf values.")
    if np.any(std <= 1e-12):
        bad = [FEATURE_NAMES[i] for i in np.where(std <= 1e-12)[0]]
        raise ValueError(f"Near-zero feature std for {bad}")
    return mean.astype(np.float64), std.astype(np.float64)


def transform(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (features.astype(np.float64) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return scaled.astype(np.float32)


def save_split(
    path: Path,
    raw: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    features_scaled = transform(raw["features_raw"], mean, std)
    np.savez_compressed(
        path,
        features=features_scaled,
        spot_forward_log_return=raw["spot_forward_log_return"],
        fut_forward_log_return=raw["fut_forward_log_return"],
    )

    return {
        "n_paths": int(features_scaled.shape[0]),
        "n_steps": int(features_scaled.shape[1]),
        "n_features": int(features_scaled.shape[2]),
        "scaled_feature_mean_abs_max": float(
            np.max(np.abs(np.mean(features_scaled.astype(np.float64), axis=(0, 1))))
        ),
        "scaled_feature_std_min": float(
            np.min(np.std(features_scaled.astype(np.float64), axis=(0, 1), ddof=0))
        ),
        "scaled_feature_std_max": float(
            np.max(np.std(features_scaled.astype(np.float64), axis=(0, 1), ddof=0))
        ),
        "spot_forward_std": float(np.std(raw["spot_forward_log_return"], ddof=1)),
        "fut_forward_std": float(np.std(raw["fut_forward_log_return"], ddof=1)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    synthetic_dir = root / "data" / "synthetic" / "frozen"
    output_dir = root / "data" / "processed" / "deep_hedging"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "validation", "test")
    raw_sequences: dict[str, dict[str, np.ndarray]] = {}

    for split in split_names:
        arrays = load_frozen_split(synthetic_dir / f"{split}.npz")
        raw_sequences[split] = build_raw_sequences(arrays)

    train_features = raw_sequences["train"]["features_raw"]
    scaler_mean, scaler_std = fit_train_scaler(train_features)

    summaries: dict[str, dict] = {}
    for split in split_names:
        summaries[split] = save_split(
            output_dir / f"{split}_sequences.npz",
            raw_sequences[split],
            scaler_mean,
            scaler_std,
        )

    metadata = {
        "dataset_role": "causal_sequences_for_deep_hedging",
        "source": "data/synthetic/frozen/*.npz from step 14",
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "causal_timing": (
            "features at t use information available at t only; hedge action h_t is applied "
            "to forward interval t->t+1"
        ),
        "hedge_return_definition": (
            "spot_forward_log_return - h_t * fut_forward_log_return"
        ),
        "scaler_policy": "mean/std fitted on synthetic TRAIN only",
        "scaler_mean": {
            FEATURE_NAMES[i]: float(scaler_mean[i]) for i in range(len(FEATURE_NAMES))
        },
        "scaler_std": {
            FEATURE_NAMES[i]: float(scaler_std[i]) for i in range(len(FEATURE_NAMES))
        },
        "splits": summaries,
        "real_market_test_used": False,
        "note": (
            "Previous hedge h_{t-1} is intentionally not precomputed as an exogenous feature. "
            "A recurrent policy may feed its own previous action internally when transaction-cost "
            "penalties are introduced."
        ),
    }

    metadata_path = output_dir / "sequence_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 100)
    print("DEEP HEDGING CAUSAL SEQUENCE PREPARATION COMPLETE")
    print("=" * 100)
    print(f"Features ({len(FEATURE_NAMES)}) : {', '.join(FEATURE_NAMES)}")
    print()
    for split in split_names:
        s = summaries[split]
        print(
            f"{split:10s} | paths={s['n_paths']:,} | steps={s['n_steps']} | "
            f"features={s['n_features']} | spot std={s['spot_forward_std']:.5f} | "
            f"fut std={s['fut_forward_std']:.5f}"
        )
    print()
    train_summary = summaries["train"]
    print(f"Train scaled mean abs max : {train_summary['scaled_feature_mean_abs_max']:.6e}")
    print(
        "Train scaled std range    : "
        f"{train_summary['scaled_feature_std_min']:.6f} -> "
        f"{train_summary['scaled_feature_std_max']:.6f}"
    )
    print(f"Output dir                : {output_dir}")
    print(f"Metadata                  : {metadata_path}")
    print("REAL MARKET TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
