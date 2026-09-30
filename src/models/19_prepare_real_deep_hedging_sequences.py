"""Prepare REAL train/validation 60-day episodes for Deep Hedging transfer/fine-tuning.

Purpose
-------
Map real VN30/VN30F1M data into the same 7-feature causal state space used by
step 16 for synthetic pretraining, while preserving the synthetic TRAIN scaler
so the pretrained LSTM sees the same input coordinate system.

Leakage policy
--------------
- Uses only real TRAIN and real VALIDATION splits from step 09.
- Real TEST (2025-2026) is never opened.
- 60-day episodes are built entirely within their own split.
- Every state at t uses information available at t only.

Real variance proxy
-------------------
Synthetic pretraining observes sqrt(v_t), where v_t is annualized Heston
instantaneous variance. On real data there is no latent v_t, so this script uses
same-day Garman-Klass range variance as a causal observable proxy:
    v_proxy,t = 252 * GK_t
    feature   = sqrt(max(v_proxy,t, 0))
This is known by the close of day t and is dimensionally comparable to the
synthetic annualized volatility state.

Episode construction
--------------------
For each contiguous 61-row block we create 60 hedge decisions. With stride=1,
real TRAIN produces overlapping fine-tuning episodes; real VALIDATION produces
overlapping selection/evaluation episodes. Overlap is documented and TEST
remains sealed.

Outputs
-------
data/processed/deep_hedging_real/train_sequences.npz
data/processed/deep_hedging_real/validation_sequences.npz
data/processed/deep_hedging_real/sequence_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
EPISODE_STEPS = 60
STRIDE = 1
EXPECTED_FEATURES = [
    "lag_spot_log_return",
    "lag_fut_log_return",
    "basis_t",
    "sqrt_variance_t",
    "log_spot_relative_t",
    "log_futures_relative_t",
    "time_remaining",
]


def garman_klass_annualized_variance(df: pd.DataFrame) -> np.ndarray:
    log_hl = np.log(df["spot_high"].to_numpy(dtype=float) / df["spot_low"].to_numpy(dtype=float))
    log_co = np.log(df["spot_close"].to_numpy(dtype=float) / df["spot_open"].to_numpy(dtype=float))
    gk_daily = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return np.maximum(gk_daily, 0.0) * TRADING_DAYS


def load_scaler(metadata_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run src/models/16_prepare_deep_hedging_sequences.py first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    names = list(meta["feature_names"])
    if names != EXPECTED_FEATURES:
        raise ValueError(
            "Synthetic feature definition differs from the expected 7-feature transfer schema."
        )

    mean = np.array([meta["scaler_mean"][name] for name in names], dtype=np.float64)
    std = np.array([meta["scaler_std"][name] for name in names], dtype=np.float64)
    if np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid synthetic scaler statistics.")
    return names, mean, std


def validate_real_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {
        "time",
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_close",
        "fut_close",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="raise")
    out = out.sort_values("time").reset_index(drop=True)

    if out["time"].duplicated().any():
        raise ValueError(f"{name}: duplicate dates detected.")
    for col in required.difference({"time"}):
        values = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: {col} contains NaN/inf.")
        if "close" in col or col.startswith("spot_"):
            if (values <= 0).any():
                raise ValueError(f"{name}: {col} contains non-positive values.")
        out[col] = values

    if len(out) < EPISODE_STEPS + 1:
        raise ValueError(f"{name}: need at least {EPISODE_STEPS + 1} rows.")
    return out


def build_episodes(
    df: pd.DataFrame,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    spot = df["spot_close"].to_numpy(dtype=np.float64)
    fut = df["fut_close"].to_numpy(dtype=np.float64)
    basis = (fut - spot) / spot
    sqrt_var = np.sqrt(garman_klass_annualized_variance(df))

    n_rows = len(df)
    starts = np.arange(0, n_rows - EPISODE_STEPS, STRIDE, dtype=int)
    n_episodes = len(starts)

    features_raw = np.empty((n_episodes, EPISODE_STEPS, len(EXPECTED_FEATURES)), dtype=np.float32)
    spot_forward = np.empty((n_episodes, EPISODE_STEPS), dtype=np.float32)
    fut_forward = np.empty((n_episodes, EPISODE_STEPS), dtype=np.float32)
    start_dates: list[str] = []
    end_dates: list[str] = []

    time_remaining = (
        (EPISODE_STEPS - np.arange(EPISODE_STEPS, dtype=np.float64)) / EPISODE_STEPS
    )

    for j, start in enumerate(starts):
        end = start + EPISODE_STEPS
        s = spot[start : end + 1]
        f = fut[start : end + 1]
        b = basis[start : end]
        sv = sqrt_var[start : end]

        s_forward = np.diff(np.log(s))
        f_forward = np.diff(np.log(f))

        lag_s = np.zeros(EPISODE_STEPS, dtype=np.float64)
        lag_f = np.zeros(EPISODE_STEPS, dtype=np.float64)
        if EPISODE_STEPS > 1:
            lag_s[1:] = s_forward[:-1]
            lag_f[1:] = f_forward[:-1]

        log_s_relative = np.log(s[:-1] / s[0])
        log_f_relative = np.log(f[:-1] / f[0])

        raw = np.column_stack(
            [
                lag_s,
                lag_f,
                b,
                sv,
                log_s_relative,
                log_f_relative,
                time_remaining,
            ]
        )
        features_raw[j] = raw.astype(np.float32)
        spot_forward[j] = s_forward.astype(np.float32)
        fut_forward[j] = f_forward.astype(np.float32)
        start_dates.append(df["time"].iloc[start].strftime("%Y-%m-%d"))
        end_dates.append(df["time"].iloc[end].strftime("%Y-%m-%d"))

    scaled = (
        (features_raw.astype(np.float64) - scaler_mean.reshape(1, 1, -1))
        / scaler_std.reshape(1, 1, -1)
    ).astype(np.float32)

    if not np.isfinite(scaled).all():
        raise ValueError("Scaled real feature tensor contains NaN/inf.")

    flat = scaled.reshape(-1, scaled.shape[-1]).astype(np.float64)
    feature_mean = np.mean(flat, axis=0)
    feature_std = np.std(flat, axis=0, ddof=0)
    feature_abs_z_gt5 = np.mean(np.abs(flat) > 5.0, axis=0)

    summary = {
        "rows": int(n_rows),
        "episodes": int(n_episodes),
        "steps_per_episode": EPISODE_STEPS,
        "stride": STRIDE,
        "date_min": df["time"].min().strftime("%Y-%m-%d"),
        "date_max": df["time"].max().strftime("%Y-%m-%d"),
        "first_episode_start": start_dates[0],
        "first_episode_end": end_dates[0],
        "last_episode_start": start_dates[-1],
        "last_episode_end": end_dates[-1],
        "spot_forward_std": float(np.std(spot_forward, ddof=1)),
        "fut_forward_std": float(np.std(fut_forward, ddof=1)),
        "scaled_feature_mean": {
            EXPECTED_FEATURES[i]: float(feature_mean[i]) for i in range(len(EXPECTED_FEATURES))
        },
        "scaled_feature_std": {
            EXPECTED_FEATURES[i]: float(feature_std[i]) for i in range(len(EXPECTED_FEATURES))
        },
        "fraction_abs_scaled_z_gt5": {
            EXPECTED_FEATURES[i]: float(feature_abs_z_gt5[i]) for i in range(len(EXPECTED_FEATURES))
        },
        "max_abs_scaled_value": float(np.max(np.abs(flat))),
    }

    arrays = {
        "features": scaled,
        "spot_forward_log_return": spot_forward,
        "fut_forward_log_return": fut_forward,
        "episode_start_index": starts.astype(np.int32),
    }
    return arrays, summary


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    splits_dir = root / "data" / "processed" / "splits"
    synthetic_meta = root / "data" / "processed" / "deep_hedging" / "sequence_metadata.json"
    out_dir = root / "data" / "processed" / "deep_hedging_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names, scaler_mean, scaler_std = load_scaler(synthetic_meta)

    train_path = splits_dir / "train.csv"
    validation_path = splits_dir / "validation.csv"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run step 09 first.")

    real_train = validate_real_split(pd.read_csv(train_path), "real train")
    real_validation = validate_real_split(pd.read_csv(validation_path), "real validation")

    train_arrays, train_summary = build_episodes(real_train, scaler_mean, scaler_std)
    validation_arrays, validation_summary = build_episodes(
        real_validation, scaler_mean, scaler_std
    )

    train_out = out_dir / "train_sequences.npz"
    validation_out = out_dir / "validation_sequences.npz"
    save_npz(train_out, train_arrays)
    save_npz(validation_out, validation_arrays)

    metadata = {
        "dataset_role": "real_train_validation_sequences_for_transfer_and_finetuning",
        "feature_names": feature_names,
        "episode_steps": EPISODE_STEPS,
        "stride": STRIDE,
        "overlapping_episodes": True,
        "scaler_source": "synthetic TRAIN scaler from step 16",
        "real_variance_proxy": (
            "sqrt(252 * same-day Garman-Klass variance), clipped at zero"
        ),
        "causal_timing": (
            "state at t uses only data observable by close t; action h_t applies to t->t+1"
        ),
        "first_step_lag_returns": (
            "set to zero in every episode to match synthetic path initialization"
        ),
        "train": train_summary,
        "validation": validation_summary,
        "real_test_used": False,
        "important_note": (
            "These are overlapping 60-day episodes intended for transfer learning. "
            "Do not interpret episode count as independent market samples."
        ),
    }

    metadata_path = out_dir / "sequence_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 110)
    print("REAL DEEP HEDGING SEQUENCE PREPARATION COMPLETE — TRAIN + VALIDATION ONLY")
    print("=" * 110)
    print(f"Episode length      : {EPISODE_STEPS} trading days")
    print(f"Episode stride      : {STRIDE} day")
    print(f"Scaler              : synthetic TRAIN scaler from step 16")
    print(f"Variance proxy      : sqrt(252 * same-day Garman-Klass variance)")
    print()
    print(
        f"real train      | rows={train_summary['rows']:,} | episodes={train_summary['episodes']:,} | "
        f"{train_summary['date_min']} -> {train_summary['date_max']} | "
        f"spot std={train_summary['spot_forward_std']:.5f} | fut std={train_summary['fut_forward_std']:.5f}"
    )
    print(
        f"real validation | rows={validation_summary['rows']:,} | episodes={validation_summary['episodes']:,} | "
        f"{validation_summary['date_min']} -> {validation_summary['date_max']} | "
        f"spot std={validation_summary['spot_forward_std']:.5f} | fut std={validation_summary['fut_forward_std']:.5f}"
    )
    print()
    print("Scaled-domain shift diagnostics:")
    for split_name, summary in (
        ("train", train_summary),
        ("validation", validation_summary),
    ):
        print(
            f"{split_name:10s} | max|z|={summary['max_abs_scaled_value']:.3f} | "
            f"max frac(|z|>5)={max(summary['fraction_abs_scaled_z_gt5'].values()):.6f}"
        )
    print()
    print(f"Train output      : {train_out}")
    print(f"Validation output : {validation_out}")
    print(f"Metadata          : {metadata_path}")
    print("REAL TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
