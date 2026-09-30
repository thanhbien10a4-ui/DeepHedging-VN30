"""Diagnose synthetic->real feature-domain shift before fine-tuning.

This step does NOT modify, clip, winsorize or rescale any data. It only explains
which real features produce large synthetic-scaled z-scores and whether those
extremes come from many unique market dates or repeated overlapping episodes.

Inputs
------
data/processed/deep_hedging/sequence_metadata.json
data/processed/deep_hedging_real/train_sequences.npz
data/processed/deep_hedging_real/validation_sequences.npz
data/processed/splits/train.csv
data/processed/splits/validation.csv

Real TEST is never opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sequences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["features"].astype(np.float64), z["episode_start_index"].astype(int)


def feature_diagnostics(
    split_name: str,
    features: np.ndarray,
    starts: np.ndarray,
    dates: pd.Series,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_ep, n_steps, n_feat = features.shape
    if n_feat != len(feature_names):
        raise ValueError("Feature count mismatch.")

    rows: list[dict] = []
    date_rows: list[dict] = []

    # Track date-level maximum absolute z across all repeated episode appearances.
    per_date_max = np.full((len(dates), n_feat), np.nan, dtype=float)

    for j, name in enumerate(feature_names):
        x = features[:, :, j]
        flat = x.reshape(-1)
        abs_flat = np.abs(flat)

        max_flat_idx = int(np.argmax(abs_flat))
        ep_idx, step_idx = np.unravel_index(max_flat_idx, x.shape)
        date_idx = int(starts[ep_idx] + step_idx)
        z_max_signed = float(x[ep_idx, step_idx])
        raw_max = float(z_max_signed * scaler_std[j] + scaler_mean[j])

        # Map every episode-time occurrence back to its source market date.
        occurrence_date_idx = starts[:, None] + np.arange(n_steps, dtype=int)[None, :]
        for d in np.unique(occurrence_date_idx):
            vals = x[occurrence_date_idx == d]
            per_date_max[d, j] = float(np.max(np.abs(vals)))

        gt5_mask = abs_flat > 5.0
        gt5_date_indices = np.unique(occurrence_date_idx.reshape(-1)[gt5_mask])

        rows.append(
            {
                "split": split_name,
                "feature": name,
                "mean_z_episode_weighted": float(np.mean(flat)),
                "std_z_episode_weighted": float(np.std(flat, ddof=0)),
                "q01_z": float(np.quantile(flat, 0.01)),
                "q50_z": float(np.quantile(flat, 0.50)),
                "q99_z": float(np.quantile(flat, 0.99)),
                "max_abs_z": float(np.max(abs_flat)),
                "fraction_abs_z_gt3": float(np.mean(abs_flat > 3.0)),
                "fraction_abs_z_gt5": float(np.mean(gt5_mask)),
                "fraction_abs_z_gt10": float(np.mean(abs_flat > 10.0)),
                "unique_dates_abs_z_gt5": int(len(gt5_date_indices)),
                "fraction_unique_dates_abs_z_gt5": float(len(gt5_date_indices) / len(dates)),
                "max_z_date": dates.iloc[date_idx].strftime("%Y-%m-%d"),
                "max_z_signed": z_max_signed,
                "raw_value_at_max_z": raw_max,
                "scaler_mean": float(scaler_mean[j]),
                "scaler_std": float(scaler_std[j]),
            }
        )

    # Aggregate the strongest shifted features at each unique real date.
    for d in range(len(dates)):
        vals = per_date_max[d]
        if np.all(np.isnan(vals)):
            continue
        j = int(np.nanargmax(vals))
        date_rows.append(
            {
                "split": split_name,
                "date": dates.iloc[d].strftime("%Y-%m-%d"),
                "max_abs_z_any_feature": float(vals[j]),
                "feature_at_max": feature_names[j],
                "n_features_abs_z_gt5": int(np.nansum(vals > 5.0)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(date_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    synthetic_meta_path = root / "data" / "processed" / "deep_hedging" / "sequence_metadata.json"
    real_dir = root / "data" / "processed" / "deep_hedging_real"
    splits_dir = root / "data" / "processed" / "splits"
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    meta = load_json(synthetic_meta_path)
    feature_names = list(meta["feature_names"])
    scaler_mean = np.array([meta["scaler_mean"][f] for f in feature_names], dtype=float)
    scaler_std = np.array([meta["scaler_std"][f] for f in feature_names], dtype=float)

    feature_tables: list[pd.DataFrame] = []
    date_tables: list[pd.DataFrame] = []

    for split_name in ("train", "validation"):
        features, starts = load_sequences(real_dir / f"{split_name}_sequences.npz")
        raw = pd.read_csv(splits_dir / f"{split_name}.csv")
        raw["time"] = pd.to_datetime(raw["time"], errors="raise")
        raw = raw.sort_values("time").reset_index(drop=True)

        feat_df, date_df = feature_diagnostics(
            split_name,
            features,
            starts,
            raw["time"],
            feature_names,
            scaler_mean,
            scaler_std,
        )
        feature_tables.append(feat_df)
        date_tables.append(date_df)

    feature_df = pd.concat(feature_tables, ignore_index=True)
    date_df = pd.concat(date_tables, ignore_index=True)

    feature_path = tables_dir / "real_domain_shift_by_feature.csv"
    dates_path = tables_dir / "real_domain_shift_top_dates.csv"
    feature_df.to_csv(feature_path, index=False)
    date_df.sort_values(["split", "max_abs_z_any_feature"], ascending=[True, False]).to_csv(
        dates_path, index=False
    )

    report = {
        "purpose": "diagnose_only_no_transform_applied",
        "real_test_used": False,
        "feature_diagnostics": feature_df.to_dict(orient="records"),
        "top_shift_dates": {
            split: date_df[date_df["split"] == split]
            .sort_values("max_abs_z_any_feature", ascending=False)
            .head(20)
            .to_dict(orient="records")
            for split in ("train", "validation")
        },
        "interpretation_rule": (
            "Do not clip/rescale solely because episode-weighted |z| is large. "
            "Check which feature and how many unique dates are responsible first."
        ),
    }
    report_path = metrics_dir / "real_domain_shift_diagnostic.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("=" * 128)
    print("REAL DOMAIN-SHIFT DIAGNOSTIC COMPLETE — NO DATA TRANSFORMATION APPLIED")
    print("=" * 128)

    for split_name in ("train", "validation"):
        print()
        print(f"{split_name.upper()} FEATURE DIAGNOSTICS")
        view = feature_df[feature_df["split"] == split_name][
            [
                "feature",
                "mean_z_episode_weighted",
                "std_z_episode_weighted",
                "max_abs_z",
                "fraction_abs_z_gt5",
                "unique_dates_abs_z_gt5",
                "fraction_unique_dates_abs_z_gt5",
                "max_z_date",
                "max_z_signed",
            ]
        ].sort_values("max_abs_z", ascending=False)
        print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        print()
        print(f"Top 10 {split_name} dates by any-feature |z|:")
        top_dates = (
            date_df[date_df["split"] == split_name]
            .sort_values("max_abs_z_any_feature", ascending=False)
            .head(10)
        )
        print(top_dates.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print()
    print(f"Feature table : {feature_path}")
    print(f"Date table    : {dates_path}")
    print(f"JSON report   : {report_path}")
    print("REAL TEST DATA WAS NOT READ.")


if __name__ == "__main__":
    main()
