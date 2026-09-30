"""Validate, align, and create the first processed VN30/VN30F1M daily dataset.

Inputs (immutable):
    data/raw/vn30_1d.csv
    data/raw/vn30f1m_1d.csv

Outputs (generated, gitignored):
    data/processed/aligned_1d.csv
    data/processed/validation_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing columns {missing_cols}")

    out = df[REQUIRED_COLUMNS].copy()
    out["time"] = pd.to_datetime(out["time"], format="%Y-%m-%d", errors="raise")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")

    summary = {
        "rows": int(len(out)),
        "date_min": out["time"].min().strftime("%Y-%m-%d"),
        "date_max": out["time"].max().strftime("%Y-%m-%d"),
        "missing_total": int(out.isna().sum().sum()),
        "duplicate_dates": int(out["time"].duplicated().sum()),
        "sorted_ascending": bool(out["time"].is_monotonic_increasing),
        "nonpositive_price_rows": int(
            (out[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
        ),
        "negative_volume_rows": int((out["volume"] < 0).sum()),
        "zero_volume_rows": int((out["volume"] == 0).sum()),
        "invalid_high_rows": int(
            (out["high"] < out[["open", "low", "close"]].max(axis=1)).sum()
        ),
        "invalid_low_rows": int(
            (out["low"] > out[["open", "high", "close"]].min(axis=1)).sum()
        ),
    }

    hard_fail_keys = [
        "missing_total",
        "duplicate_dates",
        "nonpositive_price_rows",
        "negative_volume_rows",
        "invalid_high_rows",
        "invalid_low_rows",
    ]
    hard_fail = {k: summary[k] for k in hard_fail_keys if summary[k] != 0}
    if not summary["sorted_ascending"]:
        hard_fail["sorted_ascending"] = False
    if hard_fail:
        raise ValueError(f"{name}: validation failed: {hard_fail}")

    return out, summary


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {
        c: f"{prefix}_{c}"
        for c in ["open", "high", "low", "close", "volume"]
    }
    return df.rename(columns=renamed)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    spot_raw = pd.read_csv(raw_dir / "vn30_1d.csv")
    fut_raw = pd.read_csv(raw_dir / "vn30f1m_1d.csv")

    spot, spot_summary = validate_ohlcv(spot_raw, "VN30 spot")
    fut, fut_summary = validate_ohlcv(fut_raw, "VN30F1M futures")

    spot_dates = set(spot["time"])
    fut_dates = set(fut["time"])

    only_spot = sorted(d.strftime("%Y-%m-%d") for d in spot_dates - fut_dates)
    only_fut = sorted(d.strftime("%Y-%m-%d") for d in fut_dates - spot_dates)

    aligned = (
        prefix_columns(spot, "spot")
        .merge(prefix_columns(fut, "fut"), on="time", how="inner", validate="one_to_one")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # First-stage features. Rolling features are intentionally deferred to the
    # next preprocessing stage so raw alignment can be audited independently.
    aligned["spot_log_return"] = np.log(aligned["spot_close"]).diff()
    aligned["fut_log_return"] = np.log(aligned["fut_close"]).diff()
    aligned["basis_close"] = aligned["fut_close"] - aligned["spot_close"]
    aligned["basis_pct"] = aligned["basis_close"] / aligned["spot_close"]

    summary = {
        "vn30_spot": spot_summary,
        "vn30f1m_futures": fut_summary,
        "date_alignment": {
            "spot_only_dates": only_spot,
            "futures_only_dates": only_fut,
            "common_dates": int(len(aligned)),
            "alignment_method": "inner join on exact trading date",
        },
        "processed": {
            "rows": int(len(aligned)),
            "date_min": aligned["time"].min().strftime("%Y-%m-%d"),
            "date_max": aligned["time"].max().strftime("%Y-%m-%d"),
            "first_row_return_is_nan": True,
        },
    }

    aligned.to_csv(processed_dir / "aligned_1d.csv", index=False, date_format="%Y-%m-%d")
    with open(processed_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("RAW DATA VALIDATION + ALIGNMENT COMPLETE")
    print("=" * 72)
    print(f"VN30 rows       : {len(spot):,}")
    print(f"VN30F1M rows    : {len(fut):,}")
    print(f"Common dates    : {len(aligned):,}")
    print(f"Spot-only dates : {only_spot}")
    print(f"Fut-only dates  : {only_fut}")
    print(f"Saved           : {processed_dir / 'aligned_1d.csv'}")
    print(f"Report          : {processed_dir / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
