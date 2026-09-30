"""Prepare leakage-safe chronological real-data splits for calibration and OOS evaluation.

Split policy
------------
Train / calibration : 2017-08-10 through 2022-12-31
Validation          : 2023-01-01 through 2024-12-31
Test / final OOS    : 2025-01-01 through end of available sample

The aligned dataset is split by date only. No shuffling is allowed.
All simulator estimation/calibration after this step must use TRAIN only.
Validation may be used for model/simulator selection. TEST must remain untouched
until the final out-of-sample evaluation.

Input
-----
data/processed/aligned_1d.csv

Outputs
-------
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv
data/processed/splits/split_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TRAIN_END = pd.Timestamp("2022-12-31")
VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def describe_split(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError(f"{name} split is empty.")
    return {
        "rows": int(len(df)),
        "date_min": df["time"].min().strftime("%Y-%m-%d"),
        "date_max": df["time"].max().strftime("%Y-%m-%d"),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "aligned_1d.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run 01_validate_and_align.py first."
        )

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    if df["time"].duplicated().any():
        raise ValueError("aligned_1d.csv contains duplicate dates.")

    train = df.loc[df["time"] <= TRAIN_END].copy()
    validation = df.loc[
        (df["time"] >= VALIDATION_START) & (df["time"] <= VALIDATION_END)
    ].copy()
    test = df.loc[df["time"] >= TEST_START].copy()

    # Assert complete, non-overlapping coverage.
    if len(train) + len(validation) + len(test) != len(df):
        raise ValueError("Split rows do not sum to the aligned dataset row count.")

    if train["time"].max() >= validation["time"].min():
        raise ValueError("Train and validation overlap.")
    if validation["time"].max() >= test["time"].min():
        raise ValueError("Validation and test overlap.")

    paths = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "validation.csv",
        "test": splits_dir / "test.csv",
    }

    train.to_csv(paths["train"], index=False, date_format="%Y-%m-%d")
    validation.to_csv(paths["validation"], index=False, date_format="%Y-%m-%d")
    test.to_csv(paths["test"], index=False, date_format="%Y-%m-%d")

    summary = {
        "policy": {
            "method": "chronological_no_shuffle",
            "train_end": TRAIN_END.strftime("%Y-%m-%d"),
            "validation_start": VALIDATION_START.strftime("%Y-%m-%d"),
            "validation_end": VALIDATION_END.strftime("%Y-%m-%d"),
            "test_start": TEST_START.strftime("%Y-%m-%d"),
            "rule": (
                "Fit/calibrate simulator and statistical preprocessing on train only. "
                "Use validation for selection/tuning. Keep test untouched until final OOS evaluation."
            ),
        },
        "full": describe_split("full", df),
        "train": describe_split("train", train),
        "validation": describe_split("validation", validation),
        "test": describe_split("test", test),
    }

    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("CHRONOLOGICAL REAL-DATA SPLIT COMPLETE")
    print("=" * 78)
    print(
        f"Full       : {len(df):,} rows | "
        f"{df['time'].min().date()} -> {df['time'].max().date()}"
    )
    print(
        f"Train      : {len(train):,} rows | "
        f"{train['time'].min().date()} -> {train['time'].max().date()}"
    )
    print(
        f"Validation : {len(validation):,} rows | "
        f"{validation['time'].min().date()} -> {validation['time'].max().date()}"
    )
    print(
        f"Test       : {len(test):,} rows | "
        f"{test['time'].min().date()} -> {test['time'].max().date()}"
    )
    print(f"Saved to   : {splits_dir}")
    print()
    print("Leakage rule: calibration=train only; selection=validation; final evaluation=test only.")


if __name__ == "__main__":
    main()
