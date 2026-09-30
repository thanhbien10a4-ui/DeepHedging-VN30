"""Diagnose VN30F1M rollover effects and extreme observations before Heston estimation.

Input:
    data/processed/aligned_1d.csv

Outputs:
    data/processed/rollover_diagnostics_summary.json
    results/tables/top_basis_velocity.csv
    results/tables/top_futures_return.csv
    results/tables/top_spot_return.csv
    results/tables/rollover_flagged_dates.csv
    results/figures/basis_velocity_diagnostics.png
    results/figures/return_gap_diagnostics.png

Important:
- This script does NOT delete or modify any observations.
- "Near expiry" is a calendar heuristic based on the official VN30 futures
  final trading day rule (third Thursday of expiry month; preceding trading
  day when that Thursday is a holiday). Exact holiday-adjusted expiry dates
  should be verified separately if a flagged date is used in the final report.
"""

from __future__ import annotations

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_N = 20
ROBUST_Z_THRESHOLD = 4.0
MARKET_SHOCK_Z_THRESHOLD = 3.0
EXPIRY_WINDOW_CALENDAR_DAYS = 3


def robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD robust z-score; stable under heavy tails."""
    x = pd.to_numeric(series, errors="coerce")
    median = x.median()
    mad = (x - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - median) / scale


def third_thursday(year: int, month: int) -> date:
    """Return the calendar date of the third Thursday of a month."""
    cal = calendar.monthcalendar(year, month)
    thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0]
    return date(year, month, thursdays[2])


def same_sign(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.sign(a) == np.sign(b)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "aligned_1d.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run 01_validate_and_align.py first."
        )

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d", errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    # Recompute diagnostics directly from aligned raw states.
    df["spot_log_return"] = np.log(df["spot_close"]).diff()
    df["fut_log_return"] = np.log(df["fut_close"]).diff()
    df["basis_pct"] = (df["fut_close"] - df["spot_close"]) / df["spot_close"]
    df["basis_velocity_1"] = df["basis_pct"].diff()
    df["return_gap"] = df["fut_log_return"] - df["spot_log_return"]

    # Robust outlier scores are preferable to standard z-scores here because
    # the EDA already showed strong skewness and excess kurtosis.
    for col in ["spot_log_return", "fut_log_return", "basis_velocity_1", "return_gap"]:
        df[f"rz_{col}"] = robust_zscore(df[col])

    # Calendar heuristic around the standard monthly VN30 expiry date.
    expiry_dates = [third_thursday(ts.year, ts.month) for ts in df["time"]]
    df["third_thursday"] = pd.to_datetime(expiry_dates)
    df["days_from_third_thursday"] = (
        df["time"].dt.normalize() - df["third_thursday"]
    ).dt.days
    df["near_expiry_calendar"] = (
        df["days_from_third_thursday"].abs() <= EXPIRY_WINDOW_CALENDAR_DAYS
    )

    # Classification is diagnostic only; it is deliberately conservative.
    df["market_shock"] = (
        (df["rz_spot_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & (df["rz_fut_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & same_sign(df["spot_log_return"], df["fut_log_return"])
    )

    df["futures_specific_jump"] = (
        (df["rz_fut_log_return"].abs() >= ROBUST_Z_THRESHOLD)
        & (df["rz_spot_log_return"].abs() < MARKET_SHOCK_Z_THRESHOLD)
    )

    df["basis_jump"] = (
        df["rz_basis_velocity_1"].abs() >= ROBUST_Z_THRESHOLD
    )

    df["possible_roll_artifact"] = (
        df["basis_jump"]
        & (
            df["near_expiry_calendar"]
            | (df["rz_return_gap"].abs() >= ROBUST_Z_THRESHOLD)
            | df["futures_specific_jump"]
        )
    )

    display_cols = [
        "time",
        "spot_close",
        "fut_close",
        "spot_log_return",
        "fut_log_return",
        "basis_pct",
        "basis_velocity_1",
        "return_gap",
        "rz_spot_log_return",
        "rz_fut_log_return",
        "rz_basis_velocity_1",
        "rz_return_gap",
        "third_thursday",
        "days_from_third_thursday",
        "near_expiry_calendar",
        "market_shock",
        "futures_specific_jump",
        "basis_jump",
        "possible_roll_artifact",
    ]

    def top_abs(column: str) -> pd.DataFrame:
        tmp = df.loc[df[column].notna(), display_cols].copy()
        return tmp.loc[tmp[column].abs().nlargest(TOP_N).index].sort_values(
            by=column, key=lambda s: s.abs(), ascending=False
        )

    top_basis = top_abs("basis_velocity_1")
    top_fut = top_abs("fut_log_return")
    top_spot = top_abs("spot_log_return")

    flagged = df.loc[
        df["possible_roll_artifact"]
        | df["futures_specific_jump"]
        | df["market_shock"],
        display_cols,
    ].copy()
    flagged = flagged.sort_values("time").reset_index(drop=True)

    top_basis.to_csv(tables_dir / "top_basis_velocity.csv", index=False, date_format="%Y-%m-%d")
    top_fut.to_csv(tables_dir / "top_futures_return.csv", index=False, date_format="%Y-%m-%d")
    top_spot.to_csv(tables_dir / "top_spot_return.csv", index=False, date_format="%Y-%m-%d")
    flagged.to_csv(tables_dir / "rollover_flagged_dates.csv", index=False, date_format="%Y-%m-%d")

    # Figures: one plot per figure (no subplots) for easy report reuse.
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["basis_velocity_1"], linewidth=0.8)
    flagged_roll = df[df["possible_roll_artifact"]]
    if not flagged_roll.empty:
        plt.scatter(
            flagged_roll["time"],
            flagged_roll["basis_velocity_1"],
            s=22,
            label="Possible roll artifact",
        )
        plt.legend()
    plt.title("VN30F1M Basis Velocity Diagnostics")
    plt.xlabel("Date")
    plt.ylabel("Daily change in basis_pct")
    plt.tight_layout()
    fig.savefig(figures_dir / "basis_velocity_diagnostics.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["return_gap"], linewidth=0.8)
    fut_jump = df[df["futures_specific_jump"]]
    if not fut_jump.empty:
        plt.scatter(
            fut_jump["time"],
            fut_jump["return_gap"],
            s=22,
            label="Futures-specific jump",
        )
        plt.legend()
    plt.title("VN30F1M minus VN30 Daily Log-Return Gap")
    plt.xlabel("Date")
    plt.ylabel("Futures return - spot return")
    plt.tight_layout()
    fig.savefig(figures_dir / "return_gap_diagnostics.png", dpi=160)
    plt.close(fig)

    summary = {
        "rows": int(len(df)),
        "date_min": df["time"].min().strftime("%Y-%m-%d"),
        "date_max": df["time"].max().strftime("%Y-%m-%d"),
        "thresholds": {
            "top_n": TOP_N,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "market_shock_z_threshold": MARKET_SHOCK_Z_THRESHOLD,
            "expiry_window_calendar_days": EXPIRY_WINDOW_CALENDAR_DAYS,
        },
        "counts": {
            "market_shock": int(df["market_shock"].sum()),
            "futures_specific_jump": int(df["futures_specific_jump"].sum()),
            "basis_jump": int(df["basis_jump"].sum()),
            "near_expiry_calendar": int(df["near_expiry_calendar"].sum()),
            "possible_roll_artifact": int(df["possible_roll_artifact"].sum()),
        },
        "largest_abs_basis_velocity": [
            {
                "time": row.time.strftime("%Y-%m-%d"),
                "basis_velocity_1": float(row.basis_velocity_1),
                "basis_pct": float(row.basis_pct),
                "near_expiry_calendar": bool(row.near_expiry_calendar),
                "possible_roll_artifact": bool(row.possible_roll_artifact),
            }
            for row in top_basis.head(10).itertuples()
        ],
        "policy": {
            "raw_data_mutated": False,
            "automatic_outlier_deletion": False,
            "note": (
                "Flags are diagnostic only. Confirm contract-roll/expiry context before "
                "excluding or adjusting any observation for Heston estimation."
            ),
        },
    }

    summary_path = processed_dir / "rollover_diagnostics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("ROLLOVER / OUTLIER DIAGNOSTICS COMPLETE")
    print("=" * 72)
    print(f"Rows                     : {len(df):,}")
    print(f"Market shocks            : {int(df['market_shock'].sum())}")
    print(f"Futures-specific jumps   : {int(df['futures_specific_jump'].sum())}")
    print(f"Basis jumps              : {int(df['basis_jump'].sum())}")
    print(f"Possible roll artifacts  : {int(df['possible_roll_artifact'].sum())}")
    print(f"Summary                  : {summary_path}")
    print(f"Tables                   : {tables_dir}")
    print(f"Figures                  : {figures_dir}")
    print()
    print("Top 10 absolute basis-velocity dates:")
    for row in top_basis.head(10).itertuples():
        print(
            f"  {row.time.date()} | dBasis={row.basis_velocity_1:+.6f} "
            f"| expiry_near={bool(row.near_expiry_calendar)} "
            f"| roll_flag={bool(row.possible_roll_artifact)}"
        )


if __name__ == "__main__":
    main()
