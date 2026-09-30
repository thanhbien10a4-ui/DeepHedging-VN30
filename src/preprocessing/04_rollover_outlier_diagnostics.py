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
- "Near expiry" is a calendar heuristic based on the standard VN30 futures
  final trading day convention (third Thursday of expiry month; exact
  holiday-adjusted dates should be verified separately before excluding data).
- A basis jump is NOT automatically treated as a rollover artifact. Market
  shocks, non-expiry basis dislocations and expiry-adjacent anomalies are
  reported separately.
"""

from __future__ import annotations

import calendar
import json
from datetime import date
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

    # Robust scores are preferable to ordinary z-scores because the EDA showed
    # substantial skewness and excess kurtosis.
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

    # Market-wide shock: spot and futures both move abnormally in the same direction.
    df["market_shock"] = (
        (df["rz_spot_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & (df["rz_fut_log_return"].abs() >= MARKET_SHOCK_Z_THRESHOLD)
        & same_sign(df["spot_log_return"], df["fut_log_return"])
    )

    # Futures-specific jump: futures moves abnormally while spot does not.
    df["futures_specific_jump"] = (
        (df["rz_fut_log_return"].abs() >= ROBUST_Z_THRESHOLD)
        & (df["rz_spot_log_return"].abs() < MARKET_SHOCK_Z_THRESHOLD)
    )

    df["basis_jump"] = df["rz_basis_velocity_1"].abs() >= ROBUST_Z_THRESHOLD
    df["return_gap_jump"] = df["rz_return_gap"].abs() >= ROBUST_Z_THRESHOLD

    # IMPORTANT: basis_velocity and return_gap are mechanically related, so a
    # large return gap must NOT be used by itself to call something a roll artifact.
    # We separate the cases instead.
    df["expiry_basis_jump"] = df["basis_jump"] & df["near_expiry_calendar"]

    # Strongest rollover candidate: large basis jump close to expiry that is not
    # already explained as a broad market shock. This is still only a diagnostic
    # flag, not proof of an actual contract switch.
    df["likely_roll_related"] = df["expiry_basis_jump"] & (~df["market_shock"])

    # Large basis jump close to expiry during a genuine market-wide shock.
    # Keep ambiguous rather than automatically attributing it to rollover.
    df["ambiguous_expiry_shock"] = df["expiry_basis_jump"] & df["market_shock"]

    # Large basis movement away from the expiry window. These observations may
    # represent real market dislocations, data-source continuous-contract effects,
    # liquidity events, or other microstructure effects and need separate review.
    df["non_expiry_basis_dislocation"] = df["basis_jump"] & (~df["near_expiry_calendar"])

    # Backward-compatible name used by the first version of this diagnostic.
    df["possible_roll_artifact"] = df["likely_roll_related"]

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
        "return_gap_jump",
        "expiry_basis_jump",
        "likely_roll_related",
        "ambiguous_expiry_shock",
        "non_expiry_basis_dislocation",
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
        df["basis_jump"]
        | df["futures_specific_jump"]
        | df["market_shock"],
        display_cols,
    ].copy()
    flagged = flagged.sort_values("time").reset_index(drop=True)

    top_basis.to_csv(tables_dir / "top_basis_velocity.csv", index=False, date_format="%Y-%m-%d")
    top_fut.to_csv(tables_dir / "top_futures_return.csv", index=False, date_format="%Y-%m-%d")
    top_spot.to_csv(tables_dir / "top_spot_return.csv", index=False, date_format="%Y-%m-%d")
    flagged.to_csv(tables_dir / "rollover_flagged_dates.csv", index=False, date_format="%Y-%m-%d")

    # Figures: one plot per figure for easy report reuse.
    fig = plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["basis_velocity_1"], linewidth=0.8)
    roll_candidates = df[df["likely_roll_related"]]
    if not roll_candidates.empty:
        plt.scatter(
            roll_candidates["time"],
            roll_candidates["basis_velocity_1"],
            s=22,
            label="Likely roll-related candidate",
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
            "return_gap_jump": int(df["return_gap_jump"].sum()),
            "near_expiry_calendar": int(df["near_expiry_calendar"].sum()),
            "expiry_basis_jump": int(df["expiry_basis_jump"].sum()),
            "likely_roll_related": int(df["likely_roll_related"].sum()),
            "ambiguous_expiry_shock": int(df["ambiguous_expiry_shock"].sum()),
            "non_expiry_basis_dislocation": int(df["non_expiry_basis_dislocation"].sum()),
            "possible_roll_artifact": int(df["possible_roll_artifact"].sum()),
        },
        "largest_abs_basis_velocity": [
            {
                "time": row.time.strftime("%Y-%m-%d"),
                "basis_velocity_1": float(row.basis_velocity_1),
                "basis_pct": float(row.basis_pct),
                "near_expiry_calendar": bool(row.near_expiry_calendar),
                "market_shock": bool(row.market_shock),
                "likely_roll_related": bool(row.likely_roll_related),
                "ambiguous_expiry_shock": bool(row.ambiguous_expiry_shock),
                "non_expiry_basis_dislocation": bool(row.non_expiry_basis_dislocation),
            }
            for row in top_basis.head(10).itertuples()
        ],
        "policy": {
            "raw_data_mutated": False,
            "automatic_outlier_deletion": False,
            "note": (
                "Flags are diagnostic only. Do not exclude observations automatically. "
                "Verify contract-roll/expiry context and distinguish market shocks from "
                "continuous-contract construction effects before Heston estimation."
            ),
        },
    }

    summary_path = processed_dir / "rollover_diagnostics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("ROLLOVER / OUTLIER DIAGNOSTICS COMPLETE (v2)")
    print("=" * 72)
    print(f"Rows                         : {len(df):,}")
    print(f"Market shocks                : {int(df['market_shock'].sum())}")
    print(f"Futures-specific jumps       : {int(df['futures_specific_jump'].sum())}")
    print(f"Basis jumps                  : {int(df['basis_jump'].sum())}")
    print(f"Expiry-adjacent basis jumps  : {int(df['expiry_basis_jump'].sum())}")
    print(f"Likely roll-related          : {int(df['likely_roll_related'].sum())}")
    print(f"Ambiguous expiry shocks      : {int(df['ambiguous_expiry_shock'].sum())}")
    print(f"Non-expiry basis dislocation : {int(df['non_expiry_basis_dislocation'].sum())}")
    print(f"Summary                      : {summary_path}")
    print(f"Tables                       : {tables_dir}")
    print(f"Figures                      : {figures_dir}")
    print()
    print("Top 10 absolute basis-velocity dates:")
    for row in top_basis.head(10).itertuples():
        print(
            f"  {row.time.date()} | dBasis={row.basis_velocity_1:+.6f} "
            f"| expiry_near={bool(row.near_expiry_calendar)} "
            f"| market_shock={bool(row.market_shock)} "
            f"| likely_roll={bool(row.likely_roll_related)}"
        )


if __name__ == "__main__":
    main()
