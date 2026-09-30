"""Cost-aware descriptive OOS evaluation of the FROZEN Deep Hedging models.

This script MUST NOT be used for model selection or retuning. The real TEST was
already opened in step 23. Step 24 only applies a pre-specified market-friction
layer to the already frozen policies and traditional baselines.

Economic PnL convention
-----------------------
We evaluate one continuous "contract-equivalent" VN30 spot exposure:
    spot PnL_t    = multiplier * (S_{t+1} - S_t)
    futures PnL_t = - h_t * multiplier * (F_{t+1} - F_t)

where multiplier = 100,000 VND/index point. This makes h=1 economically
comparable to one VN30 futures contract against one VN30 index-point basket.
The learned h is continuous, so fees are linearly prorated to fractional
contract-equivalents. Integer-contract rounding requires a separate portfolio
notional definition and is intentionally NOT imposed here.

Verified service-fee logic
--------------------------
- HNX index-futures trading fee: 2,700 VND per matched contract.
- Before KRX go-live (2025-05-05): VSDC position-management fee is charged on
  overnight open position, 2,550 VND/contract/account/day.
- From KRX go-live: the overnight position-management fee is replaced by VSDC
  derivative clearing fee, 2,550 VND per matched contract-equivalent.

Slippage sensitivity
--------------------
Scenarios: 0, 0.5, 1, 2 ticks per matched contract-equivalent.
One tick = 0.1 index point, so one tick on one contract-equivalent is
10,000 VND with the 100,000 VND multiplier.

Roll-cost proxy
---------------
VN30F1M is a continuous front-month series without contract IDs. To avoid
assigning zero turnover to constant-hedge strategies, we add a transparent
calendar proxy: on the third-Thursday expiry trading date (or previous observed
trading day in that month), close and reopen the hedge, adding 2*|h| matched
contract-equivalents. This affects fee/slippage costs only; the continuous price
series itself is not altered. Therefore roll economics remain approximate.

Not included in the reported 'service-cost' result
---------------------------------------------------
- Personal-income tax: requires the correct initial-margin schedule by period.
- Broker commission: broker-specific and unset in project config.
- Margin-asset management fee: account/month/collateral dependent.
- Bid/ask slippage: represented only by the stated tick scenarios.

No model or hyperparameter is changed by this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPISODE_STEPS = 60
BATCH_SIZE = 512
TAIL_FRACTION = 0.05
SLIPPAGE_TICKS = (0.0, 0.5, 1.0, 2.0)


class LSTMHedgePolicy(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, hedge_max: float) -> None:
        super().__init__()
        self.hedge_max = float(hedge_max)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        raw = self.head(hidden).squeeze(-1)
        return self.hedge_max * torch.sigmoid(raw)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(path: Path, device: torch.device) -> LSTMHedgePolicy:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen model: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMHedgePolicy(
        int(ckpt["n_features"]),
        int(ckpt["hidden_size"]),
        int(ckpt["num_layers"]),
        float(ckpt["hedge_max"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    dummy = np.zeros(x.shape[:2], dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(dummy), torch.from_numpy(dummy))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in dl:
            chunks.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def minvar_ratio(sr: np.ndarray, fr: np.ndarray) -> float:
    var_f = float(np.var(fr, ddof=1))
    if var_f <= 0:
        raise ValueError("Non-positive futures return variance")
    return float(np.cov(sr, fr, ddof=1)[0, 1] / var_f)


def build_causal_rolling60(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    starts: np.ndarray,
) -> np.ndarray:
    pre = pd.concat(
        [
            train[["time", "spot_close", "fut_close"]],
            validation[["time", "spot_close", "fut_close"]],
        ],
        ignore_index=True,
    )
    combined = pd.concat(
        [pre, test[["time", "spot_close", "fut_close"]]], ignore_index=True
    ).sort_values("time").reset_index(drop=True)
    combined["sr"] = np.log(combined["spot_close"]).diff()
    combined["fr"] = np.log(combined["fut_close"]).diff()

    first_test_idx = len(pre)
    h_daily = np.empty(len(test), dtype=np.float64)
    for j in range(len(test)):
        idx = first_test_idx + j
        hist = combined.iloc[max(0, idx - 60) : idx][["sr", "fr"]].dropna()
        if len(hist) < 20:
            raise ValueError("Insufficient causal history for rolling min-var")
        h_daily[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())

    h = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        h[i] = h_daily[start : start + EPISODE_STEPS]
    return h


def third_thursday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_thursday = (3 - first.weekday()) % 7
    first_thu = first + pd.Timedelta(days=days_to_thursday)
    return first_thu + pd.Timedelta(days=14)


def observed_expiry_roll_dates(test_dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(test_dates)).normalize()
    out: set[pd.Timestamp] = set()
    months = sorted({(d.year, d.month) for d in dates})
    for year, month in months:
        target = third_thursday(year, month)
        month_dates = dates[(dates.year == year) & (dates.month == month)]
        eligible = month_dates[month_dates <= target]
        if len(eligible):
            out.add(pd.Timestamp(eligible.max()).normalize())
    return out


def cvar_loss(pnl: np.ndarray) -> float:
    losses = -np.asarray(pnl, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k :]
    return float(np.mean(losses[idx]))


def build_price_change_episodes(
    test: pd.DataFrame, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    s = test["spot_close"].to_numpy(dtype=np.float64)
    f = test["fut_close"].to_numpy(dtype=np.float64)
    ds = np.empty((len(starts), EPISODE_STEPS), dtype=np.float64)
    df = np.empty_like(ds)
    for i, start in enumerate(starts):
        ds[i] = np.diff(s[start : start + EPISODE_STEPS + 1])
        df[i] = np.diff(f[start : start + EPISODE_STEPS + 1])
    return ds, df


def fee_components_for_episode(
    h: np.ndarray,
    episode_dates: pd.DatetimeIndex,
    multiplier: float,
    tick_size: float,
    hnx_fee: float,
    vsdc_pre_position_fee: float,
    vsdc_post_clearing_fee: float,
    krx_date: pd.Timestamp,
    roll_dates: set[pd.Timestamp],
    slippage_ticks: float,
) -> dict[str, float]:
    h = np.asarray(h, dtype=np.float64)
    if len(h) != EPISODE_STEPS or len(episode_dates) != EPISODE_STEPS + 1:
        raise ValueError("Episode shape mismatch in fee calculation")

    # Trades: open at first decision date, rebalance on subsequent decision dates,
    # close at the episode end date.
    trade_qty = np.zeros(EPISODE_STEPS + 1, dtype=np.float64)
    trade_qty[0] = abs(h[0])
    trade_qty[1:EPISODE_STEPS] = np.abs(h[1:] - h[:-1])
    trade_qty[-1] = abs(h[-1])

    roll_qty = np.zeros_like(trade_qty)
    normalized_dates = pd.DatetimeIndex(episode_dates).normalize()
    # Roll only while the strategy continues beyond the roll date. At terminal
    # date we simply close; no new contract is required.
    for t in range(EPISODE_STEPS):
        d = pd.Timestamp(normalized_dates[t])
        if d in roll_dates:
            roll_qty[t] += 2.0 * abs(h[t])

    matched_qty = trade_qty + roll_qty
    hnx = float(np.sum(matched_qty) * hnx_fee)

    post_mask = normalized_dates >= krx_date
    vsdc_clearing = float(np.sum(matched_qty[post_mask]) * vsdc_post_clearing_fee)

    # h_t is held over t -> t+1. Before KRX this represents an overnight open
    # position subject to the legacy VSDC position-management fee.
    hold_dates = normalized_dates[:EPISODE_STEPS]
    pre_hold_mask = hold_dates < krx_date
    vsdc_position = float(np.sum(np.abs(h[pre_hold_mask])) * vsdc_pre_position_fee)

    slippage_vnd_per_contract = slippage_ticks * tick_size * multiplier
    slippage = float(np.sum(matched_qty) * slippage_vnd_per_contract)

    return {
        "base_trade_qty": float(np.sum(trade_qty)),
        "roll_trade_qty_proxy": float(np.sum(roll_qty)),
        "matched_contract_equivalent": float(np.sum(matched_qty)),
        "hnx_fee_vnd": hnx,
        "vsdc_position_fee_vnd": vsdc_position,
        "vsdc_clearing_fee_vnd": vsdc_clearing,
        "slippage_vnd": slippage,
        "total_service_and_slippage_vnd": hnx + vsdc_position + vsdc_clearing + slippage,
    }


def evaluate_strategy(
    name: str,
    h: np.ndarray,
    starts: np.ndarray,
    test: pd.DataFrame,
    ds: np.ndarray,
    df: np.ndarray,
    config: dict,
    roll_dates: set[pd.Timestamp],
    slippage_ticks: float,
) -> dict:
    multiplier = float(config["market"]["contract_multiplier_vnd_per_index_point"])
    tick_size = float(config["market"]["tick_size_index_points"])
    fees = config["official_service_fees"]
    hnx_fee = float(fees["hnx_index_futures_trading_fee_vnd_per_contract_per_match"])
    vsdc_pre = float(fees["vsdc_pre_krx_position_management_fee_vnd_per_contract_account_day"])
    vsdc_post = float(fees["vsdc_post_krx_clearing_fee_vnd_per_contract_offset"])
    krx_date = pd.Timestamp(fees["krx_go_live_date"])

    gross_daily = multiplier * (ds - h.astype(np.float64) * df)
    gross_path = np.sum(gross_daily, axis=1)

    total_cost = np.empty(len(starts), dtype=np.float64)
    hnx_arr = np.empty_like(total_cost)
    vsdc_position_arr = np.empty_like(total_cost)
    vsdc_clearing_arr = np.empty_like(total_cost)
    slippage_arr = np.empty_like(total_cost)
    matched_arr = np.empty_like(total_cost)
    roll_arr = np.empty_like(total_cost)

    dates = pd.DatetimeIndex(test["time"])
    for i, start in enumerate(starts):
        episode_dates = dates[start : start + EPISODE_STEPS + 1]
        comp = fee_components_for_episode(
            h[i],
            episode_dates,
            multiplier,
            tick_size,
            hnx_fee,
            vsdc_pre,
            vsdc_post,
            krx_date,
            roll_dates,
            slippage_ticks,
        )
        total_cost[i] = comp["total_service_and_slippage_vnd"]
        hnx_arr[i] = comp["hnx_fee_vnd"]
        vsdc_position_arr[i] = comp["vsdc_position_fee_vnd"]
        vsdc_clearing_arr[i] = comp["vsdc_clearing_fee_vnd"]
        slippage_arr[i] = comp["slippage_vnd"]
        matched_arr[i] = comp["matched_contract_equivalent"]
        roll_arr[i] = comp["roll_trade_qty_proxy"]

    net_path = gross_path - total_cost
    initial_notional = test["spot_close"].to_numpy(dtype=np.float64)[starts] * multiplier

    return {
        "strategy": name,
        "slippage_ticks": float(slippage_ticks),
        "episodes": int(len(starts)),
        "gross_mean_vnd": float(np.mean(gross_path)),
        "gross_std_vnd": float(np.std(gross_path, ddof=1)),
        "gross_q05_vnd": float(np.quantile(gross_path, 0.05)),
        "gross_cvar_loss_5pct_vnd": cvar_loss(gross_path),
        "net_mean_vnd": float(np.mean(net_path)),
        "net_std_vnd": float(np.std(net_path, ddof=1)),
        "net_q05_vnd": float(np.quantile(net_path, 0.05)),
        "net_cvar_loss_5pct_vnd": cvar_loss(net_path),
        "mean_total_cost_vnd": float(np.mean(total_cost)),
        "mean_total_cost_bps_of_initial_notional": float(
            np.mean(total_cost / initial_notional) * 10000.0
        ),
        "mean_hnx_fee_vnd": float(np.mean(hnx_arr)),
        "mean_vsdc_prekrx_position_fee_vnd": float(np.mean(vsdc_position_arr)),
        "mean_vsdc_postkrx_clearing_fee_vnd": float(np.mean(vsdc_clearing_arr)),
        "mean_slippage_vnd": float(np.mean(slippage_arr)),
        "mean_matched_contract_equivalent": float(np.mean(matched_arr)),
        "mean_roll_trade_qty_proxy": float(np.mean(roll_arr)),
        "hedge_ratio_mean": float(np.mean(h)),
        "hedge_ratio_std": float(np.std(h, ddof=1)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "vn30f1m_market_friction.yaml")

    splits_dir = root / "data" / "processed" / "splits"
    train = pd.read_csv(splits_dir / "train.csv")
    validation = pd.read_csv(splits_dir / "validation.csv")
    test = pd.read_csv(splits_dir / "test.csv")
    for df in (train, validation, test):
        df["time"] = pd.to_datetime(df["time"], errors="raise")
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    seq_path = root / "data" / "processed" / "deep_hedging_real" / "test_sequences_final_oos.npz"
    if not seq_path.exists():
        raise FileNotFoundError("Run step 23 first: missing final OOS test sequences")
    with np.load(seq_path) as z:
        x_test = z["features"].astype(np.float32)
        starts = z["episode_start_index"].astype(int)

    if x_test.shape[1] != EPISODE_STEPS:
        raise ValueError("Unexpected test episode length")

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(dtype=np.float64)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(dtype=np.float64)))
    static_h_value = minvar_ratio(train_sr, train_fr)

    h_unhedged = np.zeros((len(starts), EPISODE_STEPS), dtype=np.float32)
    h_delta = np.ones_like(h_unhedged)
    h_static = np.full_like(h_unhedged, static_h_value)
    h_rolling = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = load_model(
        root / "results" / "models" / "lstm_cvar_turnover_best.pt", device
    )
    final_model = load_model(
        root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt", device
    )
    h_pre = predict(pretrained_model, x_test, device)
    h_final = predict(final_model, x_test, device)

    strategy_map = {
        "unhedged": h_unhedged,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    ds, df = build_price_change_episodes(test, starts)
    roll_dates = observed_expiry_roll_dates(pd.DatetimeIndex(test["time"]))

    rows: list[dict] = []
    for slip in SLIPPAGE_TICKS:
        for name, h in strategy_map.items():
            rows.append(
                evaluate_strategy(
                    name,
                    h,
                    starts,
                    test,
                    ds,
                    df,
                    config,
                    roll_dates,
                    slip,
                )
            )

    result = pd.DataFrame(rows)
    tables_dir = root / "results" / "tables"
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    for d in (tables_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / "cost_aware_oos_frozen_comparison.csv"
    result.to_csv(table_path, index=False)

    comparisons: dict[str, dict] = {}
    for slip in SLIPPAGE_TICKS:
        sub = result[result["slippage_ticks"] == slip]
        final = sub[sub["strategy"] == "real_finetuned_lstm"].iloc[0]
        comparisons[str(slip)] = {}
        for baseline in ("delta_1to1", "real_train_static_minvar", "causal_rolling60_minvar"):
            b = sub[sub["strategy"] == baseline].iloc[0]
            comparisons[str(slip)][baseline] = {
                "net_cvar_reduction_fraction": float(
                    1.0 - final["net_cvar_loss_5pct_vnd"] / b["net_cvar_loss_5pct_vnd"]
                ),
                "net_mean_difference_vnd": float(final["net_mean_vnd"] - b["net_mean_vnd"]),
            }

    report = {
        "status": "POST_FREEZE_DESCRIPTIVE_COST_ANALYSIS_ONLY",
        "real_test_already_unsealed_in_step23": True,
        "model_retuned": False,
        "economic_pnl_convention": (
            "one continuous contract-equivalent spot exposure; multiplier times index-point change"
        ),
        "fractional_contract_equivalent": True,
        "integer_contract_rounding_applied": False,
        "slippage_ticks_scenarios": list(SLIPPAGE_TICKS),
        "expiry_roll_cost_proxy_dates": [d.strftime("%Y-%m-%d") for d in sorted(roll_dates)],
        "included_costs": [
            "HNX trading fee",
            "pre-KRX VSDC overnight position-management fee",
            "post-KRX VSDC clearing fee",
            "slippage tick sensitivity",
            "calendar expiry roll-cost proxy",
        ],
        "excluded_costs": [
            "personal income tax (historical initial-margin schedule incomplete)",
            "broker commission (broker-specific)",
            "margin asset management fee (account/month/collateral dependent)",
        ],
        "results": rows,
        "final_model_vs_baselines": comparisons,
        "warning": (
            "Do not use these TEST results to retune the model. VN30F1M roll handling is a calendar cost proxy "
            "because the continuous daily file has no explicit contract IDs."
        ),
    }
    report_path = metrics_dir / "cost_aware_oos_frozen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(10, 5))
    for strategy in ("delta_1to1", "causal_rolling60_minvar", "synthetic_pretrained_lstm", "real_finetuned_lstm"):
        sub = result[result["strategy"] == strategy].sort_values("slippage_ticks")
        plt.plot(sub["slippage_ticks"], sub["net_cvar_loss_5pct_vnd"], marker="o", label=strategy)
    plt.xlabel("Slippage assumption (ticks per matched contract-equivalent)")
    plt.ylabel("Net 60-day CVaR loss (VND)")
    plt.title("Frozen OOS Cost Sensitivity")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "cost_aware_oos_cvar_slippage_sensitivity.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=" * 132)
    print("COST-AWARE OOS EVALUATION COMPLETE — FROZEN MODELS, NO RETUNING")
    print("=" * 132)
    print(f"Test period                  : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Episodes                     : {len(starts):,} overlapping 60-day paths")
    print(f"Device                       : {device}")
    print(f"Contract multiplier          : {config['market']['contract_multiplier_vnd_per_index_point']:,} VND/point")
    print(f"Tick size                    : {config['market']['tick_size_index_points']} point")
    print(f"HNX fee                      : {config['official_service_fees']['hnx_index_futures_trading_fee_vnd_per_contract_per_match']:,} VND/matched contract")
    print(f"VSDC pre-KRX position fee    : {config['official_service_fees']['vsdc_pre_krx_position_management_fee_vnd_per_contract_account_day']:,} VND/contract/day")
    print(f"VSDC post-KRX clearing fee   : {config['official_service_fees']['vsdc_post_krx_clearing_fee_vnd_per_contract_offset']:,} VND/matched contract")
    print(f"KRX go-live                  : {config['official_service_fees']['krx_go_live_date']}")
    print(f"Expiry roll proxy dates      : {', '.join(d.strftime('%Y-%m-%d') for d in sorted(roll_dates))}")
    print()

    display = result[
        [
            "slippage_ticks",
            "strategy",
            "net_mean_vnd",
            "net_q05_vnd",
            "net_cvar_loss_5pct_vnd",
            "mean_total_cost_vnd",
            "mean_total_cost_bps_of_initial_notional",
            "mean_matched_contract_equivalent",
            "mean_roll_trade_qty_proxy",
            "hedge_ratio_mean",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("FINAL FROZEN MODEL — NET CVaR REDUCTION VS BASELINES")
    for slip in SLIPPAGE_TICKS:
        print(f"slippage={slip:.1f} ticks")
        for baseline, vals in comparisons[str(slip)].items():
            print(
                f"  {baseline:26s} | CVaR reduction={vals['net_cvar_reduction_fraction']:+.2%} | "
                f"mean PnL diff={vals['net_mean_difference_vnd']:+,.0f} VND"
            )

    print()
    print("EXCLUDED: PIT tax, broker commission, margin-asset management fee.")
    print("IMPORTANT: This is post-freeze descriptive TEST analysis only. DO NOT RETUNE FROM THESE RESULTS.")
    print(f"Table  : {table_path}")
    print(f"Report : {report_path}")
    print(f"Figure : {figure_path}")


if __name__ == "__main__":
    main()
