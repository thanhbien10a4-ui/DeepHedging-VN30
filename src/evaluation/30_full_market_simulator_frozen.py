"""Full VN30F1M market simulator for frozen OOS hedging strategies.

This step implements the project-specified market mechanics after the model has
already been frozen and TEST opened. It is therefore descriptive evaluation only
and MUST NOT be used for retuning.

Project assumptions frozen in configs/vn30f1m_market_friction.yaml
-----------------------------------------------------------------
- VN30 spot portfolio initial notional: 10,000,000,000 VND
- Futures contract multiplier: 100,000 VND/index point
- Initial margin ratio assumption: 1.0 (100%, project simplification)
- Integer futures contracts, nearest-integer half-up rounding
- Broker commission: 3,000 VND per matched contract
- HNX/VSDC service fees from config
- PIT formula uses the project 100% margin assumption
- Default slippage: 1 tick = 0.1 index point
- Calendar expiry roll proxy: close old + open next contract
- Daily mark-to-market futures PnL

Important limitation
--------------------
VN30F1M is a continuous front-month daily series without explicit contract IDs.
Roll execution therefore affects transaction costs through a calendar proxy but
does not alter the continuous price series itself.
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
        y, _ = self.lstm(x)
        return self.hedge_max * torch.sigmoid(self.head(y).squeeze(-1))


def load_model(path: Path, device: torch.device) -> LSTMHedgePolicy:
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
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in dl:
            out.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def minvar_ratio(sr: np.ndarray, fr: np.ndarray) -> float:
    var_f = float(np.var(fr, ddof=1))
    if var_f <= 0:
        raise ValueError("Non-positive futures variance")
    return float(np.cov(sr, fr, ddof=1)[0, 1] / var_f)


def build_causal_rolling60(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    pre = pd.concat(
        [train[["time", "spot_close", "fut_close"]], validation[["time", "spot_close", "fut_close"]]],
        ignore_index=True,
    )
    combined = pd.concat([pre, test[["time", "spot_close", "fut_close"]]], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    combined["sr"] = np.log(combined["spot_close"]).diff()
    combined["fr"] = np.log(combined["fut_close"]).diff()
    first_test = len(pre)
    day_h = np.empty(len(test), dtype=np.float64)
    for j in range(len(test)):
        hist = combined.iloc[max(0, first_test + j - 60): first_test + j][["sr", "fr"]].dropna()
        if len(hist) < 20:
            raise ValueError("Insufficient causal history")
        day_h[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())
    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = day_h[start:start + EPISODE_STEPS]
    return out


def round_half_up_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.floor(np.maximum(x, 0.0) + 0.5).astype(np.int64)


def third_thursday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (3 - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 14)


def observed_roll_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(dates).normalize()
    out: set[pd.Timestamp] = set()
    for year, month in sorted({(d.year, d.month) for d in dates}):
        target = third_thursday(year, month)
        md = dates[(dates.year == year) & (dates.month == month)]
        eligible = md[md <= target]
        if len(eligible):
            out.add(pd.Timestamp(eligible.max()).normalize())
    return out


def cvar_loss(pnl: np.ndarray) -> float:
    losses = -np.asarray(pnl, dtype=np.float64).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k:]
    return float(np.mean(losses[idx]))


def strategy_contract_targets(
    h: np.ndarray,
    spot_path: np.ndarray,
    fut_path: np.ndarray,
    initial_portfolio_vnd: float,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Decision t uses S_t, F_t and h_t for the next interval.
    spot_now = spot_path[:-1]
    fut_now = fut_path[:-1]
    current_portfolio = initial_portfolio_vnd * spot_now / spot_path[0]
    desired_contracts = h.astype(np.float64) * current_portfolio / (fut_now * multiplier)
    integer_contracts = round_half_up_nonnegative(desired_contracts)
    realized_h = integer_contracts * fut_now * multiplier / current_portfolio
    return integer_contracts, desired_contracts, realized_h


def simulate_episode(
    h: np.ndarray,
    spot_path: np.ndarray,
    fut_path: np.ndarray,
    dates: pd.DatetimeIndex,
    roll_dates: set[pd.Timestamp],
    cfg: dict,
) -> dict[str, float]:
    multiplier = float(cfg["market"]["contract_multiplier_vnd_per_index_point"])
    initial_portfolio = float(cfg["portfolio"]["initial_vn30_portfolio_notional_vnd"])
    fees = cfg["official_service_fees"]
    hnx_fee = float(fees["hnx_index_futures_trading_fee_vnd_per_contract_per_match"])
    broker_fee = float(fees["broker_commission_vnd_per_contract_per_match"])
    vsdc_pre = float(fees["vsdc_pre_krx_position_management_fee_vnd_per_contract_account_day"])
    vsdc_post = float(fees["vsdc_post_krx_clearing_fee_vnd_per_contract_offset"])
    krx = pd.Timestamp(fees["krx_go_live_date"])
    tax_rate = float(cfg["personal_income_tax"]["tax_rate"])
    margin_rate = float(cfg["personal_income_tax"]["initial_margin_rate_assumption"])
    slip_ticks = float(cfg["execution_model"]["default_slippage_ticks"])
    tick_size = float(cfg["market"]["tick_size_index_points"])
    slip_per_contract = slip_ticks * tick_size * multiplier

    n, desired_n, realized_h = strategy_contract_targets(
        h, spot_path, fut_path, initial_portfolio, multiplier
    )

    # Long VN30 portfolio economic PnL.
    spot_units = initial_portfolio / spot_path[0]
    spot_daily_pnl = spot_units * np.diff(spot_path)

    # Short futures daily mark-to-market PnL.
    fut_daily_pnl = -n.astype(np.float64) * multiplier * np.diff(fut_path)
    gross_daily_pnl = spot_daily_pnl + fut_daily_pnl

    # Base matched quantity: open, rebalance, close.
    trade_qty = np.zeros(EPISODE_STEPS + 1, dtype=np.float64)
    trade_qty[0] = float(n[0])
    trade_qty[1:EPISODE_STEPS] = np.abs(np.diff(n)).astype(np.float64)
    trade_qty[-1] = float(n[-1])

    # Calendar roll proxy: close old and open next, no change in net target contracts.
    roll_qty = np.zeros_like(trade_qty)
    ndates = pd.DatetimeIndex(dates).normalize()
    for t in range(EPISODE_STEPS):
        if pd.Timestamp(ndates[t]) in roll_dates:
            roll_qty[t] += 2.0 * float(n[t])

    matched_qty = trade_qty + roll_qty

    hnx_cost = float(np.sum(matched_qty) * hnx_fee)
    broker_cost = float(np.sum(matched_qty) * broker_fee)

    post_mask = ndates >= krx
    vsdc_clearing = float(np.sum(matched_qty[post_mask]) * vsdc_post)

    hold_dates = ndates[:EPISODE_STEPS]
    pre_hold_mask = hold_dates < krx
    vsdc_position = float(np.sum(n[pre_hold_mask].astype(np.float64)) * vsdc_pre)

    # Execution-price proxy uses observed continuous futures close at trade date.
    execution_prices = fut_path.copy()
    tax = float(
        np.sum(
            execution_prices
            * multiplier
            * matched_qty
            * margin_rate
            / 2.0
            * tax_rate
        )
    )

    slippage = float(np.sum(matched_qty) * slip_per_contract)

    total_cost = hnx_cost + broker_cost + vsdc_clearing + vsdc_position + tax + slippage
    gross_terminal = float(np.sum(gross_daily_pnl))
    net_terminal = gross_terminal - total_cost

    margin_required = n.astype(np.float64) * fut_path[:-1] * multiplier * margin_rate

    return {
        "gross_terminal_pnl_vnd": gross_terminal,
        "net_terminal_pnl_vnd": net_terminal,
        "total_cost_vnd": total_cost,
        "hnx_fee_vnd": hnx_cost,
        "broker_fee_vnd": broker_cost,
        "vsdc_position_fee_vnd": vsdc_position,
        "vsdc_clearing_fee_vnd": vsdc_clearing,
        "pit_tax_vnd": tax,
        "slippage_vnd": slippage,
        "matched_contracts": float(np.sum(matched_qty)),
        "roll_contracts_proxy": float(np.sum(roll_qty)),
        "mean_integer_contracts": float(np.mean(n)),
        "min_integer_contracts": int(np.min(n)),
        "max_integer_contracts": int(np.max(n)),
        "mean_desired_contracts": float(np.mean(desired_n)),
        "mean_realized_hedge_ratio": float(np.mean(realized_h)),
        "mean_abs_rounding_error_contracts": float(np.mean(np.abs(n - desired_n))),
        "mean_margin_required_vnd": float(np.mean(margin_required)),
        "max_margin_required_vnd": float(np.max(margin_required)),
        "margin_to_initial_portfolio_max": float(np.max(margin_required) / initial_portfolio),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with open(root / "configs" / "vn30f1m_market_friction.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = root / "data" / "processed" / "splits"
    train = pd.read_csv(splits / "train.csv")
    validation = pd.read_csv(splits / "validation.csv")
    test = pd.read_csv(splits / "test.csv")
    for frame in (train, validation, test):
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    seq_path = root / "data" / "processed" / "deep_hedging_real" / "test_sequences_final_oos.npz"
    with np.load(seq_path) as z:
        x_test = z["features"].astype(np.float32)
        starts = z["episode_start_index"].astype(int)

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(dtype=np.float64)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(dtype=np.float64)))
    static_h = minvar_ratio(train_sr, train_fr)

    shape = (len(starts), EPISODE_STEPS)
    h_zero = np.zeros(shape, dtype=np.float32)
    h_delta = np.ones(shape, dtype=np.float32)
    h_static = np.full(shape, static_h, dtype=np.float32)
    h_rolling = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_pre = predict(
        load_model(root / "results" / "models" / "lstm_cvar_turnover_best.pt", device), x_test, device
    )
    h_final = predict(
        load_model(root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt", device), x_test, device
    )

    strategies = {
        "unhedged": h_zero,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_rolling,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    roll_dates = observed_roll_dates(pd.DatetimeIndex(test["time"]))
    test_dates = pd.DatetimeIndex(test["time"])
    spot = test["spot_close"].to_numpy(dtype=np.float64)
    fut = test["fut_close"].to_numpy(dtype=np.float64)

    episode_rows: list[dict] = []
    summary_rows: list[dict] = []

    for strategy, hmat in strategies.items():
        rows: list[dict] = []
        for i, start in enumerate(starts):
            end = int(start) + EPISODE_STEPS
            rec = simulate_episode(
                hmat[i],
                spot[start:end + 1],
                fut[start:end + 1],
                test_dates[start:end + 1],
                roll_dates,
                cfg,
            )
            rec.update(
                {
                    "strategy": strategy,
                    "episode_index": int(i),
                    "start_date": str(test_dates[start].date()),
                    "end_date": str(test_dates[end].date()),
                }
            )
            rows.append(rec)
            episode_rows.append(rec)

        df = pd.DataFrame(rows)
        gross = df["gross_terminal_pnl_vnd"].to_numpy(dtype=float)
        net = df["net_terminal_pnl_vnd"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "strategy": strategy,
                "episodes": int(len(df)),
                "gross_mean_pnl_vnd": float(np.mean(gross)),
                "gross_cvar_loss_5pct_vnd": cvar_loss(gross),
                "net_mean_pnl_vnd": float(np.mean(net)),
                "net_std_pnl_vnd": float(np.std(net, ddof=1)),
                "net_q05_pnl_vnd": float(np.quantile(net, 0.05)),
                "net_cvar_loss_5pct_vnd": cvar_loss(net),
                "mean_total_cost_vnd": float(df["total_cost_vnd"].mean()),
                "mean_pit_tax_vnd": float(df["pit_tax_vnd"].mean()),
                "mean_broker_fee_vnd": float(df["broker_fee_vnd"].mean()),
                "mean_hnx_fee_vnd": float(df["hnx_fee_vnd"].mean()),
                "mean_vsdc_cost_vnd": float((df["vsdc_position_fee_vnd"] + df["vsdc_clearing_fee_vnd"]).mean()),
                "mean_slippage_vnd": float(df["slippage_vnd"].mean()),
                "mean_integer_contracts": float(df["mean_integer_contracts"].mean()),
                "min_contracts_observed": int(df["min_integer_contracts"].min()),
                "max_contracts_observed": int(df["max_integer_contracts"].max()),
                "mean_realized_hedge_ratio": float(df["mean_realized_hedge_ratio"].mean()),
                "mean_abs_rounding_error_contracts": float(df["mean_abs_rounding_error_contracts"].mean()),
                "mean_margin_required_vnd": float(df["mean_margin_required_vnd"].mean()),
                "max_margin_required_vnd": float(df["max_margin_required_vnd"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    episodes = pd.DataFrame(episode_rows)

    tables = root / "results" / "tables"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"
    for d in (tables, metrics, figures):
        d.mkdir(parents=True, exist_ok=True)

    summary_path = tables / "full_market_simulator_frozen_summary.csv"
    episode_path = tables / "full_market_simulator_frozen_episodes.csv"
    summary.to_csv(summary_path, index=False)
    episodes.to_csv(episode_path, index=False)

    final = summary[summary["strategy"] == "real_finetuned_lstm"].iloc[0]
    comparisons: dict[str, dict[str, float]] = {}
    for baseline in ("delta_1to1", "real_train_static_minvar", "causal_rolling60_minvar"):
        b = summary[summary["strategy"] == baseline].iloc[0]
        comparisons[baseline] = {
            "net_cvar_reduction_fraction": float(
                1.0 - final["net_cvar_loss_5pct_vnd"] / b["net_cvar_loss_5pct_vnd"]
            ),
            "net_mean_pnl_difference_vnd": float(final["net_mean_pnl_vnd"] - b["net_mean_pnl_vnd"]),
        }

    report = {
        "status": "POST_FREEZE_FULL_MARKET_SIMULATOR",
        "model_retuned": False,
        "portfolio_initial_notional_vnd": cfg["portfolio"]["initial_vn30_portfolio_notional_vnd"],
        "initial_margin_rate_assumption": cfg["personal_income_tax"]["initial_margin_rate_assumption"],
        "broker_fee_vnd_per_contract": cfg["official_service_fees"]["broker_commission_vnd_per_contract_per_match"],
        "default_slippage_ticks": cfg["execution_model"]["default_slippage_ticks"],
        "integer_contracts": True,
        "roll_proxy_dates": [d.strftime("%Y-%m-%d") for d in sorted(roll_dates)],
        "final_model_vs_baselines": comparisons,
        "summary": summary_rows,
        "limitations": cfg["known_limitations"],
        "warning": "Do not retune from these TEST simulator results.",
    }
    report_path = metrics / "full_market_simulator_frozen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(10, 5))
    plot = summary[summary["strategy"].isin([
        "delta_1to1",
        "real_train_static_minvar",
        "causal_rolling60_minvar",
        "synthetic_pretrained_lstm",
        "real_finetuned_lstm",
    ])]
    x = np.arange(len(plot))
    plt.bar(x, plot["net_cvar_loss_5pct_vnd"])
    plt.xticks(x, plot["strategy"], rotation=30, ha="right")
    plt.ylabel("Net 60-day CVaR loss (VND)")
    plt.title("Full Integer-Contract VN30F1M Market Simulator — Frozen OOS")
    plt.tight_layout()
    fig_path = figures / "full_market_simulator_frozen_cvar.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print("=" * 132)
    print("FULL VN30F1M MARKET SIMULATOR COMPLETE — FROZEN MODELS, NO RETUNING")
    print("=" * 132)
    print(f"Test period                 : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Episodes                    : {len(starts):,} overlapping 60-day paths")
    print(f"Initial VN30 portfolio      : {cfg['portfolio']['initial_vn30_portfolio_notional_vnd']:,.0f} VND")
    print(f"Margin assumption           : {cfg['personal_income_tax']['initial_margin_rate_assumption']:.0%}")
    print(f"Broker commission           : {cfg['official_service_fees']['broker_commission_vnd_per_contract_per_match']:,.0f} VND/contract/match")
    print(f"Default slippage            : {cfg['execution_model']['default_slippage_ticks']:.1f} tick")
    print("Contract sizing             : integer, nearest half-up")
    print("Roll                         : calendar expiry proxy, close old + open next")
    print()
    cols = [
        "strategy",
        "net_mean_pnl_vnd",
        "net_q05_pnl_vnd",
        "net_cvar_loss_5pct_vnd",
        "mean_total_cost_vnd",
        "mean_pit_tax_vnd",
        "mean_integer_contracts",
        "max_contracts_observed",
        "mean_realized_hedge_ratio",
        "max_margin_required_vnd",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("FINAL FROZEN MODEL VS BASELINES — FULL MARKET SIMULATOR")
    for baseline, vals in comparisons.items():
        print(
            f"{baseline:26s} | CVaR reduction={vals['net_cvar_reduction_fraction']:+.2%} | "
            f"mean PnL diff={vals['net_mean_pnl_difference_vnd']:+,.0f} VND"
        )
    print()
    print("IMPORTANT: 100% margin is a project assumption and strongly affects PIT tax.")
    print("TEST is already unsealed; do not retune from these results.")
    print(f"Summary : {summary_path}")
    print(f"Episodes: {episode_path}")
    print(f"Report  : {report_path}")
    print(f"Figure  : {fig_path}")


if __name__ == "__main__":
    main()
