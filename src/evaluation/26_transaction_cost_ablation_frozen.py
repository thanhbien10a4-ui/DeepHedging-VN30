"""Transaction-cost ablation and liquidity stress for frozen OOS strategies.

Post-freeze descriptive analysis only. The real TEST was already opened in step
23. No model, threshold, feature, objective, or hyperparameter is modified here.

Ablation scenarios
------------------
1. gross_no_cost
2. hnx_only
3. hnx_plus_vsdc
4. hnx_vsdc_roll
5. full_slip_0p5
6. full_slip_1p0
7. full_slip_2p0
8. full_slip_4p0_liquidity_stress

The 4-tick case is a deliberately severe liquidity-stress scenario, not an
empirical claim about typical VN30F1M execution.

Roll mechanics are the same transparent calendar proxy used in step 24 because
the continuous VN30F1M file contains no explicit contract IDs.
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

SCENARIOS = [
    {"name": "gross_no_cost", "hnx": False, "vsdc": False, "roll": False, "slip": 0.0},
    {"name": "hnx_only", "hnx": True, "vsdc": False, "roll": False, "slip": 0.0},
    {"name": "hnx_plus_vsdc", "hnx": True, "vsdc": True, "roll": False, "slip": 0.0},
    {"name": "hnx_vsdc_roll", "hnx": True, "vsdc": True, "roll": True, "slip": 0.0},
    {"name": "full_slip_0p5", "hnx": True, "vsdc": True, "roll": True, "slip": 0.5},
    {"name": "full_slip_1p0", "hnx": True, "vsdc": True, "roll": True, "slip": 1.0},
    {"name": "full_slip_2p0", "hnx": True, "vsdc": True, "roll": True, "slip": 2.0},
    {"name": "full_slip_4p0_liquidity_stress", "hnx": True, "vsdc": True, "roll": True, "slip": 4.0},
]


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
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _, _ in dl:
            chunks.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def minvar_ratio(sr: np.ndarray, fr: np.ndarray) -> float:
    return float(np.cov(sr, fr, ddof=1)[0, 1] / np.var(fr, ddof=1))


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
    day_h = np.empty(len(test), dtype=float)
    for j in range(len(test)):
        hist = combined.iloc[max(0, first_test + j - 60) : first_test + j][["sr", "fr"]].dropna()
        day_h[j] = minvar_ratio(hist["sr"].to_numpy(), hist["fr"].to_numpy())
    out = np.empty((len(starts), EPISODE_STEPS), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = day_h[start : start + EPISODE_STEPS]
    return out


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
    losses = -np.asarray(pnl, dtype=float).reshape(-1)
    k = max(1, int(math.ceil(TAIL_FRACTION * len(losses))))
    idx = np.argpartition(losses, len(losses) - k)[len(losses) - k :]
    return float(np.mean(losses[idx]))


def price_change_episodes(test: pd.DataFrame, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = test["spot_close"].to_numpy(float)
    f = test["fut_close"].to_numpy(float)
    ds = np.empty((len(starts), EPISODE_STEPS), dtype=float)
    df = np.empty_like(ds)
    for i, start in enumerate(starts):
        ds[i] = np.diff(s[start : start + EPISODE_STEPS + 1])
        df[i] = np.diff(f[start : start + EPISODE_STEPS + 1])
    return ds, df


def episode_cost(
    h: np.ndarray,
    episode_dates: pd.DatetimeIndex,
    scenario: dict,
    config: dict,
    roll_dates: set[pd.Timestamp],
) -> tuple[float, dict[str, float]]:
    fees = config["official_service_fees"]
    multiplier = float(config["market"]["contract_multiplier_vnd_per_index_point"])
    tick_size = float(config["market"]["tick_size_index_points"])
    hnx_fee = float(fees["hnx_index_futures_trading_fee_vnd_per_contract_per_match"])
    vsdc_pre = float(fees["vsdc_pre_krx_position_management_fee_vnd_per_contract_account_day"])
    vsdc_post = float(fees["vsdc_post_krx_clearing_fee_vnd_per_contract_offset"])
    krx = pd.Timestamp(fees["krx_go_live_date"])

    h = np.asarray(h, dtype=float)
    d = pd.DatetimeIndex(episode_dates).normalize()

    base_qty = np.zeros(EPISODE_STEPS + 1, dtype=float)
    base_qty[0] = abs(h[0])
    base_qty[1:EPISODE_STEPS] = np.abs(h[1:] - h[:-1])
    base_qty[-1] = abs(h[-1])

    roll_qty = np.zeros_like(base_qty)
    if scenario["roll"]:
        for t in range(EPISODE_STEPS):
            if pd.Timestamp(d[t]) in roll_dates:
                roll_qty[t] = 2.0 * abs(h[t])

    matched = base_qty + roll_qty
    hnx = float(np.sum(matched) * hnx_fee) if scenario["hnx"] else 0.0

    vsdc_position = 0.0
    vsdc_clearing = 0.0
    if scenario["vsdc"]:
        hold_dates = d[:EPISODE_STEPS]
        vsdc_position = float(np.sum(np.abs(h[hold_dates < krx])) * vsdc_pre)
        vsdc_clearing = float(np.sum(matched[d >= krx]) * vsdc_post)

    slip_per_contract = float(scenario["slip"]) * tick_size * multiplier
    slippage = float(np.sum(matched) * slip_per_contract)
    total = hnx + vsdc_position + vsdc_clearing + slippage
    return total, {
        "hnx": hnx,
        "vsdc_position": vsdc_position,
        "vsdc_clearing": vsdc_clearing,
        "slippage": slippage,
        "matched_qty": float(np.sum(matched)),
        "roll_qty": float(np.sum(roll_qty)),
    }


def evaluate(
    strategy: str,
    h: np.ndarray,
    scenario: dict,
    starts: np.ndarray,
    test: pd.DataFrame,
    ds: np.ndarray,
    df: np.ndarray,
    config: dict,
    roll_dates: set[pd.Timestamp],
) -> dict:
    multiplier = float(config["market"]["contract_multiplier_vnd_per_index_point"])
    gross = multiplier * np.sum(ds - h.astype(float) * df, axis=1)
    dates = pd.DatetimeIndex(test["time"])
    costs = np.empty(len(starts), dtype=float)
    matched = np.empty(len(starts), dtype=float)
    roll = np.empty(len(starts), dtype=float)
    for i, start in enumerate(starts):
        cost, comp = episode_cost(
            h[i], dates[start : start + EPISODE_STEPS + 1], scenario, config, roll_dates
        )
        costs[i] = cost
        matched[i] = comp["matched_qty"]
        roll[i] = comp["roll_qty"]
    net = gross - costs
    return {
        "scenario": scenario["name"],
        "strategy": strategy,
        "slippage_ticks": float(scenario["slip"]),
        "gross_mean_vnd": float(np.mean(gross)),
        "gross_cvar_loss_5pct_vnd": cvar_loss(gross),
        "net_mean_vnd": float(np.mean(net)),
        "net_std_vnd": float(np.std(net, ddof=1)),
        "net_q05_vnd": float(np.quantile(net, 0.05)),
        "net_cvar_loss_5pct_vnd": cvar_loss(net),
        "mean_cost_vnd": float(np.mean(costs)),
        "mean_matched_contract_equivalent": float(np.mean(matched)),
        "mean_roll_qty_proxy": float(np.mean(roll)),
        "hedge_ratio_mean": float(np.mean(h)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with open(root / "configs" / "vn30f1m_market_friction.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    train_sr = np.diff(np.log(train["spot_close"].to_numpy(float)))
    train_fr = np.diff(np.log(train["fut_close"].to_numpy(float)))
    static_h = minvar_ratio(train_sr, train_fr)

    base_shape = (len(starts), EPISODE_STEPS)
    h_zero = np.zeros(base_shape, dtype=np.float32)
    h_delta = np.ones(base_shape, dtype=np.float32)
    h_static = np.full(base_shape, static_h, dtype=np.float32)
    h_roll = build_causal_rolling60(train, validation, test, starts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_pre = predict(load_model(root / "results" / "models" / "lstm_cvar_turnover_best.pt", device), x_test, device)
    h_final = predict(load_model(root / "results" / "models" / "lstm_cvar_turnover_real_finetuned_best.pt", device), x_test, device)

    strategies = {
        "unhedged": h_zero,
        "delta_1to1": h_delta,
        "real_train_static_minvar": h_static,
        "causal_rolling60_minvar": h_roll,
        "synthetic_pretrained_lstm": h_pre,
        "real_finetuned_lstm": h_final,
    }

    ds, df = price_change_episodes(test, starts)
    roll_dates = observed_roll_dates(pd.DatetimeIndex(test["time"]))

    rows: list[dict] = []
    for scenario in SCENARIOS:
        for strategy, h in strategies.items():
            rows.append(evaluate(strategy, h, scenario, starts, test, ds, df, config, roll_dates))

    result = pd.DataFrame(rows)
    tables = root / "results" / "tables"
    metrics = root / "results" / "metrics"
    figures = root / "results" / "figures"
    for d in (tables, metrics, figures):
        d.mkdir(parents=True, exist_ok=True)

    table_path = tables / "transaction_cost_ablation_frozen.csv"
    result.to_csv(table_path, index=False)

    final_rows = result[result["strategy"] == "real_finetuned_lstm"].copy()
    gross_cvar = float(final_rows[final_rows["scenario"] == "gross_no_cost"].iloc[0]["net_cvar_loss_5pct_vnd"])
    final_rows["cvar_degradation_vs_gross_fraction"] = final_rows["net_cvar_loss_5pct_vnd"] / gross_cvar - 1.0

    vs_delta: dict[str, dict] = {}
    for scenario in SCENARIOS:
        name = scenario["name"]
        sub = result[result["scenario"] == name]
        frow = sub[sub["strategy"] == "real_finetuned_lstm"].iloc[0]
        drow = sub[sub["strategy"] == "delta_1to1"].iloc[0]
        vs_delta[name] = {
            "final_net_cvar_vnd": float(frow["net_cvar_loss_5pct_vnd"]),
            "delta_net_cvar_vnd": float(drow["net_cvar_loss_5pct_vnd"]),
            "final_cvar_reduction_vs_delta_fraction": float(
                1.0 - frow["net_cvar_loss_5pct_vnd"] / drow["net_cvar_loss_5pct_vnd"]
            ),
            "final_mean_cost_vnd": float(frow["mean_cost_vnd"]),
        }

    report = {
        "status": "POST_FREEZE_TRANSACTION_COST_ABLATION_ONLY",
        "model_retuned": False,
        "real_test_already_unsealed": True,
        "scenarios": SCENARIOS,
        "final_model_vs_delta": vs_delta,
        "final_model_cvar_degradation_vs_gross": final_rows[
            ["scenario", "cvar_degradation_vs_gross_fraction"]
        ].to_dict(orient="records"),
        "excluded": [
            "personal income tax",
            "broker commission",
            "margin-asset management fee",
        ],
        "warning": (
            "The 4-tick liquidity stress is a stress scenario, not an empirical execution-cost estimate. "
            "Do not retune the model from these TEST results."
        ),
    }
    report_path = metrics / "transaction_cost_ablation_frozen_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    fig = plt.figure(figsize=(11, 5))
    x = np.arange(len(final_rows))
    plt.plot(x, final_rows["net_cvar_loss_5pct_vnd"], marker="o", label="real_finetuned_lstm")
    delta_rows = result[result["strategy"] == "delta_1to1"]
    delta_rows = delta_rows.set_index("scenario").loc[final_rows["scenario"]].reset_index()
    plt.plot(x, delta_rows["net_cvar_loss_5pct_vnd"], marker="o", label="delta_1to1")
    plt.xticks(x, final_rows["scenario"], rotation=35, ha="right")
    plt.ylabel("Net 60-day CVaR loss (VND)")
    plt.title("Frozen OOS Transaction-Cost Ablation and Liquidity Stress")
    plt.legend()
    plt.tight_layout()
    fig_path = figures / "transaction_cost_ablation_frozen.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print("=" * 132)
    print("TRANSACTION-COST ABLATION COMPLETE — FROZEN MODELS, NO RETUNING")
    print("=" * 132)
    print(f"Test period   : {test['time'].min().date()} -> {test['time'].max().date()}")
    print(f"Episodes      : {len(starts):,} overlapping 60-day paths")
    print(f"Device        : {device}")
    print("4-tick scenario is a deliberate liquidity stress, not a typical-slippage estimate.")
    print()

    display = result[
        [
            "scenario",
            "strategy",
            "net_mean_vnd",
            "net_q05_vnd",
            "net_cvar_loss_5pct_vnd",
            "mean_cost_vnd",
            "mean_matched_contract_equivalent",
            "hedge_ratio_mean",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("FINAL FROZEN MODEL — COST ABLATION SUMMARY")
    for _, row in final_rows.iterrows():
        print(
            f"{row['scenario']:32s} | CVaR={row['net_cvar_loss_5pct_vnd']:,.0f} VND | "
            f"degradation vs gross={row['cvar_degradation_vs_gross_fraction']:+.2%} | "
            f"mean cost={row['mean_cost_vnd']:,.0f} VND"
        )

    print()
    print("FINAL FROZEN MODEL VS DELTA 1:1")
    for name, vals in vs_delta.items():
        print(
            f"{name:32s} | CVaR reduction={vals['final_cvar_reduction_vs_delta_fraction']:+.2%} | "
            f"final mean cost={vals['final_mean_cost_vnd']:,.0f} VND"
        )

    print()
    print("EXCLUDED: PIT tax, broker commission, margin-asset management fee.")
    print("IMPORTANT: post-freeze TEST analysis only; DO NOT RETUNE from these results.")
    print(f"Table  : {table_path}")
    print(f"Report : {report_path}")
    print(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()
