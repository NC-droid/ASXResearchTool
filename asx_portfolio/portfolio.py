"""Paper portfolio — allocate starting capital across daily picks and track it."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "portfolio_state.json"


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    entry_date: str
    name: str = ""
    sector: str = ""


@dataclass
class PortfolioState:
    initial_capital: float
    cash: float
    created_at: str
    last_rebalance: str
    rebalance_interval_days: int = 30
    transaction_cost: float = 0.0
    positions: list[Position] = field(default_factory=list)
    value_history: list[dict[str, Any]] = field(default_factory=list)
    rebalance_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "PortfolioState | None":
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        positions = [Position(**p) for p in raw.get("positions", [])]
        return cls(
            initial_capital=raw["initial_capital"],
            cash=raw["cash"],
            created_at=raw["created_at"],
            last_rebalance=raw["last_rebalance"],
            rebalance_interval_days=raw.get("rebalance_interval_days", 30),
            transaction_cost=raw.get("transaction_cost", 0.0),
            positions=positions,
            value_history=raw.get("value_history", []),
            rebalance_history=raw.get("rebalance_history", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _last_close_price(stock_data) -> float:
    if stock_data is None or stock_data.history is None or stock_data.history.empty:
        return float("nan")
    return float(stock_data.history["Close"].dropna().iloc[-1])


def initialize(
    picks_df: pd.DataFrame, fetched: dict, state_path: Path,
    initial_capital: float = 10_000.0, rebalance_interval_days: int = 30, transaction_cost: float = 0.0,
) -> PortfolioState:
    if picks_df is None or picks_df.empty:
        raise ValueError("picks_df is empty; cannot initialize portfolio")
    n = len(picks_df)
    per_position = initial_capital / n
    today = _today_iso()
    positions: list[Position] = []
    cash = initial_capital
    for _, row in picks_df.iterrows():
        t = row["ticker"]
        sd = fetched.get(t)
        price = _last_close_price(sd)
        if price != price or price <= 0:
            continue
        deployable = per_position - transaction_cost
        shares = deployable / price
        cost = shares * price + transaction_cost
        cash -= cost
        positions.append(Position(
            ticker=t, shares=shares, avg_cost=price, entry_date=today,
            name=row.get("name", ""), sector=row.get("sector", ""),
        ))
        log.info("Buy %s: %.4f shares @ %.2f = $%.2f", t, shares, price, shares * price)
    state = PortfolioState(
        initial_capital=initial_capital, cash=round(cash, 4),
        created_at=today, last_rebalance=today,
        rebalance_interval_days=rebalance_interval_days, transaction_cost=transaction_cost,
        positions=positions,
    )
    state.rebalance_history.append({
        "date": today, "picks": [p.ticker for p in positions],
        "cost_basis": round(sum(p.shares * p.avg_cost for p in positions), 2),
        "cash_after": state.cash,
    })
    mark_to_market(state, fetched)
    state.save(state_path)
    return state


def mark_to_market(state: PortfolioState, fetched: dict) -> dict[str, Any]:
    today = _today_iso()
    total = state.cash
    per_pos: list[dict[str, Any]] = []
    for p in state.positions:
        sd = fetched.get(p.ticker)
        price = _last_close_price(sd)
        if price != price:
            price = p.avg_cost
        market_val = p.shares * price
        pnl = market_val - p.shares * p.avg_cost
        pnl_pct = pnl / (p.shares * p.avg_cost) if p.avg_cost > 0 else 0.0
        per_pos.append({
            "ticker": p.ticker, "name": p.name, "sector": p.sector,
            "shares": p.shares, "avg_cost": p.avg_cost, "price": price,
            "market_value": market_val, "weight_pct": 0.0,
            "unrealized_pnl": pnl, "unrealized_pnl_pct": pnl_pct,
        })
        total += market_val
    for r in per_pos:
        r["weight_pct"] = r["market_value"] / total if total > 0 else 0.0
    snapshot = {
        "date": today, "total_value": round(total, 2),
        "cash": round(state.cash, 2),
        "invested_value": round(total - state.cash, 2),
        "total_return_pct": (total - state.initial_capital) / state.initial_capital if state.initial_capital else 0.0,
    }
    if state.value_history and state.value_history[-1]["date"] == today:
        state.value_history[-1] = snapshot
    else:
        state.value_history.append(snapshot)
    return {"snapshot": snapshot, "per_position": per_pos}


def should_rebalance(state: PortfolioState) -> bool:
    last = datetime.strptime(state.last_rebalance, "%Y-%m-%d")
    return (datetime.now() - last).days >= state.rebalance_interval_days


def rebalance(state: PortfolioState, picks_df: pd.DataFrame, fetched: dict) -> None:
    if picks_df is None or picks_df.empty:
        return
    mtm = mark_to_market(state, fetched)
    total_equity = mtm["snapshot"]["total_value"]
    log.info("Rebalancing: total equity = $%.2f", total_equity)
    proceeds = 0.0
    for p in state.positions:
        sd = fetched.get(p.ticker)
        price = _last_close_price(sd)
        if price != price:
            price = p.avg_cost
        proceeds += p.shares * price - state.transaction_cost
    state.cash += proceeds
    state.positions = []
    n = len(picks_df)
    today = _today_iso()
    per_position = (total_equity - n * state.transaction_cost) / n
    new_positions: list[Position] = []
    for _, row in picks_df.iterrows():
        t = row["ticker"]
        sd = fetched.get(t)
        price = _last_close_price(sd)
        if price != price or price <= 0:
            continue
        shares = (per_position - state.transaction_cost) / price
        cost = shares * price + state.transaction_cost
        state.cash -= cost
        new_positions.append(Position(
            ticker=t, shares=shares, avg_cost=price, entry_date=today,
            name=row.get("name", ""), sector=row.get("sector", ""),
        ))
    state.positions = new_positions
    state.last_rebalance = today
    state.rebalance_history.append({
        "date": today, "picks": [p.ticker for p in new_positions],
        "cost_basis": round(sum(p.shares * p.avg_cost for p in new_positions), 2),
        "cash_after": round(state.cash, 2),
    })
    mark_to_market(state, fetched)


def write_portfolio_report(state: PortfolioState, per_position: list[dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    holdings_df = pd.DataFrame(per_position)
    value_df = pd.DataFrame(state.value_history)
    rebalances_df = pd.DataFrame(state.rebalance_history)
    summary_df = _summary_frame(state)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            holdings_df.to_excel(writer, sheet_name="Holdings", index=False)
            value_df.to_excel(writer, sheet_name="Value History", index=False)
            rebalances_df.to_excel(writer, sheet_name="Rebalances", index=False)
            wb = writer.book
            money = wb.add_format({"num_format": "$#,##0.00"})
            pct = wb.add_format({"num_format": "0.00%"})
            for sheet, frame in (("Holdings", holdings_df), ("Value History", value_df)):
                ws = writer.sheets[sheet]
                for i, c in enumerate(frame.columns):
                    fmt = money if c in {"avg_cost","price","market_value","unrealized_pnl","total_value","cash","invested_value"} else (pct if c in {"weight_pct","unrealized_pnl_pct","total_return_pct"} else None)
                    ws.set_column(i, i, 16, fmt)
            if len(value_df) > 1:
                chart = wb.add_chart({"type": "line"})
                chart.add_series({
                    "name": "Portfolio Value",
                    "categories": ["Value History", 1, 0, len(value_df), 0],
                    "values": ["Value History", 1, 1, len(value_df), 1],
                })
                chart.set_title({"name": "Paper Portfolio Value"})
                writer.sheets["Value History"].insert_chart("F2", chart)
    except ImportError:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            holdings_df.to_excel(writer, sheet_name="Holdings", index=False)
            value_df.to_excel(writer, sheet_name="Value History", index=False)
            rebalances_df.to_excel(writer, sheet_name="Rebalances", index=False)
    log.info("Wrote portfolio report: %s", out_path)
    return out_path


def _summary_frame(state: PortfolioState) -> pd.DataFrame:
    latest = state.value_history[-1] if state.value_history else {"total_value": state.initial_capital, "total_return_pct": 0.0}
    days_held = (datetime.now() - datetime.strptime(state.created_at, "%Y-%m-%d")).days or 1
    total_return = latest.get("total_return_pct", 0.0)
    annualized = (1 + total_return) ** (365.0 / days_held) - 1 if total_return > -1 else 0.0
    rows = [
        ("Portfolio created", state.created_at),
        ("Last rebalance", state.last_rebalance),
        ("Rebalance interval (days)", state.rebalance_interval_days),
        ("Initial capital (AUD)", f"${state.initial_capital:,.2f}"),
        ("Current total value (AUD)", f"${latest.get('total_value', 0):,.2f}"),
        ("Total return", f"{total_return*100:.2f}%"),
        ("Days held", days_held),
        ("Annualized return", f"{annualized*100:.2f}%"),
        ("Number of positions", len(state.positions)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
