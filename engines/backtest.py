from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_bps: float = 0.0
    latency_bars: int = 1
    risk_per_trade: float = 0.01
    allow_short: bool = True
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass
class Trade:
    trade_id: str
    side: str
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: pd.Timestamp
    exit_px: float
    size: float
    pnl: float
    pnl_net: float
    costs: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict = field(default_factory=dict)


def _side_cost(cfg: BacktestConfig, px: float) -> float:
    return px * (cfg.spread_bps / 2 + cfg.slippage_bps + cfg.commission_bps) / 10_000


def run_backtest(candles: pd.DataFrame,
                 signal_fn: Callable[[pd.DataFrame], pd.Series],
                 cfg: BacktestConfig) -> BacktestResult:
    df = candles.copy().reset_index(drop=True)
    raw = signal_fn(df).fillna(0).astype(int).clip(-1, 1)
    signal = raw.shift(cfg.latency_bars).fillna(0).astype(int)
    if not cfg.allow_short:
        signal = signal.clip(lower=0)

    equity = cfg.initial_capital
    curve, trades = [], []
    position, entry_px, entry_ts, entry_size, entry_cost = 0, None, None, 0.0, 0.0

    for i, row in df.iterrows():
        ts = pd.to_datetime(row["ts"])
        px = float(row["close"])
        desired = int(signal.iloc[i])

        if position != 0 and entry_px is not None:
            hit = None
            if position == 1:
                if cfg.stop_loss_pct and row["low"] <= entry_px * (1 - cfg.stop_loss_pct):
                    hit = entry_px * (1 - cfg.stop_loss_pct)
                elif cfg.take_profit_pct and row["high"] >= entry_px * (1 + cfg.take_profit_pct):
                    hit = entry_px * (1 + cfg.take_profit_pct)
            else:
                if cfg.stop_loss_pct and row["high"] >= entry_px * (1 + cfg.stop_loss_pct):
                    hit = entry_px * (1 + cfg.stop_loss_pct)
                elif cfg.take_profit_pct and row["low"] <= entry_px * (1 - cfg.take_profit_pct):
                    hit = entry_px * (1 - cfg.take_profit_pct)
            if hit is not None:
                equity, t = _close(position, entry_px, hit, entry_size,
                                   entry_cost, entry_ts, ts, cfg, equity)
                trades.append(t)
                position, entry_px, entry_size, entry_cost = 0, None, 0.0, 0.0

        if desired != position:
            if position != 0 and entry_px is not None:
                equity, t = _close(position, entry_px, px, entry_size,
                                   entry_cost, entry_ts, ts, cfg, equity)
                trades.append(t)
                position, entry_px, entry_size, entry_cost = 0, None, 0.0, 0.0
            if desired != 0:
                fill = px + (1 if desired > 0 else -1) * _side_cost(cfg, px)
                recent = df.iloc[max(0, i - 20): i + 1]
                rng = float((recent["high"] - recent["low"]).mean() or px * 0.01)
                rng = max(rng, px * 0.001)
                size = (equity * cfg.risk_per_trade) / rng
                size = min(size, equity * 10 / fill)
                position = desired
                entry_px, entry_ts, entry_size = fill, ts, size
                entry_cost = _side_cost(cfg, fill) * size

        if position != 0 and entry_px is not None:
            mtm = (px - entry_px) * position * entry_size
            curve.append({"ts": ts, "equity": equity + mtm})
        else:
            curve.append({"ts": ts, "equity": equity})

    eq = pd.DataFrame(curve).set_index("ts")["equity"]
    return BacktestResult(eq, trades, _metrics(eq, trades, cfg.initial_capital))


def _close(position, entry_px, exit_px, size, entry_cost,
           entry_ts, exit_ts, cfg, equity):
    costs = entry_cost + _side_cost(cfg, exit_px) * size
    pnl = (exit_px - entry_px) * position * size
    equity += pnl - costs
    t = Trade(
        trade_id=f"{entry_ts.isoformat()}-{position}",
        side="LONG" if position > 0 else "SHORT",
        entry_ts=entry_ts, entry_px=entry_px,
        exit_ts=exit_ts, exit_px=exit_px,
        size=size, pnl=pnl, pnl_net=pnl - costs, costs=costs,
    )
    return equity, t


def _metrics(eq: pd.Series, trades: list[Trade], initial: float) -> dict:
    if eq.empty:
        return {}
    ret = eq.pct_change().fillna(0)
    ann = np.sqrt(252 * 24 * 4)
    sharpe = (ret.mean() / ret.std() * ann) if ret.std() > 0 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial - 1),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses else float("inf"),
    }
