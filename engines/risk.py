from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from data.storage import Storage


@dataclass
class RiskLimits:
    daily_loss_limit_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    max_exposure_x_equity: float = 1.0
    max_leverage: float = 5.0
    max_concurrent_positions: int = 3
    max_correlation: float = 0.7
    max_cvar_95_pct: float = 3.0
    cooldown_minutes_after_loss: int = 30


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    size_multiplier: float = 1.0
    kill_switch: bool = False


class RiskEngine:
    def __init__(self, storage: Storage, limits: RiskLimits):
        self.storage = storage
        self.limits = limits
        self._killed = False
        self._cooldown_until: Optional[datetime] = None

    # ---------- daily PnL ----------
    def today_pnl(self) -> float:
        today = date.today().isoformat()
        row = self.storage.con.execute("""
            SELECT COALESCE(SUM(pnl_net),0) FROM trades
            WHERE CAST(entry_ts AS DATE) = ?::DATE
        """, [today]).fetchone()
        return float(row[0] or 0.0)

    def equity(self) -> float:
        row = self.storage.con.execute("""
            SELECT COALESCE(SUM(pnl_net),0) FROM trades
        """).fetchone()
        # fallback إلى رأس مال افتراضي
        return 100_000.0 + float(row[0] or 0.0)

    def daily_drawdown_pct(self) -> float:
        return abs(self.today_pnl()) / self.equity() * 100

    # ---------- CVaR ----------
    def cvar_95(self, lookback: int = 200) -> float:
        df = self.storage.con.execute("""
            SELECT pnl_net FROM trades ORDER BY entry_ts DESC LIMIT ?
        """, [lookback]).df()
        if df.empty:
            return 0.0
        rets = df["pnl_net"].astype(float)
        var95 = rets.quantile(0.05)
        tail = rets[rets <= var95]
        return float(abs(tail.mean() / self.equity() * 100)) if len(tail) else 0.0

    # ---------- Bayesian Kelly ----------
    def kelly_fraction(self, lookback: int = 200) -> float:
        df = self.storage.con.execute("""
            SELECT pnl_net FROM trades ORDER BY entry_ts DESC LIMIT ?
        """, [lookback]).df()
        if len(df) < 20:
            return 0.0
        wins = df[df["pnl_net"] > 0]["pnl_net"]
        losses = df[df["pnl_net"] < 0]["pnl_net"]
        if wins.empty or losses.empty:
            return 0.0
        p = len(wins) / len(df)
        b = wins.mean() / abs(losses.mean())
        kelly = p - (1 - p) / b
        # shrinkage: نصف Kelly لتفادي overbetting
        return float(max(0.0, min(kelly * 0.5, 0.25)))

    # ---------- correlation ----------
    def correlation_ok(self, candidates: dict[str, pd.Series]) -> bool:
        if len(candidates) < 2:
            return True
        df = pd.DataFrame(candidates).pct_change().dropna()
        if df.empty:
            return True
        c = df.corr().abs().values
        np.fill_diagonal(c, 0)
        return c.max() <= self.limits.max_correlation

    # ---------- main gate ----------
    def pre_trade(self, symbol: str, notional: float,
                  concurrent_positions: int) -> RiskDecision:
        if self._killed:
            return RiskDecision(False, "kill_switch_active")

        if self._cooldown_until and datetime.now() < self._cooldown_until:
            return RiskDecision(False, f"cooldown_until_{self._cooldown_until.isoformat()}")

        if self.daily_drawdown_pct() >= self.limits.daily_loss_limit_pct:
            self.trigger_kill("daily_loss_limit_reached")
            return RiskDecision(False, "daily_loss_limit", kill_switch=True)

        if concurrent_positions >= self.limits.max_concurrent_positions:
            return RiskDecision(False, "max_concurrent_positions")

        exp = notional / max(self.equity(), 1e-9)
        if exp > self.limits.max_exposure_x_equity:
            return RiskDecision(False, "max_exposure_exceeded",
                                size_multiplier=self.limits.max_exposure_x_equity / exp)

        if self.cvar_95() > self.limits.max_cvar_95_pct:
            return RiskDecision(False, "cvar_limit")

        kelly = self.kelly_fraction()
        mult = min(1.0, kelly / 0.05) if kelly > 0 else 0.5
        return RiskDecision(True, "ok", size_multiplier=mult)

    def on_trade_close(self, pnl_net: float):
        if pnl_net < 0:
            self._cooldown_until = datetime.now() + timedelta(
                minutes=self.limits.cooldown_minutes_after_loss
            )
            self._log_event("loss_cooldown", "info", {"pnl": pnl_net})

    def trigger_kill(self, reason: str):
        self._killed = True
        self._log_event("kill_switch", "critical", {"reason": reason})

    def release_kill(self):
        self._killed = False
        self._log_event("kill_release", "info", {})

    def _log_event(self, kind: str, severity: str, detail: dict):
        self.storage.con.execute("""
            INSERT INTO risk_events (id, kind, severity, detail) VALUES
            ((SELECT COALESCE(MAX(id),0)+1 FROM risk_events), ?, ?, ?)
        """, [kind, severity, json.dumps(detail, default=str)])
