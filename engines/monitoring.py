from __future__ import annotations
import json, logging, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from data.storage import Storage


# أضف لـ schema
MON_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    ts TIMESTAMP, name VARCHAR, value DOUBLE
);
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    severity VARCHAR, name VARCHAR, message VARCHAR, meta JSON
);
"""


def setup_logging(log_dir: str = "./storage/logs", level: int = logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(Path(log_dir) / "brain.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("brain")


@dataclass
class AlertRule:
    name: str
    severity: str          # info | warn | critical
    predicate: callable    # (metrics: dict) -> bool
    message: str


class Monitor:
    def __init__(self, storage: Storage, telegram=None):
        self.s = storage
        self.tg = telegram
        self.log = logging.getLogger("monitor")
        for stmt in MON_SQL.strip().split(";"):
            if stmt.strip():
                self.s.con.execute(stmt)
        self.rules: list[AlertRule] = []

    # ---------- metric recording ----------
    def record(self, name: str, value: float):
        self.s.con.execute(
            "INSERT INTO metrics VALUES (?,?,?)",
            [datetime.now(timezone.utc).isoformat(), name, float(value)]
        )

    def snapshot(self) -> dict:
        eq = self.s.con.execute(
            "SELECT COALESCE(SUM(pnl_net),0) FROM trades"
        ).fetchone()[0] or 0.0
        open_trades = self.s.con.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_ts IS NULL"
        ).fetchone()[0] or 0
        win_rate = self.s.con.execute("""
            SELECT AVG(CASE WHEN pnl_net > 0 THEN 1.0 ELSE 0.0 END)
            FROM trades WHERE exit_ts IS NOT NULL
        """).fetchone()[0] or 0.0
        return {
            "total_pnl": float(eq),
            "open_trades": int(open_trades),
            "win_rate": float(win_rate),
        }

    # ---------- alerting ----------
    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def add_default_rules(self, risk_engine=None):
        self.add_rule(AlertRule(
            name="drawdown_high", severity="warn",
            predicate=lambda m: m.get("total_pnl", 0) < -1000,
            message="⚠️ الخسارة التراكمية تجاوزت 1000",
        ))
        if risk_engine:
            self.add_rule(AlertRule(
                name="daily_loss_limit", severity="critical",
                predicate=lambda m: risk_engine.daily_drawdown_pct() >
                                    risk_engine.limits.daily_loss_limit_pct,
                message="🛑 وصلت حدّ الخسارة اليومي",
            ))

    def check(self) -> list[dict]:
        m = self.snapshot()
        self.record("total_pnl", m["total_pnl"])
        self.record("open_trades", m["open_trades"])
        self.record("win_rate", m["win_rate"])

        fired = []
        for rule in self.rules:
            try:
                if rule.predicate(m):
                    self._fire(rule, m)
                    fired.append({"rule": rule.name, "severity": rule.severity})
            except Exception as e:
                self.log.exception("rule %s failed: %s", rule.name, e)
        return fired

    def _fire(self, rule: AlertRule, metrics: dict):
        self.s.con.execute("""
            INSERT INTO alerts (id, severity, name, message, meta)
            VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM alerts), ?, ?, ?, ?)
        """, [rule.severity, rule.name, rule.message,
              json.dumps(metrics, default=str)])
        self.log.warning("[ALERT/%s] %s", rule.severity, rule.message)
        if self.tg:
            icon = {"info": "ℹ️", "warn": "⚠️", "critical": "🛑"}.get(rule.severity, "•")
            try:
                self.tg.send(f"{icon} <b>تنبيه</b> — {rule.name}\n{rule.message}")
            except Exception:
                pass

    # ---------- health report ----------
    def health_report(self) -> dict:
        last_metrics = self.s.con.execute("""
            SELECT name, value FROM metrics
            WHERE ts = (SELECT MAX(ts) FROM metrics)
        """).df().to_dict(orient="records")
        recent_alerts = self.s.con.execute("""
            SELECT severity, name, message, ts FROM alerts
            ORDER BY ts DESC LIMIT 10
        """).df().to_dict(orient="records")
        return {"metrics": last_metrics, "recent_alerts": recent_alerts}
