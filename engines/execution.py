from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from connectors.capital import CapitalClient
from data.storage import Storage
from engines.risk import RiskEngine


@dataclass
class OrderRequest:
    epic: str
    side: str               # "BUY" | "SELL"
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "default"


@dataclass
class FillReport:
    deal_id: str
    epic: str
    side: str
    size: float
    price: float
    status: str
    ts: str
    raw: dict = field(default_factory=dict)


class ExecutionEngine:
    """
    كل تنفيذ يمر عبر RiskEngine أولًا.
    يدعم: market order + SL/TP + partial fills tracking + kill switch.
    """

    def __init__(self, client: CapitalClient, storage: Storage,
                 risk: RiskEngine, live: bool = False):
        self.client = client
        self.s = storage
        self.risk = risk
        self.live = live  # إذا False → paper trading (تسجيل فقط)

    def _price_from_df(self, df: pd.DataFrame) -> float:
        return float(df["close"].iloc[-1])

    def submit(self, req: OrderRequest, last_df: pd.DataFrame,
               concurrent_positions: int = 0,
               size_multiplier: float = 1.0) -> FillReport:
        px = self._price_from_df(last_df)
        notional = px * req.size

        decision = self.risk.pre_trade(
            symbol=req.epic, notional=notional,
            concurrent_positions=concurrent_positions,
        )
        if not decision.allowed:
            return FillReport(
                deal_id=f"rejected-{uuid.uuid4().hex[:8]}",
                epic=req.epic, side=req.side, size=0.0, price=px,
                status=f"REJECTED:{decision.reason}",
                ts=datetime.now(timezone.utc).isoformat(),
            )

        size = req.size * decision.size_multiplier * size_multiplier

        if not self.live:
            deal_id = f"paper-{uuid.uuid4().hex[:10]}"
            rep = FillReport(deal_id=deal_id, epic=req.epic, side=req.side,
                             size=size, price=px, status="PAPER_FILLED",
                             ts=datetime.now(timezone.utc).isoformat())
        else:
            payload = {
                "epic": req.epic,
                "direction": req.side.upper(),
                "size": size,
                "orderType": "MARKET",
                "guaranteedStop": False,
            }
            if req.stop_loss:
                payload["stopLevel"] = req.stop_loss
            if req.take_profit:
                payload["profitLevel"] = req.take_profit
            data = self.client.place_order(**payload)
            deal_ref = data.get("dealReference")
            # تأكيد الصفقة
            time.sleep(0.7)
            conf = self.client.session.request(
                "GET", f"/api/v1/confirms/{deal_ref}"
            ).json()
            fill_price = float(conf.get("level", px))
            fill_size = float(conf.get("size", size))
            status = conf.get("dealStatus", "UNKNOWN")
            rep = FillReport(
                deal_id=conf.get("dealId", deal_ref),
                epic=req.epic, side=req.side, size=fill_size,
                price=fill_price, status=status,
                ts=datetime.now(timezone.utc).isoformat(),
                raw=conf,
            )

        # سجّل الصفقة كـ open
        self.s.con.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, epic, strategy, side, entry_ts, entry_px, size, meta)
            VALUES (?,?,?,?,?,?,?,?)
        """, [rep.deal_id, rep.epic, req.strategy, rep.side,
              rep.ts, rep.price, rep.size,
              json.dumps({"status": rep.status, "raw": rep.raw}, default=str)])

        return rep

    # ---------- إغلاق صفقة موجودة ----------
    def close_position(self, deal_id: str, exit_px: float, exit_ts: str | None = None):
        row = self.s.con.execute(
            "SELECT epic, side, entry_px, size FROM trades WHERE trade_id = ?",
            [deal_id]).fetchone()
        if not row:
            raise RuntimeError(f"trade {deal_id} not found")
        epic, side, entry_px, size = row
        pnl = (exit_px - entry_px) * size * (1 if side == "BUY" else -1)
        costs = abs(exit_px * size) * 0.0005  # تقدير spread
        pnl_net = pnl - costs

        exit_ts = exit_ts or datetime.now(timezone.utc).isoformat()
        self.s.con.execute("""
            UPDATE trades SET exit_ts = ?, exit_px = ?, pnl = ?,
                              pnl_net = ?, costs = ?
            WHERE trade_id = ?
        """, [exit_ts, exit_px, pnl, pnl_net, costs, deal_id])
        self.risk.on_trade_close(pnl_net)

        # حفظ JSON نسخة
        p = self.s.root / "trades" / f"{deal_id}.json"
        p.write_text(json.dumps({
            "trade_id": deal_id, "epic": epic, "side": side,
            "entry_px": entry_px, "exit_px": exit_px, "size": size,
            "pnl": pnl, "pnl_net": pnl_net, "costs": costs,
            "exit_ts": exit_ts,
        }, default=str, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"deal_id": deal_id, "pnl_net": pnl_net}

    def close_all(self, last_df: pd.DataFrame):
        open_rows = self.s.con.execute("""
            SELECT trade_id FROM trades WHERE exit_ts IS NULL
        """).fetchall()
        px = self._price_from_df(last_df)
        results = [self.close_position(t[0], px) for t in open_rows]
        self.risk.trigger_kill("close_all_invoked")
        return results
