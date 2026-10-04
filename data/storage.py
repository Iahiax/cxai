from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from data.schema import SCHEMA_SQL


class Storage:
    def __init__(self, root: str = "./storage", duckdb_file: str = "market.duckdb"):
        self.root = Path(root)
        for sub in ["market_data", "features", "signals", "trades",
                    "strategies", "models", "research", "logs", "snapshots"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        self.duck_path = self.root / duckdb_file
        self.con = duckdb.connect(str(self.duck_path))
        self._init_schema()

    def _init_schema(self):
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                self.con.execute(s)

    # ---------- candles ----------
    def upsert_candles(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.con.register("_incoming_candles", df)
        self.con.execute("""
            INSERT OR REPLACE INTO candles
            SELECT epic, resolution, ts, open, high, low, close, volume FROM _incoming_candles
        """)
        self.con.unregister("_incoming_candles")

        df = df.copy()
        df["_date"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
        for (epic, res, d), chunk in df.groupby(["epic", "resolution", "_date"]):
            p = self.root / "market_data" / f"epic={epic}" / f"res={res}" / f"date={d}"
            p.mkdir(parents=True, exist_ok=True)
            chunk.drop(columns=["_date"]).to_parquet(p / "part.parquet", index=False)
        return len(df)

    def load_candles(self, epic: str, resolution: str,
                     start: Optional[str] = None,
                     end: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM candles WHERE epic = ? AND resolution = ?"
        params: list = [epic, resolution]
        if start: q += " AND ts >= ?"; params.append(start)
        if end:   q += " AND ts <= ?"; params.append(end)
        q += " ORDER BY ts"
        return self.con.execute(q, params).df()

    # ---------- trades ----------
    def save_trade(self, trade: dict):
        self.con.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, epic, strategy, side, entry_ts, entry_px, exit_ts, exit_px,
             size, pnl, pnl_net, costs, meta)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            trade["trade_id"], trade["epic"], trade["strategy"], trade["side"],
            trade["entry_ts"], trade["entry_px"], trade.get("exit_ts"), trade.get("exit_px"),
            trade["size"], trade.get("pnl"), trade.get("pnl_net"), trade.get("costs"),
            json.dumps(trade.get("meta", {})),
        ])
        p = self.root / "trades" / f"{trade['trade_id']}.json"
        p.write_text(json.dumps(trade, default=str, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    # ---------- proposals (لـ Self-Proposal Engine) ----------
    def add_proposal(self, kind: str, title: str, body: str,
                     meta: dict | None = None) -> int:
        next_id = self.con.execute(
            "SELECT COALESCE(MAX(id),0)+1 FROM proposals"
        ).fetchone()[0]
        self.con.execute("""
            INSERT INTO proposals (id, kind, title, body, status, score, meta)
            VALUES (?,?,?,?,?,?,?)
        """, [next_id, kind, title, body, "pending", None,
              json.dumps(meta or {})])
        return next_id

    def list_proposals(self, status: Optional[str] = None) -> pd.DataFrame:
        if status:
            return self.con.execute(
                "SELECT * FROM proposals WHERE status = ? ORDER BY id", [status]
            ).df()
        return self.con.execute("SELECT * FROM proposals ORDER BY id").df()

    def close(self):
        self.con.close()
