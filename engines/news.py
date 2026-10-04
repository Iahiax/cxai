from __future__ import annotations
import json, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests

from data.storage import Storage


# أضف لـ schema
NEWS_SQL = """
CREATE TABLE IF NOT EXISTS news_events (
    id VARCHAR PRIMARY KEY, ts TIMESTAMP, source VARCHAR,
    title VARCHAR, body VARCHAR, sentiment DOUBLE, impact VARCHAR,
    symbols VARCHAR, meta JSON
);
CREATE TABLE IF NOT EXISTS macro_series (
    series_id VARCHAR, ts TIMESTAMP, value DOUBLE,
    PRIMARY KEY (series_id, ts)
);
"""


@dataclass
class NewsKeys:
    fred: Optional[str] = None
    fmp: Optional[str] = None
    finnhub: Optional[str] = None


class NewsEngine:
    def __init__(self, storage: Storage, keys: NewsKeys):
        self.s = storage
        self.keys = keys
        for stmt in NEWS_SQL.strip().split(";"):
            if stmt.strip():
                self.s.con.execute(stmt)

    # ---------- FRED ----------
    def fred_series(self, series_id: str, limit: int = 500) -> dict:
        if not self.keys.fred:
            raise RuntimeError("FRED key missing")
        r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": series_id,
            "api_key": self.keys.fred,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }, timeout=20)
        r.raise_for_status()
        data = r.json().get("observations", [])
        rows = []
        for o in data:
            if o["value"] in (".", ""):
                continue
            rows.append({"series_id": series_id,
                         "ts": o["date"],
                         "value": float(o["value"])})
        if rows:
            self.s.con.executemany(
                "INSERT OR REPLACE INTO macro_series VALUES (?,?,?)",
                [(r["series_id"], r["ts"], r["value"]) for r in rows])
        return {"inserted": len(rows)}

    # ---------- FMP (economic calendar) ----------
    def fmp_calendar(self, days_ahead: int = 7) -> dict:
        if not self.keys.fmp:
            raise RuntimeError("FMP key missing")
        frm = datetime.now(timezone.utc).date().isoformat()
        to = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).date().isoformat()
        r = requests.get("https://financialmodelingprep.com/api/v3/economic_calendar",
                         params={"from": frm, "to": to, "apikey": self.keys.fmp},
                         timeout=20)
        r.raise_for_status()
        events = r.json() or []
        n = 0
        for e in events:
            eid = f"fmp:{e.get('date')}:{e.get('event','')[:40]}"
            try:
                self.s.con.execute("""
                    INSERT OR REPLACE INTO news_events
                    (id, ts, source, title, body, sentiment, impact, symbols, meta)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, [
                    eid, e.get("date"), "fmp", e.get("event", ""),
                    json.dumps({k: e.get(k) for k in
                                ["actual", "estimate", "previous", "change", "changePercentage", "country"]}),
                    None, self._impact_from_event(e),
                    e.get("currency", ""), json.dumps(e),
                ])
                n += 1
            except Exception:
                pass
        return {"inserted": n}

    @staticmethod
    def _impact_from_event(e: dict) -> str:
        actual = e.get("actual")
        estimate = e.get("estimate")
        if actual is None or estimate is None:
            return "unknown"
        try:
            delta = abs(float(actual) - float(estimate))
            base = abs(float(estimate)) or 1.0
            ratio = delta / base
            if ratio > 0.15: return "high"
            if ratio > 0.05: return "medium"
            return "low"
        except Exception:
            return "unknown"

    # ---------- Finnhub (news + sentiment) ----------
    def finnhub_news(self, symbol: str = "forex",
                     hours_back: int = 24) -> dict:
        if not self.keys.finnhub:
            raise RuntimeError("Finnhub key missing")
        frm = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).date().isoformat()
        to = datetime.now(timezone.utc).date().isoformat()
        r = requests.get("https://finnhub.io/api/v1/news",
                         params={"category": symbol, "token": self.keys.finnhub},
                         timeout=20)
        r.raise_for_status()
        items = r.json() or []
        n = 0
        for it in items:
            ts = datetime.fromtimestamp(it.get("datetime", 0), tz=timezone.utc)
            if ts.date().isoformat() < frm:
                continue
            nid = f"finnhub:{it.get('id')}"
            self.s.con.execute("""
                INSERT OR REPLACE INTO news_events
                (id, ts, source, title, body, sentiment, impact, symbols, meta)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, [
                nid, ts.isoformat(), "finnhub",
                it.get("headline", ""), it.get("summary", ""),
                None, "unknown", it.get("related", ""),
                json.dumps(it),
            ])
            n += 1
        return {"inserted": n}

    # ---------- simple sentiment ----------
    def score_sentiment(self, text: str) -> float:
        """بسيط جدًا: عدّ كلمات اتجاهية. استبدله بنموذج HF لاحقًا."""
        pos = {"up", "bull", "beat", "strong", "rise", "gain", "surge", "rally",
               "أعلى", "صعود", "قوي", "ارتفاع", "مكسب"}
        neg = {"down", "bear", "miss", "weak", "fall", "loss", "drop_t", "crash",
               "أدنى", "هبوط", "ضعيف", "انخفاض", "خسارة"}
        t = text.lower()
        p = sum(1 for w in pos if w in t)
        n = sum(1 for w in neg if w in t)
        if p + n == 0:
            return 0.0
        return (p - n) / (p + n)

    # ---------- gate: هل نتداول الآن؟ ----------
    def blackout_window(self, minutes: int = 30) -> bool:
        """يمنع التداول حول أحداث high impact."""
        now = datetime.now(timezone.utc)
        soon = (now + timedelta(minutes=minutes)).isoformat()
        past = (now - timedelta(minutes=minutes)).isoformat()
        row = self.s.con.execute("""
            SELECT COUNT(*) FROM news_events
            WHERE impact = 'high' AND ts BETWEEN ? AND ?
        """, [past, soon]).fetchone()
        return (row[0] or 0) > 0
