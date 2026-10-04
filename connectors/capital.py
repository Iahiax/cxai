from __future__ import annotations
from typing import Any, Optional
from core.session import CapitalSession


class CapitalClient:
    def __init__(self, session: CapitalSession):
        self.session = session

    def search_market(self, term: str) -> dict:
        return self.session.request(
            "GET", "/api/v1/markets", params={"searchTerm": term}
        ).json()

    def get_candles(self, epic: str, resolution: str = "MINUTE_15",
                    max_bars: int = 1000,
                    from_ts: Optional[str] = None,
                    to_ts: Optional[str] = None) -> dict:
        params: dict[str, Any] = {"resolution": resolution, "max": max_bars}
        if from_ts: params["from"] = from_ts
        if to_ts:   params["to"] = to_ts
        return self.session.request("GET", f"/api/v1/prices/{epic}",
                                    params=params).json()

    def account(self) -> dict:
        return self.session.request("GET", "/api/v1/accounts").json()

    def positions(self) -> dict:
        return self.session.request("GET", "/api/v1/positions").json()

    def place_order(self, **payload) -> dict:
        return self.session.request("POST", "/api/v1/positions",
                                    json=payload).json()
