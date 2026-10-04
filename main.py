from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv

from core.session import CapitalSession
from connectors.capital import CapitalClient
from data.storage import Storage
from engines.backtest import BacktestConfig, run_backtest


def fetch_and_store(client, storage, epic="GOLD",
                    resolution="MINUTE_15", max_bars=1000) -> pd.DataFrame:
    data = client.get_candles(epic, resolution, max_bars)
    prices = data.get("prices", [])
    if not prices:
        print("No prices:", data)
        return pd.DataFrame()

    df = pd.DataFrame(prices)
    df["epic"] = epic
    df["resolution"] = resolution
    df["ts"] = pd.to_datetime(df["snapshotTimeUTC"])
    for k in ["open", "high", "low", "close"]:
        df[k] = df[f"{k}Price"].apply(lambda x: x["bid"] if isinstance(x, dict) else x)
    df["volume"] = df.get("lastTradedVolume", 0)

    out = df[["epic", "resolution", "ts", "open", "high", "low", "close", "volume"]]
    storage.upsert_candles(out)
    return out


def ma_signal(df: pd.DataFrame) -> pd.Series:
    fast = df["close"].ewm(span=10, adjust=False).mean()
    slow = df["close"].ewm(span=50, adjust=False).mean()
    s = pd.Series(0, index=df.index)
    s[fast > slow] = 1
    s[fast < slow] = -1
    return s


def main():
    load_dotenv()
    storage = Storage(root="./storage")

    session = CapitalSession(
        api_key=os.environ["CAPITAL_API_KEY"],
        identifier=os.environ["CAPITAL_EMAIL"],
        password=os.environ["CAPITAL_API_PASSWORD"],
        demo=os.environ.get("CAPITAL_DEMO_OR_LIVE", "DEMO").upper() == "DEMO",
    )
    session.login()
    client = CapitalClient(session)
    print("Account:", client.account())

    df = fetch_and_store(client, storage, "GOLD", "MINUTE_15", 1000)
    if not df.empty:
        res = run_backtest(df, ma_signal, BacktestConfig())
        print("Backtest metrics:", res.metrics)

    storage.close()


if __name__ == "__main__":
    main()
