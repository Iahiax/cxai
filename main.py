from __future__ import annotations
import os, threading, time
import pandas as pd
from dotenv import load_dotenv

from core.session import CapitalSession
from core.telegram import TelegramBot, TelegramConfig, register_default_commands
from connectors.capital import CapitalClient
from data.storage import Storage
from engines.backtest import BacktestConfig
from engines.indicators import ema
from engines.risk import RiskEngine, RiskLimits
from engines.proposal import SelfProposalEngine


def make_signal(params: dict):
    def sig(df: pd.DataFrame) -> pd.Series:
        f = ema(df["close"], params["fast"])
        s = ema(df["close"], params["slow"])
        out = pd.Series(0, index=df.index)
        out[f > s] = 1
        out[f < s] = -1
        return out
    return sig


def fetch_df(client, epic="GOLD", resolution="MINUTE_15", bars=2000) -> pd.DataFrame:
    data = client.get_candles(epic, resolution, bars)
    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame(prices)
    df["ts"] = pd.to_datetime(df["snapshotTimeUTC"])
    for k in ["open", "high", "low", "close"]:
        df[k] = df[f"{k}Price"].apply(lambda x: x["bid"] if isinstance(x, dict) else x)
    df["volume"] = df.get("lastTradedVolume", 0)
    return df[["ts", "open", "high", "low", "close", "volume"]]


def main():
    load_dotenv()
    storage = Storage("./storage")

    session = CapitalSession(
        api_key=os.environ["CAPITAL_API_KEY"],
        identifier=os.environ["CAPITAL_EMAIL"],
        password=os.environ["CAPITAL_API_PASSWORD"],
        demo=os.environ.get("CAPITAL_DEMO_OR_LIVE", "DEMO").upper() == "DEMO",
    )
    session.login()
    client = CapitalClient(session)
    df = fetch_df(client)

    cfg = BacktestConfig()
    risk = RiskEngine(storage, RiskLimits())

    tg = TelegramBot(TelegramConfig(
        bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        channel_id=os.environ["TELEGRAM_CHANNEL_ID"],
    ), storage)
    register_default_commands(tg, storage, risk)
    threading.Thread(target=tg.run_forever, daemon=True).start()

    engine = SelfProposalEngine(storage, cfg, telegram=tg)
    proposals = engine.explore(
        df,
        base_signal_fn=lambda d, p: make_signal(p)(d),
        base_params={"fast": 10, "slow": 50},
        n_mutations=5,
    )
    print("Top proposal:", proposals[0] if proposals else None)
    print("Knowledge:", engine.knowledge_summary())

    storage.close()


if __name__ == "__main__":
    main()
