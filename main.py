"""
Capital AI Brain — Main Orchestrator
====================================
نقطة الدخول الموحّدة لكل الطبقات:
    Storage + Schema Bootstrap
    Capital.com Connector (with auto-refresh)
    Risk Engine (kill switch, CVaR, Kelly)
    Telegram Interface (Arabic, allowlist, audit)
    Monitor (metrics + alerts)
    News Engine (FRED / FMP / Finnhub)
    ML Engine (LightGBM + Triple Barrier)
    Self-Proposal Engine (walk-forward)
    Execution Engine (paper/live)
    Async Scheduler

المتطلبات:
    pip install -e .
    cp .env.example .env
    python main.py

المتغيرات الاختيارية في .env:
    SYMBOL=GOLD
    RESOLUTION=MINUTE_15
    BARS=2000
    TELEGRAM_ALLOWED_IDS=12345,67890
    TELEGRAM_ADMIN_IDS=12345
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# ---- Brain modules ----
from core.session import CapitalSession
from core.telegram import TelegramBot, TelegramConfig, register_default_commands
from connectors.capital import CapitalClient
from data.storage import Storage
from engines.backtest import BacktestConfig
from engines.execution import ExecutionEngine, OrderRequest
from engines.indicators import atr, ema
from engines.ml import MLEngine, TripleBarrierConfig
from engines.monitoring import Monitor, setup_logging
from engines.news import NewsEngine, NewsKeys
from engines.proposal import SelfProposalEngine
from engines.risk import RiskEngine, RiskLimits
from engines.walk_forward import walk_forward
from infra.scheduler import AsyncScheduler, install_signal_handlers


# =====================================================================
# Schema bootstrap — يضمن كل الجداول قبل أي استخدام
# =====================================================================
FULL_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_events (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    kind VARCHAR, severity VARCHAR, detail JSON
);
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    actor VARCHAR, action VARCHAR, payload JSON, result VARCHAR
);
CREATE TABLE IF NOT EXISTS news_events (
    id VARCHAR PRIMARY KEY, ts TIMESTAMP, source VARCHAR,
    title VARCHAR, body VARCHAR, sentiment DOUBLE, impact VARCHAR,
    symbols VARCHAR, meta JSON
);
CREATE TABLE IF NOT EXISTS macro_series (
    series_id VARCHAR, ts TIMESTAMP, value DOUBLE,
    PRIMARY KEY (series_id, ts)
);
CREATE TABLE IF NOT EXISTS ml_runs (
    id VARCHAR PRIMARY KEY, ts TIMESTAMP DEFAULT current_timestamp,
    model VARCHAR, params JSON, metrics JSON, artifact VARCHAR
);
CREATE TABLE IF NOT EXISTS metrics (
    ts TIMESTAMP, name VARCHAR, value DOUBLE
);
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT, ts TIMESTAMP DEFAULT current_timestamp,
    severity VARCHAR, name VARCHAR, message VARCHAR, meta JSON
);
"""


def bootstrap_schema(storage: Storage) -> None:
    for stmt in FULL_SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            storage.con.execute(s)


# =====================================================================
# Config
# =====================================================================
def _bool(v: str, default: str = "false") -> bool:
    return os.environ.get(v, default).strip().lower() in ("1", "true", "yes", "on")


def _int_list(v: str) -> list[int]:
    raw = os.environ.get(v, "").strip()
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]


@dataclass
class Config:
    # Capital
    cap_api_key: str
    cap_email: str
    cap_password: str
    cap_demo: bool

    # Telegram (اختياري)
    tg_token: Optional[str] = None
    tg_channel: Optional[str] = None
    tg_allowed: list[int] = field(default_factory=list)
    tg_admin: list[int] = field(default_factory=list)

    # News (اختياري)
    fred_key: Optional[str] = None
    fmp_key: Optional[str] = None
    finnhub_key: Optional[str] = None

    # Ops
    mode: str = "DEMO"             # DEMO | PAPER | LIVE
    risk_profile: str = "SAFE"     # SAFE | NORMAL | AGGRESSIVE
    max_leverage: float = 5.0
    daily_loss_limit: float = 2.0
    max_exposure: float = 1.0
    max_positions: int = 3

    self_improvement: bool = True
    research_enabled: bool = True
    shadow_enabled: bool = True
    backtest_on_update: bool = True

    # Market
    symbol: str = "GOLD"
    resolution: str = "MINUTE_15"
    bars: int = 2000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            cap_api_key=os.environ["CAPITAL_API_KEY"],
            cap_email=os.environ["CAPITAL_EMAIL"],
            cap_password=os.environ["CAPITAL_API_PASSWORD"],
            cap_demo=os.environ.get("CAPITAL_DEMO_OR_LIVE", "DEMO").upper() != "LIVE",

            tg_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            tg_channel=os.environ.get("TELEGRAM_CHANNEL_ID") or None,
            tg_allowed=_int_list("TELEGRAM_ALLOWED_IDS"),
            tg_admin=_int_list("TELEGRAM_ADMIN_IDS"),

            fred_key=os.environ.get("FRED_API_KEY") or None,
            fmp_key=os.environ.get("FMP_API_KEY") or None,
            finnhub_key=os.environ.get("FINNHUB_API_KEY") or None,

            mode=os.environ.get("MODE", "DEMO").upper(),
            risk_profile=os.environ.get("RISK_PROFILE", "SAFE").upper(),
            max_leverage=float(os.environ.get("MAX_LEVERAGE_ALLOWED", "5")),
            daily_loss_limit=float(os.environ.get("DAILY_LOSS_LIMIT", "2.0")),
            max_exposure=float(os.environ.get("MAX_EXPOSURE", "1.0")),
            max_positions=int(os.environ.get("MAX_CONCURRENT_POSITIONS", "3")),

            self_improvement=_bool("SELF_IMPROVEMENT_ENABLED", "true"),
            research_enabled=_bool("RESEARCH_ENABLED", "true"),
            shadow_enabled=_bool("SHADOW_TRADING_ENABLED", "true"),
            backtest_on_update=_bool("BACKTEST_ON_UPDATE", "true"),

            symbol=os.environ.get("SYMBOL", "GOLD"),
            resolution=os.environ.get("RESOLUTION", "MINUTE_15"),
            bars=int(os.environ.get("BARS", "2000")),
        )


RISK_PROFILES: dict[str, RiskLimits] = {
    "SAFE": RiskLimits(
        daily_loss_limit_pct=1.0, max_drawdown_pct=5.0,
        max_exposure_x_equity=0.5, max_leverage=3.0,
        max_concurrent_positions=2, max_cvar_95_pct=1.5,
    ),
    "NORMAL": RiskLimits(
        daily_loss_limit_pct=2.0, max_drawdown_pct=10.0,
        max_exposure_x_equity=1.0, max_leverage=5.0,
        max_concurrent_positions=3, max_cvar_95_pct=3.0,
    ),
    "AGGRESSIVE": RiskLimits(
        daily_loss_limit_pct=4.0, max_drawdown_pct=20.0,
        max_exposure_x_equity=2.0, max_leverage=10.0,
        max_concurrent_positions=5, max_cvar_95_pct=6.0,
    ),
}


# =====================================================================
# Signal factory
# =====================================================================
def make_signal_fn(params: dict):
    """MA crossover factory — params: {fast, slow}."""
    def sig(df: pd.DataFrame) -> pd.Series:
        f = ema(df["close"], int(params["fast"]))
        s = ema(df["close"], int(params["slow"]))
        out = pd.Series(0, index=df.index)
        out[f > s] = 1
        out[f < s] = -1
        return out
    return sig


# =====================================================================
# Brain Context — يجمع كل شيء
# =====================================================================
class BrainContext:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = logging.getLogger("brain")

        # ---- Storage ----
        self.storage = Storage("./storage")
        bootstrap_schema(self.storage)

        # ---- Capital ----
        self.session = CapitalSession(
            api_key=cfg.cap_api_key,
            identifier=cfg.cap_email,
            password=cfg.cap_password,
            demo=cfg.cap_demo,
        )
        self.session.login()
        self.client = CapitalClient(self.session)
        self.log.info("capital session established (demo=%s)", cfg.cap_demo)

        # ---- Risk ----
        limits = RISK_PROFILES.get(cfg.risk_profile, RISK_PROFILES["SAFE"])
        # override from env
        limits.daily_loss_limit_pct = cfg.daily_loss_limit
        limits.max_exposure_x_equity = cfg.max_exposure
        limits.max_concurrent_positions = cfg.max_positions
        limits.max_leverage = cfg.max_leverage
        self.risk = RiskEngine(self.storage, limits)

        # ---- Telegram ----
        self.tg: Optional[TelegramBot] = None
        if cfg.tg_token and cfg.tg_channel:
            self.tg = TelegramBot(
                TelegramConfig(
                    bot_token=cfg.tg_token,
                    channel_id=cfg.tg_channel,
                    allowed_user_ids=cfg.tg_allowed,
                    admin_user_ids=cfg.tg_admin,
                ),
                self.storage,
            )
            register_default_commands(self.tg, self.storage, self.risk)
            self.log.info("telegram bot ready (channel=%s)", cfg.tg_channel)
        else:
            self.log.info("telegram disabled (no token/channel)")

        # ---- Monitor ----
        self.monitor = Monitor(self.storage, telegram=self.tg)
        self.monitor.add_default_rules(self.risk)

        # ---- News ----
        self.news = NewsEngine(
            self.storage,
            NewsKeys(fred=cfg.fred_key, fmp=cfg.fmp_key, finnhub=cfg.finnhub_key),
        )

        # ---- ML ----
        self.ml = MLEngine(self.storage)

        # ---- Backtest config ----
        self.bt_cfg = BacktestConfig()

        # ---- Execution ----
        self.exec_engine = ExecutionEngine(
            client=self.client,
            storage=self.storage,
            risk=self.risk,
            live=(cfg.mode == "LIVE"),
        )
        self.log.info("execution mode: %s", "LIVE" if cfg.mode == "LIVE" else "PAPER")

        # ---- Self-Proposal ----
        self.proposal_engine = SelfProposalEngine(
            storage=self.storage, cfg=self.bt_cfg, telegram=self.tg,
        )

        # ---- State ----
        self._df_cache: Optional[pd.DataFrame] = None
        self._df_cache_ts: float = 0.0
        self._stop = threading.Event()

    # =================================================================
    # Data
    # =================================================================
    def fetch_df(self, cache_seconds: int = 55) -> pd.DataFrame:
        now = time.time()
        if self._df_cache is not None and now - self._df_cache_ts < cache_seconds:
            return self._df_cache

        try:
            data = self.client.get_candles(
                self.cfg.symbol, self.cfg.resolution, self.cfg.bars,
            )
        except Exception as e:
            self.log.exception("get_candles failed: %s", e)
            return self._df_cache if self._df_cache is not None else pd.DataFrame()

        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices)
        df["epic"] = self.cfg.symbol
        df["resolution"] = self.cfg.resolution
        df["ts"] = pd.to_datetime(df["snapshotTimeUTC"])
        for k in ["open", "high", "low", "close"]:
            df[k] = df[f"{k}Price"].apply(
                lambda x: x.get("bid") if isinstance(x, dict) else x
            ).astype(float)
        df["volume"] = pd.to_numeric(
            df.get("lastTradedVolume", 0), errors="coerce"
        ).fillna(0).astype("int64")

        out = df[["epic", "resolution", "ts", "open", "high", "low",
                  "close", "volume"]].sort_values("ts").reset_index(drop=True)

        try:
            self.storage.upsert_candles(out)
        except Exception as e:
            self.log.exception("upsert_candles failed: %s", e)

        self._df_cache = out
        self._df_cache_ts = now
        return out

    # =================================================================
    # Cycles
    # =================================================================
    def heartbeat(self) -> None:
        df = self.fetch_df()
        if df.empty:
            self.log.warning("heartbeat: no data")
            return
        self.log.info(
            "heartbeat: rows=%d last=%s close=%.4f",
            len(df), df["ts"].iloc[-1], float(df["close"].iloc[-1]),
        )
        self.monitor.check()

    def research_cycle(self) -> None:
        if not self.cfg.research_enabled:
            return
        df = self.fetch_df()
        if df.empty or len(df) < 200:
            self.log.warning("research: insufficient data (%d rows)", len(df))
            return

        self.log.info("research: walk-forward on MA(10,50)")
        try:
            wf = walk_forward(
                df,
                make_signal_fn({"fast": 10, "slow": 50}),
                self.bt_cfg,
            )
        except Exception as e:
            self.log.exception("walk_forward failed: %s", e)
            return

        self.log.info("research: summary=%s", wf.summary)
        consistency = wf.summary.get("consistency", 0.0)
        if consistency < 0.5:
            self.log.warning("research: consistency too low (%.2f), skip proposals",
                             consistency)
            return

        try:
            props = self.proposal_engine.explore(
                df,
                base_signal_fn=lambda d, p: make_signal_fn(p)(d),
                base_params={"fast": 10, "slow": 50},
                n_mutations=3,
            )
            if props:
                top = props[0]
                self.log.info("research: top proposal '%s' score=%.3f status=%s",
                              top.title, top.score, top.status)
        except Exception as e:
            self.log.exception("proposal explore failed: %s", e)

    def trading_cycle(self) -> None:
        df = self.fetch_df()
        if df.empty or len(df) < 100:
            return

        if self.risk._killed:
            self.log.info("trading: kill switch active, skip")
            return

        # ---- news blackout ----
        try:
            if self.news.blackout_window(minutes=30):
                self.log.info("trading: news blackout active, skip")
                return
        except Exception:
            pass

        # ---- current signal ----
        sig_fn = make_signal_fn({"fast": 10, "slow": 50})
        s = int(sig_fn(df).iloc[-1])
        if s == 0:
            return

        # ---- positions ----
        open_trades = int(self.storage.con.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_ts IS NULL"
        ).fetchone()[0] or 0)
        if open_trades >= self.risk.limits.max_concurrent_positions:
            self.log.info("trading: at max positions (%d)", open_trades)
            return

        # ---- build order ----
        px = float(df["close"].iloc[-1])
        a = float(atr(df).iloc[-1])
        if a <= 0:
            self.log.warning("trading: ATR<=0, skip")
            return

        side = "BUY" if s > 0 else "SELL"
        size = 0.1 if self.cfg.mode == "LIVE" else 1.0  # حجم ضئيل جدًا في live
        sl = px - 2 * a if side == "BUY" else px + 2 * a
        tp = px + 3 * a if side == "BUY" else px - 3 * a

        req = OrderRequest(
            epic=self.cfg.symbol,
            side=side,
            size=size,
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            strategy="ma_10_50",
        )

        try:
            rep = self.exec_engine.submit(
                req, last_df=df, concurrent_positions=open_trades,
            )
            self.log.info("trade: %s status=%s size=%.4f px=%.4f",
                          rep.deal_id, rep.status, rep.size, rep.price)
        except Exception as e:
            self.log.exception("submit failed: %s", e)

    def news_cycle(self) -> None:
        refreshed = []
        try:
            if self.cfg.fred_key:
                self.news.fred_series("DGS10", limit=60)
                refreshed.append("fred")
            if self.cfg.fmp_key:
                self.news.fmp_calendar(days_ahead=7)
                refreshed.append("fmp")
            if self.cfg.finnhub_key:
                self.news.finnhub_news("forex", hours_back=6)
                refreshed.append("finnhub")
            self.log.info("news: refreshed %s", refreshed or "(no keys)")
        except Exception as e:
            self.log.exception("news cycle failed: %s", e)

    def ml_cycle(self) -> None:
        if not self.cfg.self_improvement:
            return
        df = self.fetch_df()
        if df.empty or len(df) < 500:
            self.log.info("ml: skip (need >=500 rows)")
            return
        try:
            res = self.ml.train(df, TripleBarrierConfig(
                atr_mult_tp=2.0, atr_mult_sl=1.5, max_holding_bars=48,
            ))
            self.log.info("ml: run=%s edge=%.4f",
                          res["run_id"], res["metrics"].get("edge", 0.0))
        except Exception as e:
            self.log.exception("ml train failed: %s", e)

    def health_cycle(self) -> None:
        m = self.monitor.snapshot()
        self.log.info(
            "health: pnl=%.2f open=%d win_rate=%.1f%%",
            m["total_pnl"], m["open_trades"], m["win_rate"] * 100,
        )
        if self.tg:
            try:
                self.tg.send(
                    f"💚 <b>تقرير صحة النظام</b>\n"
                    f"PnL: <b>{m['total_pnl']:.2f}</b>\n"
                    f"صفقات مفتوحة: <b>{m['open_trades']}</b>\n"
                    f"نسبة الربح: <b>{m['win_rate']:.1%}</b>"
                )
            except Exception as e:
                self.log.warning("telegram health send failed: %s", e)

    # =================================================================
    # Shutdown
    # =================================================================
    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.log.warning("shutdown requested")
        try:
            self.storage.close()
        except Exception:
            pass
        self.log.info("storage closed")


# =====================================================================
# Scheduler wiring
# =====================================================================
async def run_brain(ctx: BrainContext) -> None:
    sched = AsyncScheduler()

    sched.add("heartbeat", 60, ctx.heartbeat, run_on_start=True)
    sched.add("trading",   300, ctx.trading_cycle)
    sched.add("news",     1800, ctx.news_cycle, run_on_start=True)
    sched.add("health",   1800, ctx.health_cycle)

    if ctx.cfg.research_enabled:
        sched.add("research", 3600, ctx.research_cycle)
    if ctx.cfg.self_improvement:
        sched.add("ml", 7200, ctx.ml_cycle)

    loop = asyncio.get_running_loop()
    install_signal_handlers(sched, loop)

    ctx.log.info("scheduler jobs: %s",
                 [(j.name, j.every_seconds) for j in sched.jobs])
    await sched.serve()


def start_telegram_poller(ctx: BrainContext) -> None:
    if not ctx.tg:
        return
    def _poll():
        ctx.tg.run_forever(stop_flag=ctx._stop.is_set)
    t = threading.Thread(target=_poll, name="tg-poller", daemon=True)
    t.start()
    ctx.log.info("telegram poller started")


# =====================================================================
# Entrypoint
# =====================================================================
def _validate_env() -> list[str]:
    return [k for k in
            ("CAPITAL_API_KEY", "CAPITAL_EMAIL", "CAPITAL_API_PASSWORD")
            if not os.environ.get(k)]


def main() -> int:
    load_dotenv()
    log = setup_logging()

    log.info("=" * 62)
    log.info("  Capital AI Brain — starting")
    log.info("=" * 62)

    missing = _validate_env()
    if missing:
        log.error("missing required env vars: %s", missing)
        return 2

    try:
        cfg = Config.from_env()
    except Exception as e:
        log.exception("config load failed: %s", e)
        return 2

    log.info("mode=%s | risk=%s | leverage=%.1fx | daily_loss=%.2f%% | symbol=%s@%s",
             cfg.mode, cfg.risk_profile, cfg.max_leverage,
             cfg.daily_loss_limit, cfg.symbol, cfg.resolution)

    try:
        ctx = BrainContext(cfg)
    except Exception as e:
        log.exception("brain init failed: %s", e)
        return 3

    start_telegram_poller(ctx)

    if ctx.tg:
        try:
            ctx.tg.send(
                f"🚀 <b>Capital AI Brain</b> بدأ العمل\n"
                f"الوضع: <b>{cfg.mode}</b>\n"
                f"الملف المخاطر: <b>{cfg.risk_profile}</b>\n"
                f"الرافعة القصوى: <b>{cfg.max_leverage}x</b>\n"
                f"الرمز: <b>{cfg.symbol} @ {cfg.resolution}</b>"
            )
        except Exception as e:
            log.warning("telegram hello failed: %s", e)

    rc = 0
    try:
        asyncio.run(run_brain(ctx))
    except KeyboardInterrupt:
        log.warning("keyboard interrupt — shutting down")
    except Exception as e:
        log.exception("brain crashed: %s", e)
        rc = 4
    finally:
        ctx.shutdown()
        log.info("=" * 62)
        log.info("  Capital AI Brain — stopped (rc=%d)", rc)
        log.info("=" * 62)

    return rc


if __name__ == "__main__":
    sys.exit(main())
