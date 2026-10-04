from __future__ import annotations
import json, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from engines.backtest import BacktestConfig
from data.storage import Storage


# أضف لـ schema
ML_SQL = """
CREATE TABLE IF NOT EXISTS ml_runs (
    id VARCHAR PRIMARY KEY, ts TIMESTAMP DEFAULT currentimestamp,
    model VARCHAR, params JSON, metrics JSON, artifact VARCHAR
);
"""


@dataclass
class TripleBarrierConfig:
    tp_pct: float = 0.01
    sl_pct: float = 0.01
    max_holding_bars: int = 48
    atr_mult_tp: Optional[float] = None
    atr_mult_sl: Optional[float] = None


def triple_barrier_labels(df: pd.DataFrame,
                          cfg: TripleBarrierConfig,
                          atr_col: Optional[str] = None) -> pd.Series:
    """
    يُرجع سلسلة {-1, 0, +1}: -1=SL، +1=TP، 0=timeout.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    atr_vals = None
    if atr_col and atr_col in df:
        atr_vals = df[atr_col].values

    for i in range(n - 1):
        entry = close[i]
        if atr_vals is not None:
            tp = entry + (cfg.atr_mult_tp or 1.5) * atr_vals[i]
            sl = entry - (cfg.atr_mult_sl or 1.5) * atr_vals[i]
        else:
            tp = entry * (1 + cfg.tp_pct)
            sl = entry * (1 - cfg.sl_pct)
        end = min(n, i + 1 + cfg.max_holding_bars)
        hit = 0
        for j in range(i + 1, end):
            if high[j] >= tp:
                hit = 1; break
            if low[j] <= sl:
                hit = -1; break
        labels[i] = hit
    return pd.Series(labels, index=df.index)


def purged_embargo_split(n: int, k: int = 5, embargo: int = 24):
    """Purged K-Fold متقدّم — يمنع تسرّب البيانات المستقبلية."""
    idx = np.arange(n)
    fold = n // k
    for f in range(k):
        start = f * fold
        end = n if f == k - 1 else (f + 1) * fold
        test = idx[start:end]
        train_mask = np.ones(n, dtype=bool)
        train_mask[start:end] = False
        # embargo حول الـ test
        lo = max(0, start - embargo)
        hi = min(n, end + embargo)
        train_mask[lo:hi] = False
        yield idx[train_mask], test


class MLEngine:
    def __init__(self, storage: Storage, models_dir: str = "./storage/models"):
        self.s = storage
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stmt in ML_SQL.strip().split(";"):
            if stmt.strip():
                self.s.con.execute(stmt)

    # ---------- features ----------
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        from engines.indicators import ema, rsi, atr, adx, macd
        out = pd.DataFrame(index=df.index)
        out["ret1"] = df["close"].pct_change()
        out["ret5"] = df["close"].pct_change(5)
        out["ret20"] = df["close"].pct_change(20)
        out["vol20"] = out["ret1"].rolling(20).std()
        out["ema_ratio"] = ema(df["close"], 10) / ema(df["close"], 50) - 1
        out["rsi"] = rsi(df["close"])
        out["atr_norm"] = atr(df) / df["close"]
        out["adx"] = adx(df)
        m, s, h = macd(df["close"])
        out["macd_hist"] = h / df["close"]
        out["range_norm"] = (df["high"] - df["low"]) / df["close"]
        out["vwap_dev"] = df["close"] / df["close"].rolling(20).mean() - 1
        return out.replace([np.inf, -np.inf], np.nan).dropna()

    # ---------- train ----------
    def train(self, df: pd.DataFrame,
              label_cfg: TripleBarrierConfig,
              params: dict | None = None) -> dict:
        if not HAS_LGB:
            raise RuntimeError("LightGBM غير مثبّت: pip install lightgbm")

        feats = self.build_features(df)
        df_al = df.loc[feats.index]
        labels = triple_barrier_labels(df_al, label_cfg).loc[feats.index]
        # احذف آخر صف (label غير معروف)
        feats = feats.iloc[:-1]
        labels = labels.iloc[:-1]

        params = params or {
            "objective": "multiclass", "num_class": 3,
            "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 50, "verbose": -1,
            "metric": "multi_logloss",
        }

        preds = pd.Series(0, index=feats.index)
        for tr, te in purged_embargo_split(len(feats), k=5, embargo=24):
            ds = lgb.Dataset(feats.iloc[tr], label=(labels.iloc[tr] + 1))
            model = lgb.train(params, ds, num_boost_round=300)
            p = model.predict(feats.iloc[te])
            preds.iloc[te] = np.argmax(p, axis=1) - 1

        acc = float((preds == labels).mean())
        # accuracy baseline: الفئة الغالبة
        base = float((labels == labels.mode()[0]).mean())
        metrics = {"accuracy": acc, "baseline": base, "edge": acc - base,
                   "n_samples": len(feats)}

        run_id = f"lgb_{int(pd.Timestamp.utcnow().timestamp())}"
        art = self.dir / f"{run_id}.pkl"
        with open(art, "wb") as f:
            pickle.dump({"params": params, "label_cfg": label_cfg.__dict__}, f)

        self.s.con.execute("""
            INSERT INTO ml_runs (id, model, params, metrics, artifact)
            VALUES (?,?,?,?,?)
        """, [run_id, "lightgbm", json.dumps(params),
              json.dumps(metrics), str(art)])
        return {"run_id": run_id, "metrics": metrics}

    def latest_runs(self, k: int = 10) -> pd.DataFrame:
        return self.s.con.execute(
            "SELECT * FROM ml_runs ORDER BY ts DESC LIMIT ?", [k]
        ).df()
