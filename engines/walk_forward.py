from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterator
import numpy as np
import pandas as pd

from engines.backtest import BacktestConfig, run_backtest


class PurgedKFold:
    """Purged K-Fold مع Embargo — يمنع تسرّب المعلومات المستقبلية."""

    def __init__(self, n_splits: int = 5, embargo_frac: float = 0.01):
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(n)
        fold = n // self.n_splits
        emb = int(n * self.embargo_frac)
        for k in range(self.n_splits):
            start = k * fold
            end = n if k == self.n_splits - 1 else (k + 1) * fold
            test = idx[start:end]
            train_left = idx[: max(0, start - emb)]
            train_right = idx[min(n, end + emb):]
            train = np.concatenate([train_left, train_right])
            if len(train) and len(test):
                yield train, test


@dataclass
class WFResult:
    fold_metrics: list[dict]
    oos_equity: pd.Series
    summary: dict


def walk_forward(df: pd.DataFrame,
                 signal_fn: Callable[[pd.DataFrame], pd.Series],
                 cfg: BacktestConfig,
                 n_splits: int = 5,
                 embargo_frac: float = 0.01) -> WFResult:
    cv = PurgedKFold(n_splits, embargo_frac)
    fold_metrics, oos_curves = [], []

    for i, (_, test_idx) in enumerate(cv.split(len(df))):
        chunk = df.iloc[test_idx].reset_index(drop=True)
        if len(chunk) < 50:
            continue
        # مهم: signal_fn يستخدم فقط chunk (لا يرى التدريب)
        res = run_backtest(chunk, signal_fn, cfg)
        m = dict(res.metrics)
        m["fold"] = i
        m["start"] = str(chunk["ts"].iloc[0])
        m["end"] = str(chunk["ts"].iloc[-1])
        fold_metrics.append(m)
        oos_curves.append(res.equity_curve)

    oos = pd.concat(oos_curves).sort_index() if oos_curves else pd.Series(dtype=float)
    summary = _summarize(fold_metrics)
    return WFResult(fold_metrics, oos, summary)


def _summarize(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    keys = ["sharpe", "total_return", "max_drawdown", "win_rate", "profit_factor", "trades"]
    out = {}
    for k in keys:
        vals = [m.get(k, 0.0) for m in fold_metrics if m.get(k) is not None]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    out["n_folds"] = len(fold_metrics)
    out["consistency"] = out.get("total_return_mean", 0) / (
        out.get("total_return_std", 1e-9) + 1e-9
    )
    return out


def champion_challenger(champion_fn, challenger_fn,
                        df: pd.DataFrame, cfg: BacktestConfig,
                        n_splits: int = 5) -> dict:
    """يقارن استراتيجيتين. الفائز = consistency أعلى مع Sharpe أعلى."""
    ch_wf = walk_forward(df, champion_fn, cfg, n_splits)
    cl_wf = walk_forward(df, challenger_fn, cfg, n_splits)
    ch, cl = ch_wf.summary, cl_wf.summary
    winner = "champion"
    if cl.get("consistency", -1e9) > ch.get("consistency", -1e9) and \
       cl.get("sharpe_mean", -1e9) >= ch.get("sharpe_mean", -1e9) * 0.9:
        winner = "challenger"
    return {
        "winner": winner,
        "champion_summary": ch,
        "challenger_summary": cl,
        "delta_sharpe": cl.get("sharpe_mean", 0) - ch.get("sharpe_mean", 0),
    }
