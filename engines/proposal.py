from __future__ import annotations
import json, random
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from data.storage import Storage
from engines.backtest import BacktestConfig
from engines.walk_forward import walk_forward
from core.telegram import TelegramBot


@dataclass
class Proposal:
    kind: str
    title: str
    body: str
    metrics: dict
    score: float
    status: str


class SelfProposalEngine:
    def __init__(self, storage: Storage, cfg: BacktestConfig,
                 telegram: Optional[TelegramBot] = None, seed: int = 42):
        self.storage = storage
        self.cfg = cfg
        self.tg = telegram
        self.rng = random.Random(seed)

    def _gen_param_mutations(self, base_params: dict, n: int = 5) -> list[dict]:
        out = []
        for _ in range(n):
            p = dict(base_params)
            key = self.rng.choice(list(p.keys()))
            base = p[key]
            if isinstance(base, int):
                p[key] = max(2, base + self.rng.choice([-10, -5, -2, 2, 5, 10]))
            elif isinstance(base, float):
                p[key] = max(0.1, base * self.rng.choice([0.7, 0.85, 1.15, 1.3]))
            out.append(p)
        return out

    def _score(self, summary: dict) -> float:
        if not summary:
            return -1e9
        s = summary.get("sharpe_mean", 0)
        c = summary.get("consistency", 0)
        dd = abs(summary.get("max_drawdown_mean", 0))
        return float(s * 0.6 + c * 0.3 - dd * 0.5)

    def evaluate(self, df: pd.DataFrame,
                 signal_fn: Callable[[pd.DataFrame], pd.Series],
                 n_splits: int = 5) -> dict:
        wf = walk_forward(df, signal_fn, self.cfg, n_splits=n_splits)
        return {"summary": wf.summary, "score": self._score(wf.summary)}

    def propose(self, kind: str, title: str, body: str,
                metrics: dict, score: float,
                accept_threshold: float = 1.5) -> Proposal:
        status = "accepted" if score >= accept_threshold else "pending"
        self.storage.add_proposal(kind=kind, title=title, body=body,
                                  meta={"metrics": metrics, "score": score})
        self.storage.con.execute("""
            UPDATE proposals SET status = ?, score = ?, metrics = ?
            WHERE id = (SELECT MAX(id) FROM proposals)
        """, [status, score, json.dumps(metrics, default=str)])

        prop = Proposal(kind, title, body, metrics, score, status)

        if self.tg:
            icon = "✅" if status == "accepted" else "🔬"
            msg = (
                f"{icon} <b>اقتراح جديد</b> — {title}\n"
                f"<b>النوع:</b> {kind}\n"
                f"<b>الحالة:</b> {status}\n"
                f"<b>الدرجة:</b> {score:.3f}\n"
                f"<b>Sharpe:</b> {metrics.get('summary',{}).get('sharpe_mean',0):.2f}\n"
                f"<b>Consistency:</b> {metrics.get('summary',{}).get('consistency',0):.2f}"
            )
            try:
                self.tg.send(msg)
            except Exception as e:
                print(f"[proposal] telegram failed: {e}")
        return prop

    def explore(self, df: pd.DataFrame,
                base_signal_fn: Callable[[pd.DataFrame, dict], pd.Series],
                base_params: dict,
                n_mutations: int = 5) -> list[Proposal]:
        candidates = self._gen_param_mutations(base_params, n_mutations)
        proposals: list[Proposal] = []
        for i, params in enumerate(candidates):
            def sig(d, p=params):
                return base_signal_fn(d, p)
            try:
                res = self.evaluate(df, sig)
            except Exception as e:
                print(f"[proposal] eval failed: {e}")
                continue
            title = f"Param mutation #{i+1}"
            body = json.dumps(params, ensure_ascii=False, indent=2)
            proposals.append(self.propose(
                kind="param_mutation", title=title, body=body,
                metrics=res, score=res["score"],
            ))
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals

    def knowledge_summary(self) -> dict:
        df = self.storage.con.execute("""
            SELECT status, COUNT(*) n, AVG(score) avg_score
            FROM proposals GROUP BY status
        """).df()
        return df.to_dict(orient="records")

    def best_so_far(self, top_k: int = 5) -> pd.DataFrame:
        return self.storage.con.execute("""
            SELECT id, kind, title, status, score FROM proposals
            WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?
        """, [top_k]).df()
