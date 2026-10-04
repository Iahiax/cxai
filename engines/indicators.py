from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- أساسيات ----------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift()
    return pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        u = min(upper.iloc[i], upper.iloc[i - 1]) if df["close"].iloc[i - 1] <= upper.iloc[i - 1] else upper.iloc[i]
        l = max(lower.iloc[i], lower.iloc[i - 1]) if df["close"].iloc[i - 1] >= lower.iloc[i - 1] else lower.iloc[i]
        if df["close"].iloc[i] > u:
            dir_.iloc[i] = 1
        elif df["close"].iloc[i] < l:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
            u = min(u, upper.iloc[i]) if dir_.iloc[i] == 1 else u
            l = max(l, lower.iloc[i]) if dir_.iloc[i] == -1 else l
        st.iloc[i] = l if dir_.iloc[i] == 1 else u
    return st, dir_


def vwap(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    if window:
        return pv.rolling(window).sum() / df["volume"].rolling(window).sum()
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


def donchian(df: pd.DataFrame, period: int = 20):
    return df["high"].rolling(period).max(), df["low"].rolling(period).min()


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    m = close.rolling(period).mean()
    s = close.rolling(period).std()
    return m - k * s, m, m + k * s


# ---------- هيكل سوقي ----------
def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    """يعيد (swing_high, swing_low) كأعمدة boolean."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            sh[i] = True
        if lows[i] == min(lows[i - left:i + right + 1]):
            sl[i] = True
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)


def fair_value_gaps(df: pd.DataFrame, min_atr_mult: float = 0.3) -> pd.DataFrame:
    """FVG ثلاثية: يحفظ فقط الفجوات التي حجمها >= min_atr_mult * ATR."""
    a = atr(df)
    bull = df["low"] > df["high"].shift(2)
    bear = df["high"] < df["low"].shift(2)
    size = (df["low"] - df["high"].shift(2)).abs()
    size_bear = (df["low"].shift(2) - df["high"]).abs()
    valid_bull = bull & (size >= min_atr_mult * a)
    valid_bear = bear & (size_bear >= min_atr_mult * a)
    return pd.DataFrame({
        "fvg_bull": valid_bull,
        "fvg_bear": valid_bear,
        "fvg_top": np.where(valid_bull, df["low"], np.where(valid_bear, df["low"].shift(2), np.nan)),
        "fvg_bot": np.where(valid_bull, df["high"].shift(2), np.where(valid_bear, df["high"], np.nan)),
    }, index=df.index)


def order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Order block بسيط: آخر شمعة عكسية قبل حركة اندفاعية تكسر آخر swing.
    """
    sh, sl = swing_points(df)
    bull_ob = pd.Series(np.nan, index=df.index)
    bear_ob = pd.Series(np.nan, index=df.index)
    last_sh = last_sl = np.nan
    for i in range(len(df)):
        if sh.iloc[i]:
            last_sh = df["high"].iloc[i]
        if sl.iloc[i]:
            last_sl = df["low"].iloc[i]
        if not np.isnan(last_sh) and df["close"].iloc[i] > last_sh:
            # ابحث عن آخر شمعة هابطة قبل الاختراق
            for j in range(i - 1, max(i - lookback, 0), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    bull_ob.iloc[i] = df["low"].iloc[j]
                    break
            last_sh = np.nan
        if not np.isnan(last_sl) and df["close"].iloc[i] < last_sl:
            for j in range(i - 1, max(i - lookback, 0), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    bear_ob.iloc[i] = df["high"].iloc[j]
                    break
            last_sl = np.nan
    return pd.DataFrame({"bull_ob": bull_ob, "bear_ob": bear_ob}, index=df.index)


def volume_profile(df: pd.DataFrame, bins: int = 30, window: int = 200) -> pd.DataFrame:
    """POC + Value Area لكل نافذة متدحرجة."""
    def _one(x: pd.DataFrame):
        if len(x) < 10:
            return pd.Series({"poc": np.nan, "va_hi": np.nan, "va_lo": np.nan})
        hist, edges = np.histogram(x["close"], bins=bins, weights=x["volume"])
        poc_idx = int(np.argmax(hist))
        poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        total = hist.sum()
        target = total * 0.7
        lo = hi = poc_idx
        acc = hist[poc_idx]
        while acc < target and (lo > 0 or hi < len(hist) - 1):
            left = hist[lo - 1] if lo > 0 else -1
            right = hist[hi + 1] if hi < len(hist) - 1 else -1
            if right >= left:
                hi += 1; acc += hist[hi]
            else:
                lo -= 1; acc += hist[lo]
        return pd.Series({"poc": poc, "va_hi": edges[hi + 1], "va_lo": edges[lo]})

    return df.rolling(window).apply(lambda _: 0, raw=False).iloc[:, 0:0].join(
        df.rolling(window).apply(lambda s: 0, raw=False).to_frame().apply(lambda _: None)
    ).combine_first(
        df.set_index(pd.RangeIndex(len(df))).rolling(window).apply(
            lambda w: np.nan, raw=False
        )
    ) * 0  # عنصر نائب
