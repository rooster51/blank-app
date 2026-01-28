# ===============================
# Credit Spread Trend Scanner
# Streamlit App (Rate-Limit Safe)
# ===============================

import os
import math
import time
import random
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from scipy.stats import norm
import matplotlib.pyplot as plt

# Optional Finnhub
try:
    import finnhub
except Exception:
    finnhub = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Credit Spread Trend Scanner", layout="wide")
st.title("📈 Trend Scanner → Credit Spread Builder")
st.caption("Educational only. Options trading involves substantial risk.")

TZ = "America/New_York"

# Timeframes (1m REMOVED)
TIMEFRAMES = ["5m", "1h", "1D"]

DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "AMD", "META", "AMZN", "GOOGL"]

# -----------------------------
# Indicators
# -----------------------------
def sma(s, n): return s.rolling(n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = up.rolling(n).mean() / down.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def compute_features(df):
    df = df.copy()
    df["SMA20"] = sma(df["close"], 20)
    df["SMA50"] = sma(df["close"], 50)
    df["SMA200"] = sma(df["close"], 200)
    df["RSI14"] = rsi(df["close"])
    df["ATR14"] = atr(df)
    return df

def classify_trend(df):
    if len(df) < 220 or df["SMA200"].isna().iloc[-1]:
        return {"direction": "insufficient", "strength": "", "regime": "", "notes": "Need ~200 bars"}

    last = df.iloc[-1]
    s20, s50, s200 = last["SMA20"], last["SMA50"], last["SMA200"]
    r = last["RSI14"]

    if s20 > s50 > s200:
        d, s = "bullish", "strong"
    elif s20 > s50:
        d, s = "bullish", "mild"
    elif s20 < s50 < s200:
        d, s = "bearish", "strong"
    elif s20 < s50:
        d, s = "bearish", "mild"
    else:
        d, s = "neutral", "mild"

    regime = "range" if 45 <= r <= 55 else "trending"
    return {"direction": d, "strength": s, "regime": regime, "notes": f"RSI={r:.1f}"}

def decide_bias(trends):
    score = 0
    weights = {"1D": 3, "1h": 2, "5m": 1}
    for tf, w in weights.items():
        d = trends.get(tf, {}).get("direction")
        if d == "bullish": score += w
        if d == "bearish": score -= w
    return "bullish" if score >= 4 else "bearish" if score <= -4 else "neutral"

# -----------------------------
# Yahoo Finance (RATE SAFE)
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_yf(symbol, interval, period):
    tkr = yf.Ticker(symbol)
    for attempt in range(4):
        try:
            hist = tkr.history(interval=interval, period=period, auto_adjust=False)
            if hist.empty:
                return None
            df = hist.rename(columns={
                "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
            })[["open","high","low","close","volume"]]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df.tz_convert(TZ)
        except YFRateLimitError:
            time.sleep((2 ** attempt) + random.uniform(0, 0.8))
        except Exception:
            return None
    return None

def resample_ohlcv(df, rule):
    return pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()

def fetch_timeframes(symbol):
    df_5m = fetch_yf(symbol, "5m", "60d")
    df_1d = fetch_yf(symbol, "1d", "2y")
    df_1h = resample_ohlcv(df_5m, "1H") if df_5m is not None else None
    return {"5m": df_5m, "1h": df_1h, "1D": df_1d}

# -----------------------------
# Options / Greeks
# -----------------------------
def bs_delta(S, K, T, r, iv, call=True):
    if T <= 0 or iv <= 0: return np.nan
    d1 = (math.log(S/K)+(r+0.5*iv*iv)*T)/(iv*math.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1)-1

def enrich_chain(df, S, expiry, r, call=True):
    T = max((datetime.strptime(expiry,"%Y-%m-%d").date()-date.today()).days/365, 1e-6)
    df = df.copy()
    df["mid"] = (df["bid"]+df["ask"])/2
    df["delta"] = df.apply(lambda x: bs_delta(S, x["strike"], T, r, x["impliedVolatility"], call), axis=1)
    return df.dropna(subset=["mid","delta"])

def build_iron_condor(puts, calls, delta, wings):
    sp = puts.iloc[(puts["delta"] + delta).abs().argsort()].iloc[0]
    sc = calls.iloc[(calls["delta"] - delta).abs().argsort()].iloc[0]
    lp = puts[puts["strike"] < sp["strike"]].iloc[wings]
    lc = calls[calls["strike"] > sc["strike"]].iloc[wings]
    credit = (sp["mid"]-lp["mid"]) + (sc["mid"]-lc["mid"])
    width = max(sp["strike"]-lp["strike"], lc["strike"]-sc["strike"])
    return credit, width

# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("Controls")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

    symbol = st.selectbox("Ticker", st.session_state.watchlist)
    dte_min, dte_max = st.slider("DTE Range", 7, 60, (14, 30))
    delta = st.slider("Target Delta", 0.10, 0.35, 0.20)
    wings = st.slider("Wing Distance", 1, 6, 3)
    r = st.number_input("Risk-Free Rate", value=0.04)
    run = st.button("🚀 Run Scan")

# -----------------------------
# Main
# -----------------------------
if run:
    st.subheader(f"{symbol} Trend Analysis")

    frames = fetch_timeframes(symbol)
    trends = {}
    last_price = None

    for tf in TIMEFRAMES:
        df = frames.get(tf)
        if df is None:
            st.warning(f"{tf}: data unavailable")
            continue

        df = compute_features(df)
        t = classify_trend(df)
        trends[tf] = t
        last_price = df["close"].iloc[-1]

        with st.expander(f"{tf}: {t['direction']} {t['strength']} ({t['regime']})", expanded=(tf=="1D")):
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(df.tail(200)["close"], label="Close")
            ax.plot(df.tail(200)["SMA20"], label="SMA20")
            ax.plot(df.tail(200)["SMA50"], label="SMA50")
            ax.plot(df.tail(200)["SMA200"], label="SMA200")
            ax.legend()
            st.pyplot(fig)

    bias = decide_bias(trends)
    st.info(f"Bias: **{bias.upper()}**")

    # Options
    tkr = yf.Ticker(symbol)
    spot = last_price
    expiry = [e for e in tkr.options if dte_min <= (datetime.strptime(e,"%Y-%m-%d").date()-date.today()).days <= dte_max][0]

    chain = tkr.option_chain(expiry)
    calls = enrich_chain(chain.calls, spot, expiry, r, True)
    puts = enrich_chain(chain.puts, spot, expiry, r, False)

    credit, width = build_iron_condor(puts, calls, delta, wings)

    st.success("Iron Condor Suggested")
    st.write(f"Expiry: {expiry}")
    st.write(f"Est Credit: {credit:.2f}")
    st.write(f"Max Loss: {width-credit:.2f}")
    st.caption("Educational only — not financial advice.")