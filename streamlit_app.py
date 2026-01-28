# ===============================
# Mobile-Style Trend + Credit Spread App (Streamlit)
# 15m + 4h (resampled), 1D (daily) + multi-timeframe (1W/1M/3M/1Y)
# Strategies: Auto / Bull Put / Bear Call / Iron Condor (clean toggles)
# ===============================

import math
import time
import random
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from scipy.stats import norm

import plotly.graph_objects as go

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Trend + Spreads", layout="wide")

TZ = "America/New_York"
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "TSLA", "AMD", "META"]

# -----------------------------
# CSS to mimic the screenshots (dark cards + chips)
# -----------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0b0f14;
  --card: #121821;
  --card2: #0f151d;
  --muted: #7f8a9a;
  --text: #e7edf6;
  --accent: #20c997;   /* green */
  --accent2: #4dabf7;  /* blue */
  --danger: #ff6b6b;
  --border: rgba(255,255,255,0.06);
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { right: 1rem; }

.block-container { padding-top: 1rem !important; padding-bottom: 3.2rem !important; max-width: 1200px; }

h1, h2, h3, h4, h5, h6, p, div, span, label { color: var(--text) !important; }

.small-muted { color: var(--muted) !important; font-size: 0.88rem; }
.section-title { font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0 0.75rem 0; }

.card {
  background: linear-gradient(180deg, var(--card), var(--card2));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px;
}
.card-tight {
  background: linear-gradient(180deg, var(--card), var(--card2));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 12px;
}
.metric {
  font-size: 1.35rem;
  font-weight: 750;
  line-height: 1.15;
}
.submetric { color: var(--muted) !important; font-size: 0.88rem; margin-top: 2px; }
.pill {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  font-size: 0.88rem;
  color: var(--muted) !important;
}
.pill-green { color: var(--accent) !important; border-color: rgba(32,201,151,0.35); background: rgba(32,201,151,0.08); }
.pill-red   { color: var(--danger) !important; border-color: rgba(255,107,107,0.35); background: rgba(255,107,107,0.08); }

.chiprow { display:flex; gap:10px; overflow-x:auto; padding-bottom: 6px; }
.chip {
  min-width: 140px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 10px 12px;
}
.chip-selected {
  border-color: rgba(32,201,151,0.6);
  background: rgba(32,201,151,0.10);
  box-shadow: 0 0 0 1px rgba(32,201,151,0.2) inset;
}

.kv { display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
.kv .k { color: var(--muted) !important; font-size: 0.88rem; }
.kv .v { font-weight: 700; font-size: 1.05rem; }

hr { border-color: var(--border) !important; }

[data-testid="stTabs"] button[role="tab"] { color: var(--muted) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--text) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers: Indicators + Trend/Strength
# -----------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = up.rolling(n).mean() / down.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series):
    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA20"] = sma(d["close"], 20)
    d["SMA50"] = sma(d["close"], 50)
    d["RSI14"] = rsi(d["close"], 14)
    m, s, h = macd(d["close"])
    d["MACD"] = m
    d["MACD_SIGNAL"] = s
    d["MACD_HIST"] = h
    return d

def support_resistance(df: pd.DataFrame, lookback: int = 40):
    # Simple rolling support/resistance
    w = min(lookback, max(10, len(df)))
    sup = float(df["low"].tail(w).min())
    res = float(df["high"].tail(w).max())
    return sup, res

def classify_and_strength(df: pd.DataFrame) -> dict:
    # Returns direction + strength% like the screenshots
    if df is None or len(df) < 60:
        return {"direction": "INSUFFICIENT", "strength": 0, "notes": "Need more bars"}

    last = df.iloc[-1]
    close = float(last["close"])
    s20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
    s50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else np.nan
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    m = float(last["MACD"]) if pd.notna(last["MACD"]) else 0.0

    # Direction logic (simple + stable)
    direction = "NEUTRAL"
    if np.isfinite(s20) and np.isfinite(s50):
        if s20 > s50 and close >= s20:
            direction = "BULLISH"
        elif s20 < s50 and close <= s20:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

    # Strength heuristic 0-100
    strength = 50.0
    if np.isfinite(s20) and np.isfinite(s50) and s50 != 0:
        spread = (s20 - s50) / abs(s50)  # small number
        strength += np.clip(spread * 4000, -25, 25)  # cap contribution

    # RSI push (trendiness)
    strength += np.clip((r - 50) * 0.7, -20, 20) if direction != "NEUTRAL" else np.clip((abs(r - 50)) * 0.3, 0, 12)

    # MACD push
    strength += np.clip(m * 3.0, -15, 15)

    # If neutral, damp strength a bit
    if direction == "NEUTRAL":
        strength = 35 + (strength - 50) * 0.5

    strength = int(np.clip(strength, 0, 100))
    return {"direction": direction, "strength": strength, "notes": f"RSI={r:.1f}, MACD={m:.2f}"}

# -----------------------------
# Data: Yahoo (rate-safe) + caching
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_yf(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    tkr = yf.Ticker(symbol)
    for attempt in range(4):
        try:
            hist = tkr.history(interval=interval, period=period, auto_adjust=False)
            if hist is None or hist.empty:
                return None
            df = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})[
                ["open","high","low","close","volume"]
            ].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.tz_convert(TZ)
            df.index.name = "time"
            return df
        except YFRateLimitError:
            time.sleep((2 ** attempt) + random.uniform(0, 0.8))
        except Exception:
            return None
    return None

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low":  df["low"].resample(rule).min(),
        "close":df["close"].resample(rule).last(),
        "volume":df["volume"].resample(rule).sum(),
    }).dropna()
    out.index.name = "time"
    return out

def fetch_timeframes(symbol: str):
    # Only TWO Yahoo calls: 15m + 1D
    df_15m = fetch_yf(symbol, "15m", "60d")  # call #1
    df_1d  = fetch_yf(symbol, "1d", "2y")    # call #2

    df_4h = None
    if df_15m is not None and not df_15m.empty:
        df_4h = resample_ohlcv(df_15m, "4H")

    return {"15m": df_15m, "4h": df_4h, "1D": df_1d}

# -----------------------------
# Multi-timeframe from daily: 1D/1W/1M/3M/1Y (like screenshot bars)
# -----------------------------
def slice_daily_for_tf(df_daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return df_daily
    if tf == "1D":  return df_daily.tail(60)   # need indicators; not literally 1 bar
    if tf == "1W":  return df_daily.tail(90)
    if tf == "1M":  return df_daily.tail(140)
    if tf == "3M":  return df_daily.tail(220)
    if tf == "1Y":  return df_daily.tail(520)
    return df_daily.tail(220)

# -----------------------------
# Options + spreads
# -----------------------------
def bs_delta(S, K, T, r, sigma, is_call: bool):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

def enrich_chain_with_delta(chain_df: pd.DataFrame, S: float, expiry: str, r: float, is_call: bool):
    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = max((ed - date.today()).days / 365.0, 1e-6)
    df = chain_df.copy()

    if "impliedVolatility" not in df.columns:
        df["impliedVolatility"] = np.nan

    bid = df.get("bid", pd.Series(np.nan, index=df.index))
    ask = df.get("ask", pd.Series(np.nan, index=df.index))
    df["mid"] = np.where(np.isfinite(bid) & np.isfinite(ask) & (ask > 0), (bid + ask) / 2.0, np.nan)

    deltas = []
    for _, row in df.iterrows():
        K = float(row.get("strike", np.nan))
        iv = float(row.get("impliedVolatility", np.nan))
        if not np.isfinite(K) or not np.isfinite(iv):
            deltas.append(np.nan)
        else:
            deltas.append(bs_delta(S, K, T, r, iv, is_call))
    df["delta_est"] = deltas

    return df

def pick_expiration_in_window(tkr: yf.Ticker, dte_min: int, dte_max: int):
    exps = tkr.options
    if not exps:
        return None
    today = date.today()
    target = int(round((dte_min + dte_max) / 2))
    scored = []
    for e in exps:
        ed = datetime.strptime(e, "%Y-%m-%d").date()
        dte = (ed - today).days
        scored.append((abs(dte - target), dte, e))
    within = [x for x in scored if dte_min <= x[1] <= dte_max]
    return sorted(within or scored, key=lambda x: x[0])[0][2]

def _pick_short_by_delta(df: pd.DataFrame, target_delta: float, want_call: bool):
    d_target = target_delta if want_call else -target_delta
    x = df.dropna(subset=["delta_est", "strike"]).copy()
    if x.empty:
        return None
    x["err"] = (x["delta_est"] - d_target).abs()
    return x.sort_values("err").iloc[0]

def _pick_wing(df: pd.DataFrame, short_strike: float, wing_steps: int, want_call: bool):
    x = df.dropna(subset=["strike"]).copy()
    if want_call:
        cand = x[x["strike"] > short_strike].sort_values("strike", ascending=True)
    else:
        cand = x[x["strike"] < short_strike].sort_values("strike", ascending=False)
    if cand.empty:
        return None
    idx = min(max(wing_steps, 1), len(cand) - 1)
    return cand.iloc[idx]

def _safe_mid(row):
    m = row.get("mid", np.nan)
    return float(m) if np.isfinite(m) else np.nan

def build_bull_put(puts_e, target_delta: float, wing_steps: int):
    short = _pick_short_by_delta(puts_e, target_delta, want_call=False)
    if short is None:
        return None
    long = _pick_wing(puts_e, float(short["strike"]), wing_steps, want_call=False)
    if long is None:
        return None
    credit = _safe_mid(short) - _safe_mid(long)
    width = abs(float(short["strike"]) - float(long["strike"]))
    max_loss = (width - credit) if np.isfinite(credit) else np.nan
    return {
        "name": "Bull Put Spread",
        "legs": [("SELL PUT", float(short["strike"])), ("BUY PUT", float(long["strike"]))],
        "credit": credit,
        "width": width,
        "max_loss": max_loss,
    }

def build_bear_call(calls_e, target_delta: float, wing_steps: int):
    short = _pick_short_by_delta(calls_e, target_delta, want_call=True)
    if short is None:
        return None
    long = _pick_wing(calls_e, float(short["strike"]), wing_steps, want_call=True)
    if long is None:
        return None
    credit = _safe_mid(short) - _safe_mid(long)
    width = abs(float(long["strike"]) - float(short["strike"]))
    max_loss = (width - credit) if np.isfinite(credit) else np.nan
    return {
        "name": "Bear Call Spread",
        "legs": [("SELL CALL", float(short["strike"])), ("BUY CALL", float(long["strike"]))],
        "credit": credit,
        "width": width,
        "max_loss": max_loss,
    }

def build_iron_condor(puts_e, calls_e, target_delta: float, wing_steps: int):
    sp = _pick_short_by_delta(puts_e, target_delta, want_call=False)
    sc = _pick_short_by_delta(calls_e, target_delta, want_call=True)
    if sp is None or sc is None:
        return None
    lp = _pick_wing(puts_e,  float(sp["strike"]), wing_steps, want_call=False)
    lc = _pick_wing(calls_e, float(sc["strike"]), wing_steps, want_call=True)
    if lp is None or lc is None:
        return None

    spk, lpk = float(sp["strike"]), float(lp["strike"])
    sck, lck = float(sc["strike"]), float(lc["strike"])

    put_credit  = _safe_mid(sp) - _safe_mid(lp)
    call_credit = _safe_mid(sc) - _safe_mid(lc)
    total_credit = (put_credit + call_credit) if np.isfinite(put_credit) and np.isfinite(call_credit) else np.nan

    put_w = abs(spk - lpk)
    call_w = abs(lck - sck)
    max_w = max(put_w, call_w)
    max_loss = (max_w - total_credit) if np.isfinite(total_credit) else np.nan

    return {
        "name": "Iron Condor",
        "legs": [("SELL PUT", spk), ("BUY PUT", lpk), ("SELL CALL", sck), ("BUY CALL", lck)],
        "credit": total_credit,
        "width": max_w,
        "max_loss": max_loss,
        "put_credit": put_credit,
        "call_credit": call_credit,
    }

# -----------------------------
# Session state
# -----------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

if "selected" not in st.session_state:
    st.session_state.selected = st.session_state.watchlist[0]

# -----------------------------
# Header row: Search + demo chip row
# -----------------------------
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
with top_right:
    st.markdown("<div style='text-align:right;'><span class='pill pill-green'>Demo Data</span></div>", unsafe_allow_html=True)

search = st.text_input("", placeholder="Search stocks…", label_visibility="collapsed").strip().upper()
if search and search not in st.session_state.watchlist:
    # quick-add on search enter
    st.session_state.watchlist.insert(0, search)

# Chip row (buttons)
chip_cols = st.columns(4)
for i, sym in enumerate(st.session_state.watchlist[:4]):
    with chip_cols[i]:
        selected = (sym == st.session_state.selected)
        # Try to show price + % change quickly (best effort)
        price, chg = None, None
        try:
            fi = yf.Ticker(sym).fast_info
            price = float(fi.get("last_price", np.nan))
            prev = float(fi.get("previous_close", np.nan))
            if np.isfinite(price) and np.isfinite(prev) and prev != 0:
                chg = (price - prev) / prev * 100.0
        except Exception:
            pass

        label = f"{sym}\n"
        if price is not None and np.isfinite(price):
            label += f"${price:,.2f}  "
            if chg is not None and np.isfinite(chg):
                label += f"{chg:+.2f}%"
        else:
            label += "—"

        if st.button(label, use_container_width=True, key=f"chip_{sym}"):
            st.session_state.selected = sym

symbol = st.session_state.selected

# -----------------------------
# Fetch data
# -----------------------------
with st.spinner("Loading charts…"):
    frames = fetch_timeframes(symbol)

df_15m = frames.get("15m")
df_4h  = frames.get("4h")
df_1d  = frames.get("1D")

# Spot price (best effort)
spot = np.nan
try:
    spot = float(yf.Ticker(symbol).fast_info.get("last_price", np.nan))
except Exception:
    pass
if not np.isfinite(spot):
    if df_15m is not None and not df_15m.empty:
        spot = float(df_15m["close"].iloc[-1])
    elif df_1d is not None and not df_1d.empty:
        spot = float(df_1d["close"].iloc[-1])

# -----------------------------
# Timeframe pills like screenshot (chart range)
# -----------------------------
tf_pills = ["1D", "1W", "1M", "3M", "1Y"]
tf_choice = st.radio("",
                     tf_pills,
                     horizontal=True,
                     label_visibility="collapsed")

# Build display df for chart based on choice
def chart_slice(df15, dfd, choice: str):
    if choice == "1D":
        # show 15m (intraday) if available
        return df15.tail(26*4) if df15 is not None and not df15.empty else dfd.tail(60)
    if choice == "1W":
        return dfd.tail(7*6) if dfd is not None else None
    if choice == "1M":
        return dfd.tail(35) if dfd is not None else None
    if choice == "3M":
        return dfd.tail(95) if dfd is not None else None
    if choice == "1Y":
        return dfd.tail(260) if dfd is not None else None
    return dfd.tail(95) if dfd is not None else None

df_chart = chart_slice(df_15m, df_1d, tf_choice)
df_chart = compute_features(df_chart) if df_chart is not None and not df_chart.empty else None

# Current timeframe analysis uses the chart slice
current_summary = classify_and_strength(df_chart) if df_chart is not None else {"direction":"INSUFFICIENT","strength":0,"notes":"No data"}

# -----------------------------
# Stock header (like screenshot)
# -----------------------------
hdr_l, hdr_r = st.columns([3, 2])
with hdr_l:
    st.markdown(f"<div style='font-size:1.45rem; font-weight:800;'>{symbol}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>{symbol}</div>", unsafe_allow_html=True)

with hdr_r:
    price_line = "—" if not np.isfinite(spot) else f"${spot:,.2f}"
    st.markdown(f"<div style='text-align:right; font-size:2.1rem; font-weight:900;'>{price_line}</div>", unsafe_allow_html=True)

# -----------------------------
# Candlestick chart (Plotly)
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
if df_chart is None or df_chart.empty:
    st.warning("Chart unavailable (rate-limited or no data). Try again shortly.")
else:
    cdf = df_chart.copy().dropna(subset=["open","high","low","close"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=cdf.index,
        open=cdf["open"],
        high=cdf["high"],
        low=cdf["low"],
        close=cdf["close"],
        name="Price"
    ))
    # overlays
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA50"], mode="lines", name="SMA50"))

    fig.update_layout(
        height=360,
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="rgba(255,255,255,0.5)"),
        yaxis=dict(showgrid=False, color="rgba(255,255,255,0.5)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Tabs: Analysis + Strategies + Watchlist (like bottom nav idea)
# -----------------------------
tabA, tabB, tabC = st.tabs(["📊 Analysis", "🧠 Strategies", "⭐ Watchlist"])

# =============================
# ANALYSIS TAB (cards + multi-timeframe bars)
# =============================
with tabA:
    st.markdown("<div class='section-title'>Current Timeframe Analysis</div>", unsafe_allow_html=True)

    dir_ = current_summary["direction"]
    strength = current_summary["strength"]
    pill_cls = "pill-green" if dir_ == "BULLISH" else ("pill-red" if dir_ == "BEARISH" else "")
    st.markdown(
        f"""
        <div class='card'>
          <div class='kv'>
            <div><span class='pill {pill_cls}'>{dir_}</span></div>
            <div style='text-align:right;'><div class='small-muted'>Strength</div><div class='metric'>{strength}%</div></div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    if df_chart is not None and not df_chart.empty:
        sup, res = support_resistance(df_chart, lookback=40)
        last = df_chart.iloc[-1]
        s20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
        s50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else np.nan
        rsi_v = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
        macd_v = float(last["MACD"]) if pd.notna(last["MACD"]) else np.nan

        # Two-column grid of metric cards
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"<div class='card-tight'><div class='submetric'>Support</div><div class='metric'>${sup:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='submetric'>SMA 20</div><div class='metric'>${s20:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='submetric'>RSI</div><div class='metric'>{rsi_v:,.1f}</div></div>", unsafe_allow_html=True)

        with g2:
            st.markdown(f"<div class='card-tight'><div class='submetric'>Resistance</div><div class='metric'>${res:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='submetric'>SMA 50</div><div class='metric'>${s50:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='height:10px'></div>", unsafe_allow_html=True)
            macd_color = "color: var(--danger) !important;" if np.isfinite(macd_v) and macd_v < 0 else "color: var(--accent) !important;"
            st.markdown(
                f"<div class='card-tight'><div class='submetric'>MACD</div><div class='metric' style='{macd_color}'>{macd_v:,.2f}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)  # close card wrapper

    st.markdown("<div class='section-title'>Multi-Timeframe Trend Analysis</div>", unsafe_allow_html=True)

    # Multi-timeframe like screenshot: 1D/1W/1M/3M/1Y bars
    mtfs = ["1D", "1W", "1M", "3M", "1Y"]
    if df_1d is None or df_1d.empty:
        st.warning("Daily data unavailable right now (rate limit).")
    else:
        for tf in mtfs:
            d = slice_daily_for_tf(df_1d, tf)
            d = compute_features(d)
            summ = classify_and_strength(d)
            dir_tf = summ["direction"]
            str_tf = summ["strength"]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            top = st.columns([1, 1])
            with top[0]:
                st.markdown(f"<div style='font-size:1.05rem; font-weight:800;'>{tf}</div>", unsafe_allow_html=True)
            with top[1]:
                st.markdown(f"<div style='text-align:right; color: var(--muted); font-weight:700;'>{dir_tf}</div>", unsafe_allow_html=True)

            st.progress(str_tf / 100.0)
            st.markdown(f"<div class='small-muted'>{str_tf}% strength</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =============================
# STRATEGIES TAB (clean toggles + auto + spreads)
# =============================
with tabB:
    st.markdown("<div class='section-title'>Strategy Builder</div>", unsafe_allow_html=True)

    # Clean toggles
    # Auto uses bias from (15m,4h,1D) quick matrix:
    bias_inputs = {}
    # 15m/4h/1D trend
    def tf_summary(df):
        if df is None or df.empty:
            return {"direction":"INSUFFICIENT", "strength":0, "notes":""}
        return classify_and_strength(compute_features(df))

    bias_inputs["15m"] = tf_summary(df_15m)
    bias_inputs["4h"]  = tf_summary(df_4h)
    bias_inputs["1D"]  = tf_summary(df_1d)

    # weighted bias
    score = 0
    for tf, w in [("1D", 3), ("4h", 2), ("15m", 1)]:
        d = bias_inputs.get(tf, {}).get("direction", "INSUFFICIENT")
        if d == "BULLISH": score += w
        if d == "BEARISH": score -= w
    auto_bias = "NEUTRAL" if -3 < score < 3 else ("BULLISH" if score >= 3 else "BEARISH")

    strat_choice = st.radio(
        "Strategy",
        ["AUTO", "BULL PUT", "BEAR CALL", "IRON CONDOR"],
        horizontal=True,
        label_visibility="collapsed"
    )

    s1, s2, s3, s4 = st.columns([1.1, 1.1, 1.1, 1.2])
    with s1:
        dte_min, dte_max = st.slider("DTE", 7, 60, (14, 30))
    with s2:
        target_delta = st.slider("Target Δ", 0.10, 0.35, 0.20, 0.01)
    with s3:
        wing_steps = st.slider("Wing steps", 1, 8, 3, 1)
    with s4:
        r_rate = st.number_input("Risk-free r", 0.0, 0.20, 0.04, 0.005)

    go_btn = st.button("Build Strategy", type="primary", use_container_width=True)

    if go_btn:
        if not np.isfinite(spot):
            st.error("No spot price available right now.")
        else:
            tkr = yf.Ticker(symbol)

            expiry = pick_expiration_in_window(tkr, int(dte_min), int(dte_max))
            if not expiry:
                st.error("No expirations found (or Yahoo options is rate-limited).")
            else:
                ed = datetime.strptime(expiry, "%Y-%m-%d").date()
                dte = (ed - date.today()).days

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><div><div class='small-muted'>Spot</div><div class='metric'>${spot:,.2f}</div></div>"
                            f"<div style='text-align:right;'><div class='small-muted'>Expiry</div><div class='metric'>{expiry}</div>"
                            f"<div class='small-muted'>DTE={dte}</div></div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                try:
                    chain = tkr.option_chain(expiry)
                except YFRateLimitError:
                    st.warning("Yahoo rate-limited the options chain. Try again in a minute.")
                    chain = None
                except Exception:
                    st.warning("Options chain fetch failed. Try again.")
                    chain = None

                if chain is not None:
                    calls_e = enrich_chain_with_delta(chain.calls, S=spot, expiry=expiry, r=float(r_rate), is_call=True)
                    puts_e  = enrich_chain_with_delta(chain.puts,  S=spot, expiry=expiry, r=float(r_rate), is_call=False)

                    # decide final strategy in AUTO
                    final = strat_choice
                    if strat_choice == "AUTO":
                        if auto_bias == "BULLISH":
                            final = "BULL PUT"
                        elif auto_bias == "BEARISH":
                            final = "BEAR CALL"
                        else:
                            final = "IRON CONDOR"

                    st.markdown(f"<div class='small-muted'>Auto bias: <b>{auto_bias}</b> → using <b>{final}</b></div>", unsafe_allow_html=True)

                    if final == "BULL PUT":
                        strat = build_bull_put(puts_e, float(target_delta), int(wing_steps))
                    elif final == "BEAR CALL":
                        strat = build_bear_call(calls_e, float(target_delta), int(wing_steps))
                    else:
                        strat = build_iron_condor(puts_e, calls_e, float(target_delta), int(wing_steps))

                    if not strat:
                        st.error("Could not build this spread (missing IV/bid/ask or not enough strikes). Try different DTE/Δ/wings.")
                    else:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:1.2rem; font-weight:900;'>{strat['name']}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                        st.markdown("**Legs**")
                        for leg in strat["legs"]:
                            st.write(f"- {leg[0]} {leg[1]}")

                        credit = strat.get("credit", np.nan)
                        width = strat.get("width", np.nan)
                        max_loss = strat.get("max_loss", np.nan)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Est. Credit", "—" if not np.isfinite(credit) else f"{credit:.2f}")
                        m2.metric("Width", "—" if not np.isfinite(width) else f"{width:.2f}")
                        m3.metric("Est. Max Loss", "—" if not np.isfinite(max_loss) else f"{max_loss:.2f}")

                        if strat["name"] == "Iron Condor" and np.isfinite(strat.get("put_credit", np.nan)):
                            st.markdown(
                                f"<div class='small-muted'>Put credit: {strat['put_credit']:.2f} | Call credit: {strat['call_credit']:.2f}</div>",
                                unsafe_allow_html=True
                            )

                        st.caption("Educational only — not investment advice.")
                        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# WATCHLIST TAB
# =============================
with tabC:
    st.markdown("<div class='section-title'>Watchlist</div>", unsafe_allow_html=True)

    cA, cB = st.columns([2, 1])
    with cA:
        new_sym = st.text_input("Add ticker", placeholder="NFLX").strip().upper()
    with cB:
        add_btn = st.button("Add", use_container_width=True)

    if add_btn and new_sym:
        if new_sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_sym)
            st.success(f"Added {new_sym}")

    # list
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    for sym in st.session_state.watchlist[:25]:
        row = st.columns([1, 1])
        with row[0]:
            if st.button(sym, key=f"wl_{sym}"):
                st.session_state.selected = sym
                st.experimental_rerun()
        with row[1]:
            if st.button("Remove", key=f"rm_{sym}"):
                if sym in st.session_state.watchlist and len(st.session_state.watchlist) > 1:
                    st.session_state.watchlist.remove(sym)
                    if st.session_state.selected == sym:
                        st.session_state.selected = st.session_state.watchlist[0]
                    st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)