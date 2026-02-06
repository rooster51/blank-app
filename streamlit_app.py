# ===============================
# Trend + Credit Spreads App (Streamlit)
# Dark mode readable UI
# Watchlist moved to TOP ribbon (above chart)
# Timeframes: 15m + 4h (resampled) + 1D
# Includes full Strategies + Watchlist pages
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
import streamlit.components.v1 as components

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Trend + Spreads", layout="wide")
TZ = "America/New_York"

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "TSLA", "AMD", "META"]

PRESETS = {
    "Conservative": {"dte_min": 30, "dte_max": 45, "target_delta": 0.15, "wing_steps": 4},
    "Standard":     {"dte_min": 14, "dte_max": 30, "target_delta": 0.20, "wing_steps": 3},
    "Aggressive":   {"dte_min":  7, "dte_max": 21, "target_delta": 0.25, "wing_steps": 2},
}

# -----------------------------
# CSS: force dark widgets + readable layout
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#0b0f14;
  --panel:#111822;
  --panel2:#0e141c;
  --text:#e7edf6;
  --muted:#93a1b3;
  --border:rgba(255,255,255,0.10);
  --accent:#20c997;
  --danger:#ff6b6b;
  --chip:#0f1620;
  --chipActive: rgba(32,201,151,0.12);
}

html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 5.8rem !important; max-width: 1100px; }

h1,h2,h3,h4,h5,h6,p,div,span,label { color: var(--text) !important; }
.small-muted { color: var(--muted) !important; font-size: 0.9rem; }
.section-title{ font-size:1.2rem; font-weight:900; margin: 0.6rem 0 0.8rem; }
.metric{ font-size:1.35rem; font-weight:950; }
.bigprice{ font-size:2.2rem; font-weight:980; letter-spacing:-0.5px; }

.card{
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px;
}
.card-tight{
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px;
}

/* Dark inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(147,161,179,0.75) !important; }

/* Buttons */
.stButton button{
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  font-weight: 900 !important;
}
.stButton button:hover{
  border-color: rgba(32,201,151,0.45) !important;
}
button[kind="primary"]{
  background: rgba(32,201,151,0.14) !important;
  border-color: rgba(32,201,151,0.60) !important;
}

/* Pills */
.pill{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  font-size: 0.88rem;
  color: var(--muted) !important;
}
.pill-green{ color: var(--accent) !important; border-color: rgba(32,201,151,0.50); background: rgba(32,201,151,0.12); }
.pill-red{ color: var(--danger) !important; border-color: rgba(255,107,107,0.50); background: rgba(255,107,107,0.12); }

/* TOP RIBBON watchlist cards */
.ribbon{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 12px;
}
.wrow{
  display:flex;
  gap:10px;
  overflow-x:auto;
  padding-bottom: 6px;
}
.wcard{
  min-width: 150px;
  background: var(--chip);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 10px 12px;
}
.wcard.active{
  border-color: rgba(32,201,151,0.60);
  background: var(--chipActive);
  box-shadow: 0 0 0 1px rgba(32,201,151,0.18) inset;
}
.wsym{ font-weight: 950; font-size: 1.05rem; }
.wpx{ font-weight: 900; font-size: 1.0rem; margin-top: 4px; }
.wchg.pos{ color: var(--accent) !important; font-weight: 900; }
.wchg.neg{ color: var(--danger) !important; font-weight: 900; }

/* Make the "Select" buttons under tiles look like tap targets but low-visual */
.tilebtn .stButton button{
  background: rgba(255,255,255,0.02) !important;
  border-color: rgba(255,255,255,0.06) !important;
  color: rgba(255,255,255,0.70) !important;
  font-weight: 850 !important;
  padding-top: 6px !important;
  padding-bottom: 6px !important;
}

/* Fixed bottom nav */
.fixed-nav{
  position: fixed; left: 0; right: 0; bottom: 0;
  background: rgba(10,14,19,0.92);
  border-top: 1px solid var(--border);
  backdrop-filter: blur(10px);
  padding: 10px 14px;
  z-index: 99999;
}
.navrow{ display:flex; gap:10px; max-width: 900px; margin: 0 auto; }
.navbtn{
  width: 32%;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.06);
  color: var(--muted);
  font-weight: 950;
  cursor: pointer;
}
.navbtn.active{
  border-color: rgba(32,201,151,0.60);
  background: rgba(32,201,151,0.12);
  color: var(--text);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Indicators
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
    w = min(lookback, max(10, len(df)))
    sup = float(df["low"].tail(w).min())
    res = float(df["high"].tail(w).max())
    return sup, res

def classify_and_strength(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 60:
        return {"direction": "INSUFFICIENT", "strength": 0, "notes": "Need more bars"}

    last = df.iloc[-1]
    close = float(last["close"])
    s20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
    s50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else np.nan
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    m = float(last["MACD"]) if pd.notna(last["MACD"]) else 0.0

    direction = "NEUTRAL"
    if np.isfinite(s20) and np.isfinite(s50):
        if s20 > s50 and close >= s20:
            direction = "BULLISH"
        elif s20 < s50 and close <= s20:
            direction = "BEARISH"

    strength = 50.0
    if np.isfinite(s20) and np.isfinite(s50) and s50 != 0:
        spread = (s20 - s50) / abs(s50)
        strength += np.clip(spread * 4000, -25, 25)

    if direction != "NEUTRAL":
        strength += np.clip((r - 50) * 0.7, -20, 20)
    else:
        strength += np.clip((abs(r - 50)) * 0.3, 0, 12)

    strength += np.clip(m * 3.0, -15, 15)
    if direction == "NEUTRAL":
        strength = 35 + (strength - 50) * 0.5

    return {"direction": direction, "strength": int(np.clip(strength, 0, 100)), "notes": f"RSI={r:.1f}, MACD={m:.2f}"}

# -----------------------------
# Yahoo Finance (rate-safe)
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
    df_15m = fetch_yf(symbol, "15m", "60d")  # call #1
    df_1d  = fetch_yf(symbol, "1d", "2y")    # call #2
    df_4h = resample_ohlcv(df_15m, "4H") if df_15m is not None and not df_15m.empty else None
    return {"15m": df_15m, "4h": df_4h, "1D": df_1d}

def slice_daily_for_tf(df_daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return df_daily
    if tf == "1D":  return df_daily.tail(60)
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
    return {"name":"Bull Put Spread", "legs":[("SELL PUT", float(short["strike"])), ("BUY PUT", float(long["strike"]))],
            "credit":credit, "width":width, "max_loss":max_loss}

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
    return {"name":"Bear Call Spread", "legs":[("SELL CALL", float(short["strike"])), ("BUY CALL", float(long["strike"]))],
            "credit":credit, "width":width, "max_loss":max_loss}

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
        "name":"Iron Condor",
        "legs":[("SELL PUT", spk), ("BUY PUT", lpk), ("SELL CALL", sck), ("BUY CALL", lck)],
        "credit":total_credit, "width":max_w, "max_loss":max_loss,
        "put_credit":put_credit, "call_credit":call_credit
    }

# Manual metrics helpers (kept for future expansion if you want manual leg picker back here)
def mid_for_strike(df: pd.DataFrame, strike: float) -> float:
    x = df[df["strike"] == strike]
    if x.empty:
        return np.nan
    return float(x["mid"].iloc[0]) if np.isfinite(x["mid"].iloc[0]) else np.nan

def calc_vertical_credit(short_mid: float, long_mid: float, width: float):
    if not np.isfinite(short_mid) or not np.isfinite(long_mid) or width <= 0:
        return np.nan, np.nan
    credit = short_mid - long_mid
    max_loss = width - credit
    return credit, max_loss

def calc_condor(put_short_mid, put_long_mid, call_short_mid, call_long_mid, put_w, call_w):
    if not all(np.isfinite(x) for x in [put_short_mid, put_long_mid, call_short_mid, call_long_mid]):
        return np.nan, np.nan, np.nan, np.nan
    put_credit = put_short_mid - put_long_mid
    call_credit = call_short_mid - call_long_mid
    total_credit = put_credit + call_credit
    max_w = max(put_w, call_w)
    max_loss = max_w - total_credit
    return total_credit, max_loss, put_credit, call_credit

# -----------------------------
# Session state + routing
# -----------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
if "selected" not in st.session_state:
    st.session_state.selected = st.session_state.watchlist[0]

qp = st.query_params
page = qp.get("page", "analysis")
if page not in ("analysis", "strategies", "watchlist"):
    page = "analysis"

symbol = st.session_state.selected

# -----------------------------
# Top ribbon: Search + Watchlist + Timeframe
# -----------------------------
st.markdown("<div class='ribbon'>", unsafe_allow_html=True)
st.markdown("<div class='section-title' style='margin:0 0 0.65rem 0;'>Analysis</div>", unsafe_allow_html=True)

search = st.text_input("", placeholder="Search stocks…", label_visibility="collapsed").strip().upper()
if search:
    if search not in st.session_state.watchlist:
        st.session_state.watchlist.insert(0, search)
    st.session_state.selected = search
    symbol = st.session_state.selected

WATCH_N = 8
prices = {}
for sym in st.session_state.watchlist[:WATCH_N]:
    try:
        fi = yf.Ticker(sym).fast_info
        last = float(fi.get("last_price", np.nan))
        prev = float(fi.get("previous_close", np.nan))
        chg = np.nan
        if np.isfinite(last) and np.isfinite(prev) and prev != 0:
            chg = (last - prev) / prev * 100.0
        prices[sym] = (last, chg)
    except Exception:
        prices[sym] = (np.nan, np.nan)

# cards row
st.markdown("<div class='wrow'>", unsafe_allow_html=True)
for sym in st.session_state.watchlist[:WATCH_N]:
    last, chg = prices.get(sym, (np.nan, np.nan))
    active = "active" if sym == st.session_state.selected else ""
    px = "—" if not np.isfinite(last) else f"${last:,.2f}"
    if np.isfinite(chg):
        chg_cls = "pos" if chg >= 0 else "neg"
        chg_txt = f"{chg:+.2f}%"
    else:
        chg_cls = ""
        chg_txt = ""

    st.markdown(
        f"""
        <div class='wcard {active}'>
          <div class='wsym'>{sym}</div>
          <div class='wpx'>{px}</div>
          <div class='wchg {chg_cls}'>{chg_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# tap targets (under tiles)
btn_cols = st.columns(min(WATCH_N, len(st.session_state.watchlist)))
for i, sym in enumerate(st.session_state.watchlist[:WATCH_N]):
    with btn_cols[i]:
        st.markdown("<div class='tilebtn'>", unsafe_allow_html=True)
        if st.button("Select", key=f"sel_{sym}", use_container_width=True):
            st.session_state.selected = sym
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

tf_choice = st.segmented_control("Timeframe", options=["1D", "1W", "1M", "3M", "1Y"], default="1M")
st.markdown("</div>", unsafe_allow_html=True)

symbol = st.session_state.selected

# -----------------------------
# Fetch data
# -----------------------------
with st.spinner("Loading…"):
    frames = fetch_timeframes(symbol)

df_15m = frames.get("15m")
df_4h  = frames.get("4h")
df_1d  = frames.get("1D")

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

# AUTO bias from 15m/4h/1D
def tf_summary(df):
    if df is None or df.empty:
        return {"direction":"INSUFFICIENT", "strength":0}
    return classify_and_strength(compute_features(df))

bias_inputs = {"15m": tf_summary(df_15m), "4h": tf_summary(df_4h), "1D": tf_summary(df_1d)}
score = 0
for tf, w in [("1D", 3), ("4h", 2), ("15m", 1)]:
    d = bias_inputs.get(tf, {}).get("direction", "INSUFFICIENT")
    if d == "BULLISH": score += w
    if d == "BEARISH": score -= w
auto_bias = "NEUTRAL" if -3 < score < 3 else ("BULLISH" if score >= 3 else "BEARISH")

# -----------------------------
# Chart setup
# -----------------------------
def chart_slice(df15, dfd, choice: str):
    if choice == "1D":
        return df15.tail(26 * 2) if df15 is not None and not df15.empty else dfd.tail(60)
    if choice == "1W":
        return dfd.tail(40) if dfd is not None else None
    if choice == "1M":
        return dfd.tail(35) if dfd is not None else None
    if choice == "3M":
        return dfd.tail(95) if dfd is not None else None
    if choice == "1Y":
        return dfd.tail(260) if dfd is not None else None
    return dfd.tail(95) if dfd is not None else None

df_chart = chart_slice(df_15m, df_1d, tf_choice)
df_chart = compute_features(df_chart) if df_chart is not None and not df_chart.empty else None
current_summary = classify_and_strength(df_chart) if df_chart is not None else {"direction":"INSUFFICIENT","strength":0,"notes":"No data"}

# Header (stock + price)
hdr_l, hdr_r = st.columns([3, 2])
with hdr_l:
    st.markdown(f"<div style='font-size:1.55rem; font-weight:980;'>{symbol}</div>", unsafe_allow_html=True)
with hdr_r:
    st.markdown(f"<div style='text-align:right;' class='bigprice'>{'—' if not np.isfinite(spot) else f'${spot:,.2f}'}</div>", unsafe_allow_html=True)

# Chart card
st.markdown("<div class='card'>", unsafe_allow_html=True)
if df_chart is None or df_chart.empty:
    st.warning("Chart unavailable (rate-limited or no data). Try again shortly.")
else:
    cdf = df_chart.dropna(subset=["open","high","low","close"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=cdf.index, open=cdf["open"], high=cdf["high"], low=cdf["low"], close=cdf["close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA50"], mode="lines", name="SMA50"))
    fig.update_layout(
        height=360,
        margin=dict(l=8,r=8,t=8,b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="rgba(255,255,255,0.70)"),
        yaxis=dict(showgrid=False, color="rgba(255,255,255,0.70)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Pages content
# -----------------------------
if page == "analysis":
    st.markdown("<div class='section-title'>Current Timeframe Analysis</div>", unsafe_allow_html=True)
    dir_ = current_summary["direction"]
    strength = current_summary["strength"]
    pill_cls = "pill-green" if dir_ == "BULLISH" else ("pill-red" if dir_ == "BEARISH" else "")

    st.markdown(
        f"""
        <div class='card'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div><span class='pill {pill_cls}'>{dir_}</span></div>
            <div style='text-align:right;'>
              <div class='small-muted'>Strength</div>
              <div class='metric'>{strength}%</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    if df_chart is not None and not df_chart.empty:
        sup, res = support_resistance(df_chart, 40)
        last = df_chart.iloc[-1]
        s20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
        s50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else np.nan
        rsi_v = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
        macd_v = float(last["MACD"]) if pd.notna(last["MACD"]) else np.nan

        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"<div class='card-tight'><div class='small-muted'>Support</div><div class='metric'>${sup:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='small-muted'>SMA 20</div><div class='metric'>${s20:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='small-muted'>RSI</div><div class='metric'>{rsi_v:,.1f}</div></div>", unsafe_allow_html=True)
        with g2:
            st.markdown(f"<div class='card-tight'><div class='small-muted'>Resistance</div><div class='metric'>${res:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tight'><div class='small-muted'>SMA 50</div><div class='metric'>${s50:,.2f}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            macd_style = "color: var(--danger) !important;" if np.isfinite(macd_v) and macd_v < 0 else "color: var(--accent) !important;"
            st.markdown(f"<div class='card-tight'><div class='small-muted'>MACD</div><div class='metric' style='{macd_style}'>{macd_v:,.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Multi-Timeframe Trend Analysis</div>", unsafe_allow_html=True)
    mtfs = ["1D", "1W", "1M", "3M", "1Y"]
    if df_1d is None or df_1d.empty:
        st.warning("Daily data unavailable right now (rate limit).")
    else:
        for tf in mtfs:
            d = compute_features(slice_daily_for_tf(df_1d, tf))
            summ = classify_and_strength(d)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            a, b = st.columns([1, 1])
            with a:
                st.markdown(f"<div style='font-size:1.05rem; font-weight:900;'>{tf}</div>", unsafe_allow_html=True)
            with b:
                st.markdown(f"<div style='text-align:right; color: var(--muted); font-weight:800;'>{summ['direction']}</div>", unsafe_allow_html=True)
            st.progress(summ["strength"] / 100.0)
            st.markdown(f"<div class='small-muted'>{summ['strength']}% strength</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "strategies":
    st.markdown("<div class='section-title'>Strategies</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>AUTO bias: <b>{auto_bias}</b></div>", unsafe_allow_html=True)

    strat_choice = st.segmented_control("Strategy", options=["AUTO", "BULL PUT", "BEAR CALL", "IRON CONDOR"], default="AUTO")
    preset = st.segmented_control("Preset", options=list(PRESETS.keys()), default="Standard")
    pv = PRESETS[preset]

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        dte_min, dte_max = st.slider("DTE", 7, 60, (pv["dte_min"], pv["dte_max"]))
    with c2:
        target_delta = st.slider("Target Δ", 0.10, 0.35, float(pv["target_delta"]), 0.01)
    with c3:
        wing_steps = st.slider("Wings", 1, 8, int(pv["wing_steps"]), 1)
    with c4:
        r_rate = st.number_input("Risk-free r", 0.0, 0.20, 0.04, 0.005)

    build_btn = st.button("Build Strategy", type="primary", use_container_width=True)

    if build_btn:
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
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<div><div class='small-muted'>Spot</div><div class='metric'>${spot:,.2f}</div></div>"
                    f"<div style='text-align:right;'><div class='small-muted'>Expiry</div>"
                    f"<div class='metric'>{expiry}</div><div class='small-muted'>DTE={dte}</div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
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

                    final = strat_choice
                    if strat_choice == "AUTO":
                        final = "BULL PUT" if auto_bias == "BULLISH" else ("BEAR CALL" if auto_bias == "BEARISH" else "IRON CONDOR")

                    if final == "BULL PUT":
                        strat = build_bull_put(puts_e, float(target_delta), int(wing_steps))
                    elif final == "BEAR CALL":
                        strat = build_bear_call(calls_e, float(target_delta), int(wing_steps))
                    else:
                        strat = build_iron_condor(puts_e, calls_e, float(target_delta), int(wing_steps))

                    if not strat:
                        st.error("Could not build spread. Try different DTE/Δ/Wings.")
                    else:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:1.2rem; font-weight:950;'>{strat['name']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='small-muted'>Using {final} | AUTO bias {auto_bias}</div>", unsafe_allow_html=True)

                        st.markdown("**Legs**")
                        for leg in strat["legs"]:
                            st.write(f"- {leg[0]} {leg[1]}")

                        credit = strat.get("credit", np.nan)
                        width = strat.get("width", np.nan)
                        max_loss = strat.get("max_loss", np.nan)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Credit", "—" if not np.isfinite(credit) else f"{credit:.2f}")
                        m2.metric("Width", "—" if not np.isfinite(width) else f"{width:.2f}")
                        m3.metric("Max Loss", "—" if not np.isfinite(max_loss) else f"{max_loss:.2f}")

                        if strat["name"] == "Iron Condor" and np.isfinite(strat.get("put_credit", np.nan)):
                            st.markdown(
                                f"<div class='small-muted'>Put credit: {strat['put_credit']:.2f} | Call credit: {strat['call_credit']:.2f}</div>",
                                unsafe_allow_html=True
                            )

                        st.caption("Educational only — not investment advice.")
                        st.markdown("</div>", unsafe_allow_html=True)

elif page == "watchlist":
    st.markdown("<div class='section-title'>Watchlist</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cA, cB = st.columns([2, 1])
    with cA:
        new_sym = st.text_input("Add ticker", placeholder="NFLX").strip().upper()
    with cB:
        add_btn = st.button("Add", use_container_width=True)

    if add_btn and new_sym:
        if new_sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_sym)
            st.success(f"Added {new_sym}")
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    for sym in st.session_state.watchlist[:40]:
        r1, r2, r3 = st.columns([2.2, 1, 1])
        with r1:
            st.markdown(f"<div style='font-weight:950;'>{sym}</div>", unsafe_allow_html=True)
        with r2:
            if st.button("Open", key=f"open_{sym}", use_container_width=True):
                st.session_state.selected = sym
                st.query_params["page"] = "analysis"
                st.rerun()
        with r3:
            if st.button("Remove", key=f"rm_{sym}", use_container_width=True):
                if sym in st.session_state.watchlist and len(st.session_state.watchlist) > 1:
                    st.session_state.watchlist.remove(sym)
                    if st.session_state.selected == sym:
                        st.session_state.selected = st.session_state.watchlist[0]
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Fixed bottom nav
# -----------------------------
active = {"analysis":"", "strategies":"", "watchlist":""}
active[page] = "active"

components.html(
    f"""
<div class="fixed-nav">
  <div class="navrow">
    <button class="navbtn {active['analysis']}" onclick="window.location.search='?page=analysis'">📊 Analysis</button>
    <button class="navbtn {active['strategies']}" onclick="window.location.search='?page=strategies'">🧠 Strategies</button>
    <button class="navbtn {active['watchlist']}" onclick="window.location.search='?page=watchlist'">⭐ Watchlist</button>
  </div>
</div>
""",
    height=72,
)
