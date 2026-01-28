import os
import math
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# Optional Finnhub
try:
    import finnhub
except Exception:
    finnhub = None

# -----------------------------
# Page / App config
# -----------------------------
st.set_page_config(page_title="Credit Spread Trend Scanner", layout="wide")

st.title("📈 Trend Scanner → Credit Spread Builder")
st.caption("Educational only. Options trading involves substantial risk, including assignment risk.")

TZ = "America/New_York"

TIMEFRAMES = {"1m": "1", "5m": "5", "1h": "60", "1D": "D"}
LOOKBACK_DAYS = {"1": 14, "5": 60, "60": 180, "D": 730}

DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "AMD", "META", "AMZN", "GOOGL"]


# -----------------------------
# API key handling (NO retyping)
# -----------------------------
def get_finnhub_key() -> str:
    # Streamlit secrets (best)
    # In Streamlit Cloud: Settings -> Secrets -> FINNHUB_API_KEY="..."
    k = st.secrets.get("FINNHUB_API_KEY", "")
    if k:
        return str(k).strip()
    # Env var fallback
    return os.environ.get("FINNHUB_API_KEY", "").strip()


def get_finnhub_client():
    key = get_finnhub_key()
    if key and finnhub is not None:
        return finnhub.Client(api_key=key)
    return None


# -----------------------------
# Indicators / Trend
# -----------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.rolling(n).mean() / down.rolling(n).mean()
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"] = sma(out["close"], 20)
    out["SMA50"] = sma(out["close"], 50)
    out["SMA200"] = sma(out["close"], 200)
    out["RSI14"] = rsi(out["close"], 14)
    out["ATR14"] = atr(out, 14)
    return out


def classify_trend(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 220 or df["SMA200"].isna().iloc[-1]:
        return {"direction": "insufficient", "strength": "", "regime": "", "notes": "Need ~200+ bars"}
    last = df.iloc[-1]
    sma20, sma50, sma200 = last["SMA20"], last["SMA50"], last["SMA200"]
    close = float(last["close"])
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0

    if sma20 > sma50 > sma200 and close > sma20:
        direction, strength = "bullish", "strong"
    elif sma20 > sma50:
        direction, strength = "bullish", "mild"
    elif sma20 < sma50 < sma200 and close < sma20:
        direction, strength = "bearish", "strong"
    elif sma20 < sma50:
        direction, strength = "bearish", "mild"
    else:
        direction, strength = "neutral", "mild"

    if direction in ("bullish", "bearish") and (r >= 60 or r <= 40):
        regime = "trending"
    elif 45 <= r <= 55:
        regime = "range"
    else:
        regime = "pullback/transition"

    return {"direction": direction, "strength": strength, "regime": regime, "notes": f"RSI={r:.1f} Close={close:.2f}"}


def decide_bias(trend_matrix: dict) -> str:
    score = 0
    for tf, w in [("1D", 3), ("1h", 2), ("5m", 1), ("1m", 1)]:
        d = trend_matrix.get(tf, {}).get("direction", "insufficient")
        if d == "bullish":
            score += w
        if d == "bearish":
            score -= w
    if score >= 4:
        return "bullish"
    if score <= -4:
        return "bearish"
    return "neutral"


# -----------------------------
# Candles (Finnhub -> yfinance fallback) + caching
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_candles_yf(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    tkr = yf.Ticker(symbol)
    hist = tkr.history(period=period, interval=interval, auto_adjust=False)
    if hist is None or hist.empty:
        return None
    df = hist.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TZ)
    df.index.name = "time"
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_candles_finnhub(symbol: str, resolution: str, from_ts: int, to_ts: int, api_key_marker: str) -> pd.DataFrame | None:
    # api_key_marker is just to separate cache entries per key; not used directly
    client = get_finnhub_client()
    if client is None:
        return None
    res = client.stock_candles(symbol, resolution, from_ts, to_ts)
    if res.get("s") != "ok" or not res.get("t"):
        return None

    df = pd.DataFrame({"open": res["o"], "high": res["h"], "low": res["l"], "close": res["c"], "volume": res["v"]})
    df.index = pd.to_datetime(res["t"], unit="s", utc=True).tz_convert(TZ)
    df.index.name = "time"
    return df


def fetch_candles(symbol: str, resolution: str, cache_ttl: int) -> pd.DataFrame | None:
    # Update cache TTL dynamically
    fetch_candles_yf.ttl = cache_ttl
    fetch_candles_finnhub.ttl = cache_ttl

    # 1) Finnhub attempt if available
    client = get_finnhub_client()
    if client is not None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=LOOKBACK_DAYS.get(resolution, 365))
        from_ts, to_ts = int(start.timestamp()), int(end.timestamp())
        api_key_marker = "keyed"  # keeps cache keyed; doesn't reveal key
        try:
            df = fetch_candles_finnhub(symbol, resolution, from_ts, to_ts, api_key_marker)
            if df is not None and not df.empty:
                return df
        except Exception:
            # common: 403 no access
            pass

    # 2) yfinance fallback
    interval_map = {"1": "1m", "5": "5m", "60": "60m", "D": "1d"}
    interval = interval_map.get(resolution, "1d")
    if interval == "1m":
        period = "7d"
    elif interval in ("5m", "60m"):
        period = "60d"
    else:
        period = "2y"

    return fetch_candles_yf(symbol, interval, period)


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

    deltas = []
    for _, row in df.iterrows():
        K = float(row.get("strike", np.nan))
        iv = float(row.get("impliedVolatility", np.nan))
        if not np.isfinite(K) or not np.isfinite(iv):
            deltas.append(np.nan)
        else:
            deltas.append(bs_delta(S, K, T, r, iv, is_call))
    df["delta_est"] = deltas

    bid = df.get("bid", pd.Series(np.nan, index=df.index))
    ask = df.get("ask", pd.Series(np.nan, index=df.index))
    df["mid"] = np.where(np.isfinite(bid) & np.isfinite(ask) & (ask > 0), (bid + ask) / 2.0, np.nan)
    return df


def pick_expiration(tkr: yf.Ticker, dte_min: int, dte_max: int):
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


def build_vertical_credit(puts_e, calls_e, bias: str, target_delta: float, wing_steps: int):
    if bias == "bullish":
        short = _pick_short_by_delta(puts_e, target_delta, want_call=False)
        if short is None:
            return None
        long = _pick_wing(puts_e, float(short["strike"]), wing_steps, want_call=False)
        if long is None:
            return None
        credit = _safe_mid(short) - _safe_mid(long)
        width = abs(float(short["strike"]) - float(long["strike"]))
        return {
            "name": "Bull Put Spread",
            "legs": [("SELL PUT", float(short["strike"])), ("BUY PUT", float(long["strike"]))],
            "credit": credit if np.isfinite(credit) else np.nan,
            "width": width,
        }

    if bias == "bearish":
        short = _pick_short_by_delta(calls_e, target_delta, want_call=True)
        if short is None:
            return None
        long = _pick_wing(calls_e, float(short["strike"]), wing_steps, want_call=True)
        if long is None:
            return None
        credit = _safe_mid(short) - _safe_mid(long)
        width = abs(float(long["strike"]) - float(short["strike"]))
        return {
            "name": "Bear Call Spread",
            "legs": [("SELL CALL", float(short["strike"])), ("BUY CALL", float(long["strike"]))],
            "credit": credit if np.isfinite(credit) else np.nan,
            "width": width,
        }

    return None


def build_iron_condor(puts_e, calls_e, target_delta: float, wing_steps: int):
    sp = _pick_short_by_delta(puts_e, target_delta, want_call=False)
    sc = _pick_short_by_delta(calls_e, target_delta, want_call=True)
    if sp is None or sc is None:
        return None
    lp = _pick_wing(puts_e, float(sp["strike"]), wing_steps, want_call=False)
    lc = _pick_wing(calls_e, float(sc["strike"]), wing_steps, want_call=True)
    if lp is None or lc is None:
        return None

    spk, lpk = float(sp["strike"]), float(lp["strike"])
    sck, lck = float(sc["strike"]), float(lc["strike"])

    put_credit = _safe_mid(sp) - _safe_mid(lp)
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
# UI Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")

    # Watchlist editing stored in session state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

    watchlist = st.session_state.watchlist

    colA, colB = st.columns([2, 1])
    with colA:
        watch_pick = st.selectbox("Watchlist", watchlist, index=0)
    with colB:
        if st.button("Remove", use_container_width=True):
            if len(watchlist) > 1 and watch_pick in watchlist:
                watchlist.remove(watch_pick)

    new_sym = st.text_input("Add ticker (e.g., NFLX)").strip().upper()
    if st.button("Add to watchlist"):
        if new_sym and new_sym not in watchlist:
            watchlist.append(new_sym)

    symbol = st.text_input("Or type a ticker", value=watch_pick).strip().upper()

    st.divider()

    dte_preset = st.selectbox("DTE window", ["7–21", "30–45", "Custom"])
    if dte_preset == "7–21":
        dte_min, dte_max = 7, 21
    elif dte_preset == "30–45":
        dte_min, dte_max = 30, 45
    else:
        dte_min = st.number_input("DTE min", min_value=1, max_value=365, value=7, step=1)
        dte_max = st.number_input("DTE max", min_value=1, max_value=365, value=21, step=1)
        if dte_min > dte_max:
            dte_min, dte_max = dte_max, dte_min

    mode = st.selectbox("Mode", ["Auto", "Bull Put", "Bear Call", "Iron Condor"])
    target_delta = st.slider("Target delta", min_value=0.10, max_value=0.35, value=0.20, step=0.01)
    wing_steps = st.slider("Wing steps", min_value=1, max_value=8, value=3, step=1)
    r_rate = st.number_input("Risk-free rate (r)", min_value=0.0, max_value=0.20, value=0.04, step=0.005)
    cache_ttl = st.slider("Cache TTL (seconds)", min_value=0, max_value=1800, value=300, step=60)

    show_charts = st.checkbox("Show charts", value=True)
    run = st.button("🚀 Run Scan", type="primary", use_container_width=True)

    st.divider()
    st.caption("Finnhub candles are optional. If your Finnhub plan blocks candles, the app falls back to yfinance automatically.")


# -----------------------------
# Main run
# -----------------------------
if run:
    if not symbol:
        st.error("Enter a ticker.")
        st.stop()

    st.subheader(f"{symbol} — Multi-timeframe trend scan")

    trend_matrix = {}
    latest_close = None

    # trend scan
    for tf_name, res in TIMEFRAMES.items():
        df = fetch_candles(symbol, res, cache_ttl=cache_ttl)
        key = tf_name
        if df is None or df.empty:
            trend_matrix[key] = {"direction": "insufficient", "strength": "", "regime": "", "notes": "No data"}
            continue

        df = compute_features(df)
        summ = classify_trend(df)
        trend_matrix[key] = summ
        latest_close = float(df["close"].iloc[-1])

        with st.expander(f"{tf_name} — {summ['direction']} {summ['strength']} | {summ['regime']} ({summ['notes']})", expanded=(tf_name == "1D")):
            st.write(summ)
            if show_charts:
                # Simple matplotlib chart
                import matplotlib.pyplot as plt
                dfp = df.tail(250).copy()
                fig = plt.figure(figsize=(10, 3.5))
                plt.plot(dfp.index, dfp["close"], label="Close")
                for c in ["SMA20", "SMA50", "SMA200"]:
                    if c in dfp.columns:
                        plt.plot(dfp.index, dfp[c], label=c)
                plt.title(f"{symbol} {tf_name}")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

    auto_bias = decide_bias(trend_matrix)
    st.info(f"Auto bias (weighted): **{auto_bias.upper()}**")

    # options
    st.subheader("Options chain → strategy builder")

    tkr = yf.Ticker(symbol)
    spot = np.nan
    try:
        spot = float(getattr(tkr, "fast_info", {}).get("last_price", np.nan))
    except Exception:
        pass
    if not np.isfinite(spot):
        h = tkr.history(period="5d")
        if not h.empty:
            spot = float(h["Close"].iloc[-1])
    if not np.isfinite(spot):
        spot = latest_close

    st.write(f"Spot: **{spot:.2f}**")

    expiry = pick_expiration(tkr, int(dte_min), int(dte_max))
    if not expiry:
        st.error("No options expirations found for this symbol (or Yahoo data unavailable).")
        st.stop()

    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    st.write(f"Chosen expiry: **{expiry}** (DTE={(ed - date.today()).days})")

    opt = tkr.option_chain(expiry)
    calls_e = enrich_chain_with_delta(opt.calls, S=spot, expiry=expiry, r=float(r_rate), is_call=True)
    puts_e = enrich_chain_with_delta(opt.puts, S=spot, expiry=expiry, r=float(r_rate), is_call=False)

    # mode -> bias
    if mode == "Auto":
        bias = auto_bias
    elif mode == "Bull Put":
        bias = "bullish"
    elif mode == "Bear Call":
        bias = "bearish"
    else:
        bias = "neutral"

    # build recommendation
    if mode == "Iron Condor" or bias == "neutral":
        strat = build_iron_condor(puts_e, calls_e, float(target_delta), int(wing_steps))
    else:
        strat = build_vertical_credit(puts_e, calls_e, bias, float(target_delta), int(wing_steps))

    if not strat:
        st.error("Could not construct a strategy (missing IV / bid-ask mids / strikes). Try a different DTE or delta.")
        st.stop()

    st.success(f"Recommended: **{strat['name']}**")

    st.markdown("**Legs**")
    for leg in strat["legs"]:
        st.write(f"- {leg[0]} {leg[1]}")

    # metrics
    credit = strat.get("credit", np.nan)
    width = strat.get("width", np.nan)
    st.markdown("**Estimates (mid-based)**")
    if np.isfinite(credit):
        st.write(f"- Est credit: **{credit:.2f}**")
    else:
        st.write("- Est credit: unavailable (missing bid/ask mids)")

    if np.isfinite(width):
        st.write(f"- Width: **{width:.2f}**")

    if strat["name"] == "Iron Condor":
        max_loss = strat.get("max_loss", np.nan)
        if np.isfinite(max_loss):
            st.write(f"- Est max loss/share: **{max_loss:.2f}** (x100 per condor)")
        if np.isfinite(credit):
            st.write(f"- Put credit: {strat.get('put_credit', np.nan):.2f} | Call credit: {strat.get('call_credit', np.nan):.2f}")
    else:
        if np.isfinite(credit) and np.isfinite(width):
            st.write(f"- Est max loss/share: **{(width - credit):.2f}** (x100 per spread)")

    st.caption("Educational only — this is not investment advice.")
else:
    st.write("Set your settings in the sidebar, then tap **Run Scan**.")