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
import requests

# =========================
# App config
# =========================
st.set_page_config(page_title="Trend + Credit Spreads", layout="wide")

TZ = "America/New_York"
DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "AMD", "META", "AMZN", "GOOGL"]

# =========================
# Styling (dark-friendly, minimal)
# =========================
st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 1.0rem; padding-bottom: 2.5rem; }
.small-muted { opacity: 0.80; font-size: 0.92rem; }
.card { border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 14px; background: rgba(255,255,255,0.03); }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 0.75rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Trend Scanner → Credit Spread Builder")
st.caption("Educational only. Options trading involves substantial risk (assignment, gap risk, liquidity).")

# =========================
# Trend indicators
# =========================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.rolling(n).mean() / down.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA20"] = sma(d["close"], 20)
    d["SMA50"] = sma(d["close"], 50)
    d["SMA200"] = sma(d["close"], 200)
    d["RSI14"] = rsi(d["close"], 14)
    return d

def classify_trend_adaptive(df: pd.DataFrame) -> dict:
    """
    Adaptive trend:
    - Uses SMA200 if available; otherwise falls back to SMA20/50.
    Prevents 4H from failing due to missing 200+ bars.
    """
    if df is None or df.empty or len(df) < 60:
        return {"direction": "insufficient", "strength": 0, "regime": "", "notes": "Need ~60+ bars"}

    last = df.iloc[-1]
    close = float(last["close"])
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    s20 = last.get("SMA20", np.nan)
    s50 = last.get("SMA50", np.nan)
    s200 = last.get("SMA200", np.nan)

    has_50 = pd.notna(s50)
    has_200 = pd.notna(s200)

    direction = "neutral"
    strength = 45

    if has_50 and pd.notna(s20):
        if has_200:
            if s20 > s50 > s200 and close > s20:
                direction, strength = "bullish", 80
            elif s20 > s50:
                direction, strength = "bullish", 60
            elif s20 < s50 < s200 and close < s20:
                direction, strength = "bearish", 80
            elif s20 < s50:
                direction, strength = "bearish", 60
            else:
                direction, strength = "neutral", 45
        else:
            # Shorter-term logic
            if s20 > s50 and close > s20:
                direction, strength = "bullish", 65
            elif s20 > s50:
                direction, strength = "bullish", 55
            elif s20 < s50 and close < s20:
                direction, strength = "bearish", 65
            elif s20 < s50:
                direction, strength = "bearish", 55
            else:
                direction, strength = "neutral", 45

    if direction in ("bullish", "bearish") and (r >= 60 or r <= 40):
        regime = "trending"
        strength = min(90, strength + 10)
    elif 45 <= r <= 55:
        regime = "range"
    else:
        regime = "transition"

    notes = f"RSI={r:.1f} Close={close:.2f}" + ("" if has_200 else " (no SMA200)")
    return {"direction": direction, "strength": strength, "regime": regime, "notes": notes}

def decide_bias(trend_matrix: dict) -> str:
    # weights: 1D strongest, 4h medium, 15m light
    score = 0
    for tf, w in [("1D", 3), ("4h", 2), ("15m", 1)]:
        d = trend_matrix.get(tf, {}).get("direction", "insufficient")
        if d == "bullish":
            score += w
        if d == "bearish":
            score -= w
    if score >= 3:
        return "bullish"
    if score <= -3:
        return "bearish"
    return "neutral"

# =========================
# Candles (yfinance + backoff)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_yf(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    tkr = yf.Ticker(symbol)
    for attempt in range(5):
        try:
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
        except YFRateLimitError:
            time.sleep((2 ** attempt) + random.uniform(0, 0.7))
        except Exception:
            return None
    return None

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum(),
        }
    ).dropna()
    out.index.name = "time"
    return out

def fetch_timeframes(symbol: str):
    df_15m = fetch_yf(symbol, "15m", "60d")
    df_4h = resample_ohlcv(df_15m, "4H") if df_15m is not None and not df_15m.empty else None
    df_1d = fetch_yf(symbol, "1d", "3y")
    return {"15m": df_15m, "4h": df_4h, "1D": df_1d}

# =========================
# Yahoo Options (DIRECT) — fixes Streamlit Cloud empty expirations
# =========================
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
                  "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

@st.cache_data(ttl=600, show_spinner=False)
def yahoo_get_options_json(symbol: str, date_ts: int | None = None) -> dict | None:
    base = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
    url = base if date_ts is None else f"{base}?date={int(date_ts)}"
    for attempt in range(6):
        try:
            r = requests.get(url, headers=YAHOO_HEADERS, timeout=12)
            if r.status_code == 429:
                time.sleep((2 ** attempt) + random.uniform(0, 0.9))
                continue
            if r.status_code >= 400:
                return None
            return r.json()
        except Exception:
            time.sleep(0.6 + random.uniform(0, 0.5))
    return None

def yahoo_expirations_with_ts(symbol: str) -> list[tuple[str, int]]:
    j = yahoo_get_options_json(symbol, None)
    try:
        ts_list = j["optionChain"]["result"][0]["expirationDates"]
        out = []
        for ts in ts_list:
            ds = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
            out.append((ds, int(ts)))
        return out
    except Exception:
        return []

def pick_expiration_in_window_ts(exps: list[tuple[str, int]], dte_min: int, dte_max: int) -> tuple[str, int] | None:
    if not exps:
        return None
    today = date.today()
    candidates = []
    for ds, ts in exps:
        try:
            ed = datetime.strptime(ds, "%Y-%m-%d").date()
            dte = (ed - today).days
            if dte_min <= dte <= dte_max:
                candidates.append((dte, ds, ts))
        except Exception:
            pass
    if not candidates:
        return None
    target = int(round((dte_min + dte_max) / 2))
    candidates.sort(key=lambda x: abs(x[0] - target))
    return (candidates[0][1], candidates[0][2])

def yahoo_option_chain(symbol: str, expiry_ts: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    j = yahoo_get_options_json(symbol, expiry_ts)
    try:
        opt = j["optionChain"]["result"][0]["options"][0]
        calls = pd.DataFrame(opt.get("calls", []))
        puts = pd.DataFrame(opt.get("puts", []))
        if calls.empty and puts.empty:
            return None
        return calls, puts
    except Exception:
        return None

# =========================
# Options + spreads
# =========================
def bs_delta(S, K, T, r, sigma, is_call: bool):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

def enrich_chain_with_delta(chain_df: pd.DataFrame, S: float, expiry: str, r: float, is_call: bool):
    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = max((ed - date.today()).days / 365.0, 1e-6)

    df = chain_df.copy()

    # Normalize likely Yahoo field names to expected ones
    # Yahoo returns bid/ask/strike/impliedVolatility already; keep safe.
    if "impliedVolatility" not in df.columns:
        df["impliedVolatility"] = np.nan
    if "bid" not in df.columns:
        df["bid"] = np.nan
    if "ask" not in df.columns:
        df["ask"] = np.nan
    if "strike" not in df.columns:
        df["strike"] = np.nan

    bid = df["bid"]
    ask = df["ask"]
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

def pick_short_by_delta(df: pd.DataFrame, target_delta: float, want_call: bool):
    d_target = target_delta if want_call else -target_delta
    x = df.dropna(subset=["delta_est", "strike"]).copy()
    if x.empty:
        return None
    x["err"] = (x["delta_est"] - d_target).abs()
    return x.sort_values("err").iloc[0]

def pick_wing_by_width(df: pd.DataFrame, short_strike: float, wing_width: float, want_call: bool):
    x = df.dropna(subset=["strike"]).copy()
    strikes = np.array(sorted(x["strike"].unique()))
    if len(strikes) < 2:
        return None

    if want_call:
        target = short_strike + wing_width
        cand = strikes[strikes > short_strike]
        if len(cand) == 0:
            return None
        chosen = cand[np.argmin(np.abs(cand - target))]
    else:
        target = short_strike - wing_width
        cand = strikes[strikes < short_strike]
        if len(cand) == 0:
            return None
        chosen = cand[np.argmin(np.abs(cand - target))]

    row = x[x["strike"] == chosen]
    return row.iloc[0] if not row.empty else None

def safe_mid(row):
    m = row.get("mid", np.nan)
    return float(m) if np.isfinite(m) else np.nan

def build_bull_put(puts_e, target_delta: float, wing_width: float):
    short = pick_short_by_delta(puts_e, target_delta, want_call=False)
    if short is None:
        return None
    long = pick_wing_by_width(puts_e, float(short["strike"]), wing_width, want_call=False)
    if long is None:
        return None
    credit = safe_mid(short) - safe_mid(long)
    width = abs(float(short["strike"]) - float(long["strike"]))
    max_loss = (width - credit) if np.isfinite(credit) else np.nan
    return {"name": "Bull Put Spread", "legs": [("SELL PUT", float(short["strike"])), ("BUY PUT", float(long["strike"]))],
            "credit": credit, "width": width, "max_loss": max_loss}

def build_bear_call(calls_e, target_delta: float, wing_width: float):
    short = pick_short_by_delta(calls_e, target_delta, want_call=True)
    if short is None:
        return None
    long = pick_wing_by_width(calls_e, float(short["strike"]), wing_width, want_call=True)
    if long is None:
        return None
    credit = safe_mid(short) - safe_mid(long)
    width = abs(float(long["strike"]) - float(short["strike"]))
    max_loss = (width - credit) if np.isfinite(credit) else np.nan
    return {"name": "Bear Call Spread", "legs": [("SELL CALL", float(short["strike"])), ("BUY CALL", float(long["strike"]))],
            "credit": credit, "width": width, "max_loss": max_loss}

def build_iron_condor(puts_e, calls_e, target_delta: float, wing_width: float):
    sp = pick_short_by_delta(puts_e, target_delta, want_call=False)
    sc = pick_short_by_delta(calls_e, target_delta, want_call=True)
    if sp is None or sc is None:
        return None
    lp = pick_wing_by_width(puts_e, float(sp["strike"]), wing_width, want_call=False)
    lc = pick_wing_by_width(calls_e, float(sc["strike"]), wing_width, want_call=True)
    if lp is None or lc is None:
        return None

    spk, lpk = float(sp["strike"]), float(lp["strike"])
    sck, lck = float(sc["strike"]), float(lc["strike"])

    put_credit = safe_mid(sp) - safe_mid(lp)
    call_credit = safe_mid(sc) - safe_mid(lc)
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

def build_strategy_for_bias(mode: str, auto_bias: str, puts_e, calls_e, target_delta: float, wing_width: float):
    if mode == "Auto":
        bias = auto_bias
    elif mode == "Bull Put":
        bias = "bullish"
    elif mode == "Bear Call":
        bias = "bearish"
    else:
        bias = "neutral"

    if mode == "Iron Condor" or bias == "neutral":
        return build_iron_condor(puts_e, calls_e, target_delta, wing_width)
    if bias == "bullish":
        return build_bull_put(puts_e, target_delta, wing_width)
    if bias == "bearish":
        return build_bear_call(calls_e, target_delta, wing_width)
    return None

# =========================
# Watchlist state
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

# =========================
# MAIN CONTROLS (NO SIDEBAR)
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

row1 = st.columns([2.2, 1.4, 1.2, 1.2])

with row1[0]:
    symbol = st.selectbox("Ticker", st.session_state.watchlist, index=0)
    add_row = st.columns([1.6, 0.8, 0.8])
    with add_row[0]:
        new_sym = st.text_input("Add ticker", placeholder="NFLX").strip().upper()
    with add_row[1]:
        if st.button("Add"):
            if new_sym and new_sym not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_sym)
                st.success(f"Added {new_sym}")
                st.rerun()
    with add_row[2]:
        if st.button("Remove"):
            if len(st.session_state.watchlist) > 1 and symbol in st.session_state.watchlist:
                st.session_state.watchlist.remove(symbol)
                st.rerun()

with row1[1]:
    mode = st.selectbox("Strategy mode", ["Auto", "Bull Put", "Bear Call", "Iron Condor"])
    target_delta = st.slider("Target delta", 0.10, 0.35, 0.20, 0.01)

with row1[2]:
    wing_width = st.slider("Wing width ($)", 1.00, 2.50, 1.50, 0.05)
    r_rate = st.number_input("Risk-free r", 0.0, 0.20, 0.04, 0.005)

with row1[3]:
    show_charts = st.checkbox("Show charts", value=True)
    run = st.button("🚀 Run", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if not run:
    st.write("Tap **Run** to scan trends and generate Daily/Weekly spread ideas.")
    st.stop()

# =========================
# DATA + TREND SCAN
# =========================
with st.spinner("Fetching candles…"):
    frames = fetch_timeframes(symbol)

trend_matrix = {}
latest_close = None

st.subheader(f"{symbol} — Trend scan (15m / 4h / 1D)")

for tf_name in ["15m", "4h", "1D"]:
    df = frames.get(tf_name)
    if df is None or df.empty:
        trend_matrix[tf_name] = {"direction": "insufficient", "strength": 0, "regime": "", "notes": "No data"}
        with st.expander(f"{tf_name} — insufficient (no data)", expanded=True):
            st.write(trend_matrix[tf_name])
        continue

    df_feat = compute_features(df)
    summ = classify_trend_adaptive(df_feat)
    trend_matrix[tf_name] = summ
    latest_close = float(df_feat["close"].iloc[-1])

    with st.expander(
        f"{tf_name} — {summ['direction']} ({summ['strength']}%) | {summ['regime']} | {summ['notes']}",
        expanded=True,
    ):
        if show_charts:
            import matplotlib.pyplot as plt

            dfp = df_feat.tail(260).copy()
            fig = plt.figure(figsize=(10, 3.6))
            plt.plot(dfp.index, dfp["close"], label="Close")
            for col in ["SMA20", "SMA50", "SMA200"]:
                if col in dfp.columns and dfp[col].notna().any():
                    plt.plot(dfp.index, dfp[col], label=col)
            plt.title(f"{symbol} {tf_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        st.write(summ)

auto_bias = decide_bias(trend_matrix)
st.info(f"Auto bias (weighted): **{auto_bias.upper()}**")

# =========================
# OPTIONS: Daily + Weekly ideas (Yahoo direct)
# =========================
st.subheader("Options chain → spread ideas (Daily + Weekly)")

# spot
spot = np.nan
try:
    # yfinance spot (works fine even when options is blocked)
    tkr = yf.Ticker(symbol)
    spot = float(getattr(tkr, "fast_info", {}).get("last_price", np.nan))
except Exception:
    pass
if not np.isfinite(spot):
    spot = float(latest_close) if latest_close is not None else np.nan

if not np.isfinite(spot):
    st.error("Spot price unavailable.")
    st.stop()

st.write(f"Spot: **{spot:.2f}**")

# expirations (direct Yahoo)
exps_ts = yahoo_expirations_with_ts(symbol)
if not exps_ts:
    st.error("No options expirations available right now (Yahoo may be rate-limiting). Try again in ~60 seconds.")
    st.stop()

def render_spread_idea(title: str, dte_min: int, dte_max: int):
    st.markdown(
        f"<div class='card'><b>{title}</b><div class='small-muted'>DTE window: {dte_min}–{dte_max}</div>",
        unsafe_allow_html=True,
    )

    picked = pick_expiration_in_window_ts(exps_ts, dte_min, dte_max)
    if not picked:
        st.write("No expiration found in this DTE window for this symbol.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    expiry, expiry_ts = picked
    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    st.write(f"Chosen expiry: **{expiry}** (DTE={(ed - date.today()).days})")

    chain = yahoo_option_chain(symbol, expiry_ts)
    if not chain:
        st.write("Options chain unavailable for this expiry (Yahoo may be throttling). Try again soon.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    calls_raw, puts_raw = chain

    calls_e = enrich_chain_with_delta(calls_raw, S=spot, expiry=expiry, r=float(r_rate), is_call=True)
    puts_e = enrich_chain_with_delta(puts_raw, S=spot, expiry=expiry, r=float(r_rate), is_call=False)

    strat = build_strategy_for_bias(mode, auto_bias, puts_e, calls_e, float(target_delta), float(wing_width))
    if not strat:
        st.write("Could not construct a spread (missing IV/mids/strikes). Try another delta or wing width.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.success(f"Recommended: **{strat['name']}**")

    st.markdown("**Legs**")
    for leg in strat["legs"]:
        st.write(f"- {leg[0]} {leg[1]}")

    credit = strat.get("credit", np.nan)
    max_loss = strat.get("max_loss", np.nan)

    st.markdown("**Estimates (mid-based)**")
    st.write(f"- Est credit: **{credit:.2f}**" if np.isfinite(credit) else "- Est credit: unavailable")
    st.write(f"- Est max loss/share: **{max_loss:.2f}** (x100/contract)" if np.isfinite(max_loss) else "- Est max loss: unavailable")

    if strat["name"] == "Iron Condor":
        pc = strat.get("put_credit", np.nan)
        cc = strat.get("call_credit", np.nan)
        if np.isfinite(pc) and np.isfinite(cc):
            st.write(f"- Put credit: {pc:.2f} | Call credit: {cc:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    render_spread_idea("📅 Daily Spread Idea", 0, 2)
with colB:
    render_spread_idea("🗓️ Weekly Spread Idea", 5, 12)

st.caption("Educational only — not investment advice.")