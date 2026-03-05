import os
import math
import time
import random
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import requests
from scipy.stats import norm

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Trend → Greeks → Options + Spreads", layout="wide")
TZ = "America/New_York"
DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "AMD", "META", "AMZN", "GOOGL"]
MD_BASE = "https://api.marketdata.app/v1"

st.title("📈 Trend → Greeks → Options + Credit Spreads")
st.caption("Educational only. Options trading involves substantial risk (assignment, gaps, liquidity).")

st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 1.0rem; padding-bottom: 2.0rem; }
.card { border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 14px; background: rgba(255,255,255,0.03); }
.small-muted { opacity: 0.82; font-size: 0.92rem; }
.pill { display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.05); margin-right: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# MarketData token + HTTP
# =========================================================
def get_marketdata_token() -> str:
    try:
        tok = str(st.secrets.get("MARKETDATA_TOKEN", "")).strip()
    except Exception:
        tok = ""
    if tok:
        return tok
    return os.environ.get("MARKETDATA_TOKEN", "").strip()

def md_headers() -> dict:
    tok = get_marketdata_token()
    h = {"Accept": "application/json"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def md_get_json(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, headers=md_headers(), params=params, timeout=12)
        if r.status_code in (401, 403):
            return {"s": "error", "errmsg": "Auth failed. Check MARKETDATA_TOKEN in Streamlit Secrets."}
        if r.status_code >= 400:
            return {"s": "error", "errmsg": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"s": "error", "errmsg": f"Request error: {e}"}

@st.cache_data(ttl=1800, show_spinner=False)
def md_options_expirations(symbol: str) -> list[str] | None:
    j = md_get_json(f"{MD_BASE}/options/expirations/{symbol}/")
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None
    exps = j.get("expirations", [])
    return [str(x) for x in exps] if isinstance(exps, list) else None

@st.cache_data(ttl=600, show_spinner=False)
def md_option_chain(symbol: str, expiration: str, side: str) -> pd.DataFrame | None:
    params = {"expiration": expiration, "side": side}  # "call" | "put"
    j = md_get_json(f"{MD_BASE}/options/chain/{symbol}/", params=params)
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None

    cols = {k: v for k, v in j.items() if isinstance(v, list)}
    if not cols:
        return None
    df = pd.DataFrame(cols)

    # normalize contract symbol
    if "optionSymbol" not in df.columns:
        for alt in ("symbol", "contractSymbol", "option_symbol", "option", "id"):
            if alt in df.columns:
                df["optionSymbol"] = df[alt]
                break

    for c in ("strike", "bid", "ask", "mid", "iv", "delta"):
        if c not in df.columns:
            df[c] = np.nan
    for c in ("strike", "bid", "ask", "mid", "iv", "delta"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # mid from bid/ask if missing
    if not df["mid"].notna().any():
        df["mid"] = np.where(
            np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["ask"] > 0),
            (df["bid"] + df["ask"]) / 2.0,
            np.nan,
        )
    return df

@st.cache_data(ttl=90, show_spinner=False)
def md_option_quote(option_symbol: str) -> dict | None:
    j = md_get_json(f"{MD_BASE}/options/quotes/{option_symbol}/")
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None

    def last_val(k, default=np.nan):
        v = j.get(k, default)
        if isinstance(v, list) and v:
            return v[-1]
        return v

    bid_v = last_val("bid", np.nan)
    ask_v = last_val("ask", np.nan)
    mid_v = last_val("mid", np.nan)
    iv_v = last_val("iv", np.nan)
    delta_v = last_val("delta", np.nan)
    last_v = last_val("last", np.nan)

    bid = float(bid_v) if np.isfinite(bid_v) else np.nan
    ask = float(ask_v) if np.isfinite(ask_v) else np.nan
    mid = float(mid_v) if np.isfinite(mid_v) else ((bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else np.nan)
    iv = float(iv_v) if np.isfinite(iv_v) else np.nan
    delta = float(delta_v) if np.isfinite(delta_v) else np.nan
    last = float(last_v) if np.isfinite(last_v) else np.nan
    return {"bid": bid, "ask": ask, "mid": mid, "iv": iv, "delta": delta, "last": last}

# =========================================================
# Candles: yfinance (charts)
# =========================================================
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
            time.sleep((2 ** attempt) + random.uniform(0, 0.8))
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

# =========================================================
# Indicators / Trend
# =========================================================
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
    if df is None or df.empty or len(df) < 80:
        n = 0 if df is None else len(df)
        return {"direction": "insufficient", "strength": 0, "regime": "", "notes": f"Need ~80+ bars (have {n})"}

    last = df.iloc[-1]
    close = float(last["close"])
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    s20 = last.get("SMA20", np.nan)
    s50 = last.get("SMA50", np.nan)
    s200 = last.get("SMA200", np.nan)
    has_200 = pd.notna(s200)

    direction = "neutral"
    strength = 45

    if pd.notna(s20) and pd.notna(s50):
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

# =========================================================
# Recommendation engine (trend + greeks)
# =========================================================
def pick_expiration_in_dte_window(expirations: list[str], dte_min: int, dte_max: int) -> str | None:
    today = date.today()
    candidates = []
    for e in expirations:
        try:
            ed = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (ed - today).days
            if dte_min <= dte <= dte_max:
                candidates.append((dte, e))
        except Exception:
            pass
    if not candidates:
        return None
    target = int(round((dte_min + dte_max) / 2))
    candidates.sort(key=lambda x: abs(x[0] - target))
    return candidates[0][1]

def auto_params_from_trend(auto_bias: str, strength: int, regime: str):
    """
    Converts trend into recommended DTE + delta.
    """
    # Delta: stronger trends = a bit higher delta (closer), ranges = lower delta (farther)
    if regime == "range":
        delta = 0.15
    elif strength >= 80:
        delta = 0.25
    elif strength >= 60:
        delta = 0.20
    else:
        delta = 0.18

    # DTE: range likes more time; short-term trend can be tighter
    if regime == "range":
        dte = (30, 45)
    else:
        dte = (7, 21)

    # Single-leg preference
    # bullish -> BUY CALL (or SELL PUT); bearish -> BUY PUT (or SELL CALL); neutral -> SELL premium / condor
    if auto_bias == "bullish":
        single = {"side": "call", "action": "BUY"}
        spread = "Bull Put"
    elif auto_bias == "bearish":
        single = {"side": "put", "action": "BUY"}
        spread = "Bear Call"
    else:
        single = {"side": "call", "action": "SELL"}  # default sell call as placeholder; user can switch
        spread = "Iron Condor"

    return {"target_delta": delta, "dte_min": dte[0], "dte_max": dte[1], "single_pref": single, "spread_pref": spread}

def select_candidate_symbols(chain: pd.DataFrame, spot: float, side: str, max_candidates: int = 18) -> pd.DataFrame:
    """
    Cut down the chain to a small set near-the-money/OTM to avoid tons of quote calls.
    """
    df = chain.dropna(subset=["strike", "optionSymbol"]).copy()
    df = df[np.isfinite(df["strike"])]
    if df.empty:
        return df

    if side == "call":
        # prefer OTM calls
        df["otm_dist"] = (df["strike"] - spot).clip(lower=0)
        df = df.sort_values(["otm_dist", "strike"]).head(max_candidates)
    else:
        # prefer OTM puts
        df["otm_dist"] = (spot - df["strike"]).clip(lower=0)
        df = df.sort_values(["otm_dist", "strike"], ascending=[False, False]).head(max_candidates)

    return df

def attach_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each optionSymbol, fetch quote (bid/ask/mid/iv/delta).
    This is where entitlement matters.
    """
    out = df.copy()
    bids, asks, mids, ivs, deltas = [], [], [], [], []
    for sym in out["optionSymbol"].astype(str).tolist():
        q = md_option_quote(sym)
        if not q:
            bids.append(np.nan); asks.append(np.nan); mids.append(np.nan); ivs.append(np.nan); deltas.append(np.nan)
        else:
            bids.append(q["bid"]); asks.append(q["ask"]); mids.append(q["mid"]); ivs.append(q["iv"]); deltas.append(q["delta"])
    out["q_bid"] = bids
    out["q_ask"] = asks
    out["q_mid"] = mids
    out["q_iv"] = ivs
    out["q_delta"] = deltas
    return out

def recommend_single_leg(chain: pd.DataFrame, spot: float, side: str, action: str, target_delta: float) -> pd.DataFrame:
    """
    Return top single-leg recommendations ranked by delta closeness, then mid.
    Works best when quotes return delta.
    Fallback: use percent OTM when delta missing.
    """
    base = select_candidate_symbols(chain, spot, side, max_candidates=18)
    if base.empty:
        return base

    q = attach_quotes(base)
    q = q.dropna(subset=["q_mid"], how="all")  # if no mids at all, it'll be empty

    if q.empty:
        return q

    # delta-based ranking if available
    if q["q_delta"].notna().any():
        desired = target_delta if side == "call" else -target_delta
        q["score"] = (q["q_delta"] - desired).abs()
    else:
        # percent OTM fallback
        if side == "call":
            q["score"] = ((q["strike"] / spot) - 1.0).abs()
        else:
            q["score"] = (1.0 - (q["strike"] / spot)).abs()

    q = q.sort_values(["score", "q_mid"]).head(6)

    # add some nice columns for display
    q["idea"] = f"{action} " + ( "CALL" if side=="call" else "PUT" )
    return q[["idea", "strike", "optionSymbol", "q_mid", "q_delta", "q_iv"]].reset_index(drop=True)

def pick_wing_strike(strikes: np.ndarray, short_strike: float, wing_width: float, want_call: bool) -> float | None:
    if len(strikes) < 2:
        return None
    if want_call:
        target = short_strike + wing_width
        cand = strikes[strikes > short_strike]
        if len(cand) == 0:
            return None
        return float(cand[np.argmin(np.abs(cand - target))])
    else:
        target = short_strike - wing_width
        cand = strikes[strikes < short_strike]
        if len(cand) == 0:
            return None
        return float(cand[np.argmin(np.abs(cand - target))])

def quote_mid(sym: str) -> float | None:
    q = md_option_quote(sym)
    if not q:
        return None
    return float(q["mid"]) if np.isfinite(q["mid"]) else None

def build_vertical_from_chain(chain: pd.DataFrame, spot: float, side: str, target_delta: float, wing_width: float, bearish_call: bool):
    """
    side: "put" or "call"
    bearish_call=True means bear call spread, else bull put spread.
    """
    df = chain.dropna(subset=["strike", "optionSymbol"]).copy()
    df = df[np.isfinite(df["strike"])]
    if df.empty:
        return None

    # prefer OTM region
    if side == "call":
        df = df.sort_values("strike")
        df = df[df["strike"] >= spot].head(30) if (df["strike"] >= spot).any() else df.head(30)
    else:
        df = df.sort_values("strike", ascending=False)
        df = df[df["strike"] <= spot].head(30) if (df["strike"] <= spot).any() else df.head(30)

    # Attach quotes for this reduced set
    dfq = attach_quotes(df)
    if dfq["q_mid"].notna().sum() < 2:
        return None

    # pick short by delta if available, else %OTM
    if dfq["q_delta"].notna().any():
        desired = target_delta if side == "call" else -target_delta
        dfq["err"] = (dfq["q_delta"] - desired).abs()
        short_row = dfq.sort_values("err").iloc[0]
    else:
        # fallback target 3% OTM
        tgt = spot * (1.03 if side == "call" else 0.97)
        dfq["err"] = (dfq["strike"] - tgt).abs()
        short_row = dfq.sort_values("err").iloc[0]

    short_strike = float(short_row["strike"])
    strikes = np.array(sorted(dfq["strike"].unique()))
    wing_strike = pick_wing_strike(strikes, short_strike, wing_width, want_call=(side == "call"))
    if wing_strike is None:
        return None

    short_sym = str(short_row["optionSymbol"])
    long_row = dfq[dfq["strike"] == wing_strike].head(1)
    if long_row.empty:
        return None
    long_sym = str(long_row["optionSymbol"].iloc[0])

    short_mid = quote_mid(short_sym)
    long_mid = quote_mid(long_sym)
    if short_mid is None or long_mid is None:
        return None

    credit = short_mid - long_mid
    width = abs(wing_strike - short_strike)
    max_loss = width - credit

    if bearish_call:
        name = "Bear Call Spread"
        legs = [("SELL CALL", short_strike, short_sym), ("BUY CALL", wing_strike, long_sym)]
    else:
        name = "Bull Put Spread"
        legs = [("SELL PUT", short_strike, short_sym), ("BUY PUT", wing_strike, long_sym)]

    return {"name": name, "credit": credit, "max_loss": max_loss, "legs": legs}

def build_iron_condor(puts_chain, calls_chain, spot: float, target_delta: float, wing_width: float):
    # Build put side (bull put) and call side (bear call) with same expiry
    put_spread = build_vertical_from_chain(puts_chain, spot, "put", target_delta, wing_width, bearish_call=False)
    call_spread = build_vertical_from_chain(calls_chain, spot, "call", target_delta, wing_width, bearish_call=True)
    if not put_spread or not call_spread:
        return None

    total_credit = put_spread["credit"] + call_spread["credit"]
    # max loss approx = max(widths) - total_credit
    put_w = abs(put_spread["legs"][0][1] - put_spread["legs"][1][1])
    call_w = abs(call_spread["legs"][0][1] - call_spread["legs"][1][1])
    max_w = max(put_w, call_w)
    max_loss = max_w - total_credit

    legs = put_spread["legs"] + call_spread["legs"]
    return {"name": "Iron Condor", "credit": total_credit, "max_loss": max_loss, "legs": legs,
            "put_credit": put_spread["credit"], "call_credit": call_spread["credit"]}

# =========================================================
# Session state: stop heavy reruns
# =========================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

if "results" not in st.session_state:
    st.session_state.results = None

if "last_params" not in st.session_state:
    st.session_state.last_params = None

# =========================================================
# Controls (RUN triggers refresh)
# =========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
with st.form("controls", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2.0, 1.5, 1.4, 1.2])

    with c1:
        symbol = st.selectbox("Ticker", st.session_state.watchlist, index=0, key="sym_pick")
        new_sym = st.text_input("Add ticker", placeholder="NFLX").strip().upper()
        bA, bB = st.columns(2)
        add_clicked = bA.form_submit_button("Add")
        remove_clicked = bB.form_submit_button("Remove")

    with c2:
        wing_width = st.slider("Wing width ($)", 1.00, 2.50, 1.50, 0.05)

    with c3:
        show_charts = st.checkbox("Show charts", value=True)
        show_options = st.checkbox("Show options", value=True)

    with c4:
        run = st.form_submit_button("🚀 Run", use_container_width=True)

if add_clicked:
    if new_sym and new_sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_sym)
        st.rerun()

if remove_clicked:
    if len(st.session_state.watchlist) > 1 and symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if not run and st.session_state.results is None:
    st.stop()

# =========================================================
# Refresh data ONLY when Run is pressed OR symbol changed
# =========================================================
params = {"symbol": symbol}

if run or st.session_state.last_params != params:
    with st.spinner("Refreshing data…"):
        frames = fetch_timeframes(symbol)

        trend_matrix = {}
        latest_close = None

        for tf_name in ["15m", "4h", "1D"]:
            df = frames.get(tf_name)
            if df is None or df.empty:
                trend_matrix[tf_name] = {"direction": "insufficient", "strength": 0, "regime": "", "notes": "No data"}
                continue
            df_feat = compute_features(df)
            trend_matrix[tf_name] = classify_trend_adaptive(df_feat)
            latest_close = float(df_feat["close"].iloc[-1])

        auto_bias = decide_bias(trend_matrix)
        spot = float(latest_close) if latest_close is not None else np.nan

        # "main trend" = 1D trend for strength/regime tuning
        main = trend_matrix.get("1D", {"strength": 50, "regime": "transition"})
        auto_cfg = auto_params_from_trend(auto_bias, int(main.get("strength", 50)), str(main.get("regime", "transition")))

        expirations = md_options_expirations(symbol) if get_marketdata_token() else None

        st.session_state.results = {
            "frames": frames,
            "trend": trend_matrix,
            "auto_bias": auto_bias,
            "spot": spot,
            "auto_cfg": auto_cfg,
            "expirations": expirations,
        }
        st.session_state.last_params = params

# =========================================================
# Render results (NO heavy calls)
# =========================================================
R = st.session_state.results
frames = R["frames"]
trend_matrix = R["trend"]
auto_bias = R["auto_bias"]
spot = R["spot"]
auto_cfg = R["auto_cfg"]
expirations = R["expirations"]

st.subheader(f"{symbol} — Trend scan (15m / 4h / 1D)")
st.markdown(
    f"<span class='pill'>Bias: <b>{auto_bias.upper()}</b></span>"
    f"<span class='pill'>Auto Δ: <b>{auto_cfg['target_delta']:.2f}</b></span>"
    f"<span class='pill'>Auto DTE: <b>{auto_cfg['dte_min']}-{auto_cfg['dte_max']}</b></span>",
    unsafe_allow_html=True
)

for tf_name in ["15m", "4h", "1D"]:
    summ = trend_matrix.get(tf_name, {})
    df = frames.get(tf_name)
    with st.expander(f"{tf_name} — {summ.get('direction','?')} ({summ.get('strength',0)}%) | {summ.get('regime','')} | {summ.get('notes','')}", expanded=(tf_name=="1D")):
        if show_charts and df is not None and not df.empty:
            import matplotlib.pyplot as plt
            df_feat = compute_features(df)
            dfp = df_feat.tail(260).copy()
            fig = plt.figure(figsize=(10, 3.6))
            plt.plot(dfp.index, dfp["close"], label="Close")
            for col in ["SMA20", "SMA50", "SMA200"]:
                if dfp[col].notna().any():
                    plt.plot(dfp.index, dfp[col], label=col)
            plt.title(f"{symbol} {tf_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
        st.write(summ)

st.info(f"Auto bias (weighted): **{auto_bias.upper()}**")
st.write(f"Spot: **{spot:.2f}**" if np.isfinite(spot) else "Spot: unavailable")

# =========================================================
# Options + recommendations (tabs) - only small quote calls
# =========================================================
if not show_options:
    st.stop()

if not get_marketdata_token():
    st.error("MARKETDATA_TOKEN missing. Add it in Streamlit Secrets to use MarketData options.")
    st.stop()

if not expirations:
    st.error("No expirations returned. Usually means options entitlement/OPRA not enabled, or token blocked.")
    st.stop()

st.subheader("🧠 Auto Recommendations (Trend + Greeks)")

tab1, tab2, tab3 = st.tabs(["Single Options (Auto)", "Credit Spreads (Auto)", "Manual Ticket"])

# ---------------------------
# Tab 1: Auto Single Options
# ---------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("These are **auto-picked** using trend bias + target delta. (Best results when MarketData quotes return delta.)")

    dte_min, dte_max = auto_cfg["dte_min"], auto_cfg["dte_max"]
    expiry = pick_expiration_in_dte_window(expirations, dte_min, dte_max)
    if not expiry:
        st.warning("No expiration found in the auto DTE window.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        pref = auto_cfg["single_pref"]
        side = pref["side"]  # "call" or "put"
        action = pref["action"]  # "BUY" / "SELL"
        target_delta = float(auto_cfg["target_delta"])

        chain = md_option_chain(symbol, expiry, side=side)
        if chain is None or chain.empty:
            st.warning("Chain unavailable for that expiry/side.")
        else:
            recs = recommend_single_leg(chain, float(spot), side, action, target_delta)
            st.write(f"Chosen expiry: **{expiry}**")
            if recs is None or recs.empty:
                st.warning("No recommendations (quotes/greeks missing). This usually means OPRA/quotes aren’t enabled on your MarketData plan.")
            else:
                st.dataframe(recs, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Tab 2: Auto Credit Spreads
# ---------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Auto spreads based on bias + delta + your wing width. Uses per-leg quote mids.")

    # Build DAILY + WEEKLY using auto delta
    target_delta = float(auto_cfg["target_delta"])

    def auto_spread_block(title: str, dte_min: int, dte_max: int):
        st.markdown(f"**{title}**  <span class='small-muted'>(DTE {dte_min}-{dte_max})</span>", unsafe_allow_html=True)
        expiry = pick_expiration_in_dte_window(expirations, dte_min, dte_max)
        if not expiry:
            st.warning("No expiry in this DTE window.")
            return

        calls = md_option_chain(symbol, expiry, side="call")
        puts = md_option_chain(symbol, expiry, side="put")
        if calls is None or puts is None or calls.empty or puts.empty:
            st.warning("Chain unavailable.")
            return

        if auto_bias == "bullish":
            strat = build_vertical_from_chain(puts, float(spot), "put", target_delta, float(wing_width), bearish_call=False)
        elif auto_bias == "bearish":
            strat = build_vertical_from_chain(calls, float(spot), "call", target_delta, float(wing_width), bearish_call=True)
        else:
            strat = build_iron_condor(puts, calls, float(spot), target_delta, float(wing_width))

        st.write(f"Chosen expiry: **{expiry}**")
        if not strat:
            st.warning("Could not construct spread. (Most common cause: option quotes not available / OPRA not enabled.)")
            return

        st.success(f"Recommended: **{strat['name']}**")
        st.write("Legs:")
        for leg in strat["legs"]:
            a, k, sym = leg
            st.write(f"- {a} **{k}** (`{sym}`)")

        credit = strat.get("credit", np.nan)
        max_loss = strat.get("max_loss", np.nan)
        if np.isfinite(credit):
            st.write(f"Est credit: **{credit:.2f}**")
        if np.isfinite(max_loss):
            st.write(f"Est max loss/share: **{max_loss:.2f}** (x100/contract)")
        if strat["name"] == "Iron Condor":
            pc = strat.get("put_credit", np.nan)
            cc = strat.get("call_credit", np.nan)
            if np.isfinite(pc) and np.isfinite(cc):
                st.write(f"Put credit: {pc:.2f} | Call credit: {cc:.2f}")

        st.divider()

    auto_spread_block("📅 Daily Spread Idea", 0, 2)
    auto_spread_block("🗓️ Weekly Spread Idea", 5, 12)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Tab 3: Manual Ticket (no rerun refetch)
# ---------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    expiry = st.selectbox("Expiration", expirations, index=0, key="manual_exp")
    mode = st.selectbox("Mode", ["Single Option", "Bull Put Spread", "Bear Call Spread", "Iron Condor"], index=0, key="manual_mode")

    if mode == "Single Option":
        col1, col2, col3 = st.columns(3)
        cp = col1.selectbox("Call/Put", ["Call", "Put"], key="manual_cp")
        action = col2.selectbox("Action", ["BUY", "SELL"], key="manual_act")
        qty = col3.number_input("Contracts", 1, 200, 1, 1, key="manual_qty")

        side = "call" if cp == "Call" else "put"
        chain = md_option_chain(symbol, expiry, side=side)
        if chain is None or chain.empty:
            st.warning("Chain unavailable.")
        else:
            chain = chain.dropna(subset=["strike", "optionSymbol"]).copy()
            chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
            chain = chain.dropna(subset=["strike"])
            chain = chain.sort_values("strike")

            strike = st.selectbox("Strike", chain["strike"].tolist(), index=0, key="manual_strike")
            row = chain[chain["strike"] == strike].head(1)
            opt_sym = str(row["optionSymbol"].iloc[0])

            q = md_option_quote(opt_sym)
            st.write(f"Ticket: **{action} {qty}x {cp.upper()} {strike}** exp **{expiry}**")
            st.write(f"Contract: `{opt_sym}`")

            if not q:
                st.warning("Quote not available (OPRA/quotes not enabled).")
            else:
                a, b, c, d = st.columns(4)
                a.metric("Bid", f"{q['bid']:.2f}" if np.isfinite(q["bid"]) else "—")
                b.metric("Ask", f"{q['ask']:.2f}" if np.isfinite(q["ask"]) else "—")
                c.metric("Mid", f"{q['mid']:.2f}" if np.isfinite(q["mid"]) else "—")
                d.metric("Δ", f"{q['delta']:.2f}" if np.isfinite(q["delta"]) else "—")
                if np.isfinite(q["mid"]):
                    est = q["mid"] * 100 * qty
                    label = "Est cost" if action == "BUY" else "Est credit"
                    st.info(f"{label} (mid): **${est:,.0f}**")

    else:
        # manual spreads (uses your wing width slider + auto delta)
        target_delta = float(auto_cfg["target_delta"])

        calls = md_option_chain(symbol, expiry, side="call")
        puts = md_option_chain(symbol, expiry, side="put")
        if calls is None or puts is None or calls.empty or puts.empty:
            st.warning("Chain unavailable.")
        else:
            if mode == "Bull Put Spread":
                strat = build_vertical_from_chain(puts, float(spot), "put", target_delta, float(wing_width), bearish_call=False)
            elif mode == "Bear Call Spread":
                strat = build_vertical_from_chain(calls, float(spot), "call", target_delta, float(wing_width), bearish_call=True)
            else:
                strat = build_iron_condor(puts, calls, float(spot), target_delta, float(wing_width))

            if not strat:
                st.warning("Could not construct spread. (Most common cause: quotes not available / OPRA not enabled.)")
            else:
                st.success(f"Recommended: **{strat['name']}**")
                for leg in strat["legs"]:
                    a, k, sym = leg
                    st.write(f"- {a} **{k}** (`{sym}`)")
                credit = strat.get("credit", np.nan)
                max_loss = strat.get("max_loss", np.nan)
                if np.isfinite(credit):
                    st.write(f"Est credit: **{credit:.2f}**")
                if np.isfinite(max_loss):
                    st.write(f"Est max loss/share: **{max_loss:.2f}** (x100/contract)")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Note: Streamlit reruns the script on widget changes, but this app only refreshes data when you press RUN.")