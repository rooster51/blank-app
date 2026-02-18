import os
import math
from datetime import datetime, date, timezone

import numpy as np
import pandas as pd
import streamlit as st
import requests
from scipy.stats import norm

# =========================
# App config
# =========================
st.set_page_config(page_title="Trend + Credit Spreads", layout="wide")
TZ = "America/New_York"

DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "SPY", "QQQ", "TSLA", "AMD", "META", "AMZN", "GOOGL"]

MD_BASE = "https://api.marketdata.app/v1"

# =========================
# Styling (dark-friendly)
# =========================
st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 1.0rem; padding-bottom: 2.2rem; }
.card { border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 14px; background: rgba(255,255,255,0.03); }
.small-muted { opacity: 0.80; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Trend Scanner → Credit Spread Builder")
st.caption("Educational only. Options trading involves substantial risk (assignment, gap risk, liquidity).")

# =========================
# Token handling
# =========================
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
            return {"s": "error", "errmsg": f"HTTP {r.status_code}: {r.text[:180]}"}
        return r.json()
    except Exception as e:
        return {"s": "error", "errmsg": f"Request error: {e}"}

# =========================
# Indicators / Trend
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
    # Allows trend reading even if SMA200 not present
    if df is None or df.empty or len(df) < 60:
        return {"direction": "insufficient", "strength": 0, "regime": "", "notes": "Need ~60+ bars"}

    last = df.iloc[-1]
    close = float(last["close"])
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0

    s20 = last.get("SMA20", np.nan)
    s50 = last.get("SMA50", np.nan)
    s200 = last.get("SMA200", np.nan)

    has_50 = pd.notna(s20) and pd.notna(s50)
    has_200 = pd.notna(s200)

    direction = "neutral"
    strength = 45

    if has_50:
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

# =========================
# MarketData Candles
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def md_candles(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame | None:
    """
    Uses MarketData candles endpoint.
    NOTE: Endpoint/path/params may vary slightly by plan. If your plan returns different keys,
    we normalize for common 't/o/h/l/c/v' style arrays.
    """
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=lookback_days)
    params = {
        "interval": interval,                       # e.g. "15m", "1d"
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
    }
    j = md_get_json(f"{MD_BASE}/stocks/candles/{symbol}/", params=params)
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None

    # Common response patterns: arrays for t/o/h/l/c/v
    # We handle a few key variants defensively.
    t = j.get("t") or j.get("time") or j.get("timestamp")
    o = j.get("o") or j.get("open")
    h = j.get("h") or j.get("high")
    l = j.get("l") or j.get("low")
    c = j.get("c") or j.get("close")
    v = j.get("v") or j.get("volume")

    if not (isinstance(t, list) and isinstance(c, list)) or len(t) == 0:
        return None

    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    idx = pd.to_datetime(pd.Series(t), unit="s", utc=True).dt.tz_convert(TZ)
    df.index = idx
    df.index.name = "time"
    return df.dropna()

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
    # 15m: ~60 days
    df_15m = md_candles(symbol, interval="15m", lookback_days=60)
    # 4h: resample from 15m (stable + consistent)
    df_4h = resample_ohlcv(df_15m, "4H") if df_15m is not None and not df_15m.empty else None
    # 1D: ~3y
    df_1d = md_candles(symbol, interval="1d", lookback_days=365 * 3)
    return {"15m": df_15m, "4h": df_4h, "1D": df_1d}

# =========================
# MarketData Options
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def md_stock_quote(symbol: str) -> float | None:
    j = md_get_json(f"{MD_BASE}/stocks/quotes/{symbol}/")
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None
    for key in ("last", "mid", "Last", "Mid"):
        v = j.get(key, None)
        if isinstance(v, list) and v:
            try:
                return float(v[-1])
            except Exception:
                pass
        if isinstance(v, (int, float)):
            return float(v)
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def md_options_expirations(symbol: str) -> list[str] | None:
    j = md_get_json(f"{MD_BASE}/options/expirations/{symbol}/")
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None
    exps = j.get("expirations", [])
    return [str(x) for x in exps] if isinstance(exps, list) else None

@st.cache_data(ttl=600, show_spinner=False)
def md_option_chain(symbol: str, expiration: str, side: str) -> pd.DataFrame | None:
    params = {"expiration": expiration, "side": side}  # side: "call" or "put"
    j = md_get_json(f"{MD_BASE}/options/chain/{symbol}/", params=params)
    if not isinstance(j, dict) or j.get("s") != "ok":
        return None
    cols = {k: v for k, v in j.items() if isinstance(v, list)}
    if not cols:
        return None
    df = pd.DataFrame(cols)
    for c in ("strike", "bid", "ask", "mid", "iv", "delta"):
        if c not in df.columns:
            df[c] = np.nan
    for c in ("strike", "bid", "ask", "mid", "iv", "delta"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================
# Spread logic
# =========================
def bs_delta(S, K, T, r, sigma, is_call: bool):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

def ensure_delta(df: pd.DataFrame, spot: float, expiry: str, r: float, is_call: bool) -> pd.DataFrame:
    out = df.copy()
    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = max((ed - date.today()).days / 365.0, 1e-6)

    if "delta" in out.columns and out["delta"].notna().any():
        out["delta_est"] = out["delta"]
    else:
        out["delta_est"] = [
            bs_delta(spot, float(k), T, r, float(iv), is_call)
            if np.isfinite(k) and np.isfinite(iv)
            else np.nan
            for k, iv in zip(out["strike"].values, out["iv"].values)
        ]

    if "mid" not in out.columns or not out["mid"].notna().any():
        out["mid"] = np.where(
            np.isfinite(out["bid"]) & np.isfinite(out["ask"]) & (out["ask"] > 0),
            (out["bid"] + out["ask"]) / 2.0,
            np.nan,
        )
    return out

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
    return {"name": "Bull Put Spread",
            "legs": [("SELL PUT", float(short["strike"])), ("BUY PUT", float(long["strike"]))],
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
    return {"name": "Bear Call Spread",
            "legs": [("SELL CALL", float(short["strike"])), ("BUY CALL", float(long["strike"]))],
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

    return {"name": "Iron Condor",
            "legs": [("SELL PUT", spk), ("BUY PUT", lpk), ("SELL CALL", sck), ("BUY CALL", lck)],
            "credit": total_credit, "width": max_w, "max_loss": max_loss,
            "put_credit": put_credit, "call_credit": call_credit}

def build_strategy(mode: str, auto_bias: str, puts_e, calls_e, target_delta: float, wing_width: float):
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

# =========================
# UI state
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

if not get_marketdata_token():
    st.warning("Add MARKETDATA_TOKEN in Streamlit Secrets to enable MarketData candles + options.")

# =========================
# Controls (form)
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("controls", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2.2, 1.5, 1.3, 1.2])

    with c1:
        symbol = st.selectbox("Ticker", st.session_state.watchlist, index=0)
        new_sym = st.text_input("Add ticker", placeholder="NFLX").strip().upper()
        bA, bB = st.columns(2)
        with bA:
            add_clicked = st.form_submit_button("Add")
        with bB:
            remove_clicked = st.form_submit_button("Remove")

    with c2:
        mode = st.selectbox("Strategy mode", ["Auto", "Bull Put", "Bear Call", "Iron Condor"])
        target_delta = st.slider("Target delta", 0.10, 0.35, 0.20, 0.01)

    with c3:
        wing_width = st.slider("Wing width ($)", 1.00, 2.50, 1.50, 0.05)
        r_rate = st.number_input("Risk-free r", 0.0, 0.20, 0.04, 0.005)

    with c4:
        show_charts = st.checkbox("Show charts", value=True)
        skip_options = st.checkbox("Skip options", value=False)
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

if not run:
    st.stop()

# =========================
# Trend scan (MarketData candles)
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
                if dfp[col].notna().any():
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
# Options ideas (MarketData)
# =========================
if skip_options:
    st.info("Options skipped (charts only).")
    st.stop()

st.subheader("Options chain → spread ideas (Daily + Weekly)")

spot = md_stock_quote(symbol)
if spot is None and latest_close is not None:
    spot = float(latest_close)
if spot is None or not np.isfinite(spot):
    st.error("Spot price unavailable.")
    st.stop()

st.write(f"Spot: **{spot:.2f}**")

expirations = md_options_expirations(symbol)
if not expirations:
    st.error("No expirations returned (token/plan may not include options, or symbol unsupported).")
    st.stop()

def render_spread_idea(title: str, dte_min: int, dte_max: int):
    st.markdown(
        f"<div class='card'><b>{title}</b><div class='small-muted'>DTE window: {dte_min}–{dte_max}</div>",
        unsafe_allow_html=True,
    )

    expiry = pick_expiration_in_dte_window(expirations, dte_min, dte_max)
    if not expiry:
        st.write("No expiration found in this DTE window.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ed = datetime.strptime(expiry, "%Y-%m-%d").date()
    st.write(f"Chosen expiry: **{expiry}** (DTE={(ed - date.today()).days})")

    calls = md_option_chain(symbol, expiry, side="call")
    puts = md_option_chain(symbol, expiry, side="put")
    if calls is None or puts is None or calls.empty or puts.empty:
        st.write("Options chain unavailable for this expiry (plan might not include quotes/greeks).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    calls_e = ensure_delta(calls, spot, expiry, float(r_rate), is_call=True)
    puts_e = ensure_delta(puts, spot, expiry, float(r_rate), is_call=False)

    strat = build_strategy(mode, auto_bias, puts_e, calls_e, float(target_delta), float(wing_width))
    if not strat:
        st.write("Could not construct a spread (missing mids/IV/delta/strikes). Try another delta/wing width.")
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