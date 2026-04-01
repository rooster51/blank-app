import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TZ = ZoneInfo("America/New_York")


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Liquidity Sweep Spread Finder", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def now_et() -> datetime:
    return datetime.now(APP_TZ)


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def round_to_increment(price: float, increment: float) -> float:
    if increment <= 0:
        return round(price, 2)
    return round(round(price / increment) * increment, 2)


def infer_strike_increment(spot: float) -> float:
    # Practical default for idea generation with liquid names
    if spot < 50:
        return 0.5
    if spot < 200:
        return 1.0
    return 1.0


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Flatten MultiIndex columns if needed
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    keep = {}
    for c in out.columns:
        name = str(c).strip().lower()
        if name == "open":
            keep[c] = "Open"
        elif name == "high":
            keep[c] = "High"
        elif name == "low":
            keep[c] = "Low"
        elif name == "close":
            keep[c] = "Close"
        elif name == "adj close":
            keep[c] = "Adj Close"
        elif name == "volume":
            keep[c] = "Volume"

    out = out.rename(columns=keep)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[needed].copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = 0

    out = out.sort_index()
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
    return normalize_columns(df)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(symbol: str):
    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options)
    chains = {}
    for exp in expirations[:8]:  # keep it light
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            chains[exp] = {"calls": calls, "puts": puts}
        except Exception:
            continue
    return expirations, chains


def add_ema(df: pd.DataFrame, span: int, col_name: str) -> pd.DataFrame:
    df[col_name] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def add_volume_metrics(df: pd.DataFrame, vol_period: int = 20, spike_threshold: float = 1.5) -> pd.DataFrame:
    df["vol_ma"] = df["Volume"].rolling(vol_period).mean()
    df["vol_ratio"] = np.where(df["vol_ma"] > 0, df["Volume"] / df["vol_ma"], np.nan)
    df["volume_spike"] = df["vol_ratio"] > spike_threshold
    return df


def get_trend_from_emas(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    if df.empty or len(df) < max(fast, slow) + 5:
        return "neutral"
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "bullish"
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "bearish"
    return "neutral"


def multi_timeframe_trend(daily_df: pd.DataFrame, hourly_df: pd.DataFrame, intraday_df: pd.DataFrame) -> dict:
    daily_trend = get_trend_from_emas(daily_df)
    hourly_trend = get_trend_from_emas(hourly_df)
    intraday_trend = get_trend_from_emas(intraday_df)

    trends = [daily_trend, hourly_trend, intraday_trend]
    bulls = trends.count("bullish")
    bears = trends.count("bearish")

    aligned = "neutral"
    alignment_score = 0
    if bulls >= 2:
        aligned = "bullish"
        alignment_score = bulls
    elif bears >= 2:
        aligned = "bearish"
        alignment_score = bears

    return {
        "daily": daily_trend,
        "hourly": hourly_trend,
        "intraday": intraday_trend,
        "aligned": aligned,
        "alignment_score": alignment_score,
    }


def get_levels(df: pd.DataFrame, lookback: int = 20):
    if len(df) < lookback + 2:
        return np.nan, np.nan
    recent = df.iloc[-lookback - 1:-1]
    support = recent["Low"].min()
    resistance = recent["High"].max()
    return float(support), float(resistance)


def detect_sweep(df: pd.DataFrame, support: float, resistance: float):
    if df.empty or pd.isna(support) or pd.isna(resistance):
        return None
    last = df.iloc[-1]
    if last["Low"] < support and last["Close"] > support:
        return "bullish_sweep"
    if last["High"] > resistance and last["Close"] < resistance:
        return "bearish_sweep"
    return None


def rejection_strength(df: pd.DataFrame) -> str:
    if df.empty:
        return "weak"

    last = df.iloc[-1]
    candle_range = max(last["High"] - last["Low"], 1e-9)
    upper_wick = last["High"] - max(last["Open"], last["Close"])
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]

    if last["Close"] >= last["Open"]:
        wick_ratio = lower_wick / candle_range
    else:
        wick_ratio = upper_wick / candle_range

    if wick_ratio >= 0.40:
        return "strong"
    if wick_ratio >= 0.25:
        return "moderate"
    return "weak"


def build_no_trade_windows(
    avoid_open_minutes: int,
    avoid_close_minutes: int,
    lunch_start: time,
    lunch_end: time,
    use_lunch_window: bool,
    use_midday_friday: bool,
):
    windows = []

    # Regular cash session assumptions for US equities/ETFs
    market_open = time(9, 30)
    market_close = time(16, 0)

    open_end_minutes = 30 + avoid_open_minutes
    open_end_hour = 9 + (open_end_minutes // 60)
    open_end_min = open_end_minutes % 60
    open_end = time(open_end_hour, open_end_min)

    close_start_total = (16 * 60) - avoid_close_minutes
    close_start = time(close_start_total // 60, close_start_total % 60)

    if avoid_open_minutes > 0:
        windows.append(("Open volatility window", market_open, open_end))
    if avoid_close_minutes > 0:
        windows.append(("Close volatility window", close_start, market_close))
    if use_lunch_window:
        windows.append(("Lunch drift window", lunch_start, lunch_end))

    return windows, use_midday_friday


def is_in_no_trade_window(
    ts: datetime,
    windows,
    use_midday_friday: bool,
) -> tuple[bool, str]:
    local_ts = ts.astimezone(APP_TZ)
    t = local_ts.time()
    weekday = local_ts.weekday()  # Mon=0 ... Fri=4

    for label, start_t, end_t in windows:
        if start_t <= t <= end_t:
            return True, label

    if use_midday_friday and weekday == 4 and time(12, 0) <= t <= time(13, 30):
        return True, "Friday midday chop window"

    return False, ""


def suggest_spread(
    df: pd.DataFrame,
    sweep: str | None,
    strike_width: float,
    atr_mult: float,
    strike_increment: float,
):
    if df.empty or sweep is None:
        return None

    last = df.iloc[-1]
    atr = safe_float(last.get("ATR"))
    if atr is None or atr <= 0:
        return None

    if sweep == "bullish_sweep":
        anchor = float(last["Low"]) - (atr * atr_mult)
        short_strike = round_to_increment(anchor, strike_increment)
        long_strike = round_to_increment(short_strike - strike_width, strike_increment)
        if long_strike >= short_strike:
            long_strike = round_to_increment(short_strike - strike_increment, strike_increment)
        return {
            "type": "put_credit",
            "anchor_price": round(anchor, 2),
            "short_strike": short_strike,
            "long_strike": long_strike,
        }

    if sweep == "bearish_sweep":
        anchor = float(last["High"]) + (atr * atr_mult)
        short_strike = round_to_increment(anchor, strike_increment)
        long_strike = round_to_increment(short_strike + strike_width, strike_increment)
        if long_strike <= short_strike:
            long_strike = round_to_increment(short_strike + strike_increment, strike_increment)
        return {
            "type": "call_credit",
            "anchor_price": round(anchor, 2),
            "short_strike": short_strike,
            "long_strike": long_strike,
        }

    return None


def validate_setup(
    sweep: str | None,
    rejection: str,
    mtf: dict,
    volume_spike: bool,
    in_no_trade_window: bool,
):
    if in_no_trade_window:
        return False, "Blocked by do-not-trade window"

    if sweep == "bullish_sweep":
        if mtf["aligned"] != "bullish":
            return False, "Higher timeframes not bullish"
        if rejection not in ["strong", "moderate"]:
            return False, "Rejection too weak"
        if not volume_spike:
            return False, "No volume confirmation"
        return True, "Bullish setup confirmed"

    if sweep == "bearish_sweep":
        if mtf["aligned"] != "bearish":
            return False, "Higher timeframes not bearish"
        if rejection not in ["strong", "moderate"]:
            return False, "Rejection too weak"
        if not volume_spike:
            return False, "No volume confirmation"
        return True, "Bearish setup confirmed"

    return False, "No sweep detected"


def confidence_score(
    sweep: str | None,
    rejection: str,
    mtf: dict,
    volume_spike: bool,
    in_no_trade_window: bool,
) -> int:
    score = 0

    if sweep:
        score += 30

    if rejection == "strong":
        score += 25
    elif rejection == "moderate":
        score += 15

    if mtf["alignment_score"] == 3:
        score += 30
    elif mtf["alignment_score"] == 2:
        score += 20

    if volume_spike:
        score += 15

    if in_no_trade_window:
        score -= 40

    return max(0, min(score, 100))


def apply_manual_trend_override(mtf: dict, override: str) -> dict:
    out = dict(mtf)
    if override in {"bullish", "bearish", "neutral"}:
        out["aligned"] = override
        out["manual_override"] = True
    else:
        out["manual_override"] = False
    return out


def choose_expiration(expirations: list[str], min_dte_days: int = 7, max_dte_days: int = 21):
    if not expirations:
        return None
    today = now_et().date()
    candidates = []
    for exp in expirations:
        try:
            d = pd.to_datetime(exp).date()
            dte = (d - today).days
            if min_dte_days <= dte <= max_dte_days:
                candidates.append((dte, exp))
        except Exception:
            continue
    if candidates:
        candidates.sort(key=lambda x: abs(x[0] - 14))
        return candidates[0][1]
    return expirations[0]


def find_option_idea_for_spread(symbol: str, spread: dict, expirations, chains):
    if not spread or not expirations or not chains:
        return None

    selected_exp = choose_expiration(expirations)
    if not selected_exp or selected_exp not in chains:
        return None

    side = "puts" if spread["type"] == "put_credit" else "calls"
    chain_df = chains[selected_exp][side].copy()

    if chain_df.empty or "strike" not in chain_df.columns:
        return None

    short_strike = spread["short_strike"]
    long_strike = spread["long_strike"]

    short_row = chain_df.loc[(chain_df["strike"] - short_strike).abs().idxmin()] if not chain_df.empty else None
    long_row = chain_df.loc[(chain_df["strike"] - long_strike).abs().idxmin()] if not chain_df.empty else None

    if short_row is None or long_row is None:
        return None

    short_mid = np.nanmean([short_row.get("bid", np.nan), short_row.get("ask", np.nan)])
    long_mid = np.nanmean([long_row.get("bid", np.nan), long_row.get("ask", np.nan)])
    est_credit = safe_float(short_mid, 0.0) - safe_float(long_mid, 0.0)
    width = abs(float(short_row["strike"]) - float(long_row["strike"]))
    max_loss = max(width - est_credit, 0.0) if est_credit is not None else None

    return {
        "expiration": selected_exp,
        "short_leg": {
            "strike": safe_float(short_row.get("strike")),
            "bid": safe_float(short_row.get("bid")),
            "ask": safe_float(short_row.get("ask")),
            "iv": safe_float(short_row.get("impliedVolatility")),
            "oi": safe_float(short_row.get("openInterest")),
            "volume": safe_float(short_row.get("volume")),
        },
        "long_leg": {
            "strike": safe_float(long_row.get("strike")),
            "bid": safe_float(long_row.get("bid")),
            "ask": safe_float(long_row.get("ask")),
            "iv": safe_float(long_row.get("impliedVolatility")),
            "oi": safe_float(long_row.get("openInterest")),
            "volume": safe_float(long_row.get("volume")),
        },
        "est_credit": round(est_credit, 2) if est_credit is not None and not pd.isna(est_credit) else None,
        "width": round(width, 2),
        "est_max_loss": round(max_loss, 2) if max_loss is not None else None,
    }


def analyze_setup(
    symbol: str,
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    lookback: int,
    atr_period: int,
    vol_period: int,
    vol_spike_threshold: float,
    strike_width: float,
    atr_mult: float,
    strike_increment: float,
    manual_trend_override: str,
    no_trade_windows,
    friday_midday_block: bool,
):
    intraday_df = intraday_df.copy()
    intraday_df = add_atr(intraday_df, period=atr_period)
    intraday_df = add_volume_metrics(intraday_df, vol_period=vol_period, spike_threshold=vol_spike_threshold)
    intraday_df = add_ema(intraday_df, 20, "EMA20")
    intraday_df = add_ema(intraday_df, 50, "EMA50")

    support, resistance = get_levels(intraday_df, lookback=lookback)
    sweep = detect_sweep(intraday_df, support, resistance)
    rejection = rejection_strength(intraday_df)

    mtf = multi_timeframe_trend(daily_df, hourly_df, intraday_df)
    mtf = apply_manual_trend_override(mtf, manual_trend_override)

    last_bar_ts = intraday_df.index[-1].to_pydatetime()
    in_no_trade_window, blocked_by = is_in_no_trade_window(last_bar_ts, no_trade_windows, friday_midday_block)

    last = intraday_df.iloc[-1]
    volume_spike = bool(last.get("volume_spike", False))

    is_valid, reason = validate_setup(
        sweep=sweep,
        rejection=rejection,
        mtf=mtf,
        volume_spike=volume_spike,
        in_no_trade_window=in_no_trade_window,
    )

    spread = None
    if is_valid:
        spread = suggest_spread(
            intraday_df,
            sweep=sweep,
            strike_width=strike_width,
            atr_mult=atr_mult,
            strike_increment=strike_increment,
        )

    confidence = confidence_score(
        sweep=sweep,
        rejection=rejection,
        mtf=mtf,
        volume_spike=volume_spike,
        in_no_trade_window=in_no_trade_window,
    )

    return {
        "symbol": symbol.upper(),
        "spot": round(float(last["Close"]), 2),
        "last_bar_time": last_bar_ts.astimezone(APP_TZ),
        "support": round(support, 2) if not pd.isna(support) else None,
        "resistance": round(resistance, 2) if not pd.isna(resistance) else None,
        "sweep": sweep,
        "rejection": rejection,
        "volume_spike": volume_spike,
        "vol_ratio": safe_float(last.get("vol_ratio")),
        "atr": safe_float(last.get("ATR")),
        "mtf": mtf,
        "in_no_trade_window": in_no_trade_window,
        "blocked_by": blocked_by,
        "is_valid": is_valid,
        "reason": blocked_by if in_no_trade_window else reason,
        "spread": spread,
        "confidence": confidence,
        "intraday_df": intraday_df,
    }


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Ticker", value="SPY").upper().strip()
manual_refresh = st.sidebar.button("Refresh data")

manual_trend_override = st.sidebar.selectbox(
    "Trend override",
    options=["auto", "bullish", "bearish", "neutral"],
    index=0,
)

lookback = st.sidebar.slider("Sweep lookback bars", 10, 60, 20)
atr_period = st.sidebar.slider("ATR period", 5, 30, 14)
vol_period = st.sidebar.slider("Volume MA period", 5, 40, 20)
vol_spike_threshold = st.sidebar.slider("Volume spike threshold", 1.0, 3.0, 1.5, 0.1)

default_strike_increment = infer_strike_increment(100)
strike_increment = st.sidebar.selectbox(
    "Strike increment",
    options=[0.5, 1.0, 2.5, 5.0],
    index=[0.5, 1.0, 2.5, 5.0].index(default_strike_increment if default_strike_increment in [0.5, 1.0, 2.5, 5.0] else 1.0),
)

strike_width = st.sidebar.number_input("Spread width", min_value=float(strike_increment), value=5.0, step=float(strike_increment))
atr_mult = st.sidebar.slider("ATR buffer multiplier", 0.25, 2.0, 0.75, 0.05)

st.sidebar.subheader("Do-not-trade windows")
avoid_open_minutes = st.sidebar.slider("Avoid after open (minutes)", 0, 90, 15)
avoid_close_minutes = st.sidebar.slider("Avoid before close (minutes)", 0, 120, 60)
use_lunch_window = st.sidebar.checkbox("Block lunch drift window", value=False)
lunch_start_h = st.sidebar.number_input("Lunch start hour", min_value=10, max_value=14, value=12)
lunch_start_m = st.sidebar.number_input("Lunch start minute", min_value=0, max_value=59, value=0)
lunch_end_h = st.sidebar.number_input("Lunch end hour", min_value=10, max_value=15, value=13)
lunch_end_m = st.sidebar.number_input("Lunch end minute", min_value=0, max_value=59, value=0)
use_midday_friday = st.sidebar.checkbox("Block Friday midday", value=True)

show_options_idea = st.sidebar.checkbox("Try options-chain idea lookup", value=True)


# -----------------------------
# Refresh cache
# -----------------------------
if manual_refresh:
    st.cache_data.clear()


# -----------------------------
# Load data
# -----------------------------
try:
    daily_df = fetch_price_history(symbol, period="1y", interval="1d")
    hourly_df = fetch_price_history(symbol, period="60d", interval="60m")
    intraday_df = fetch_price_history(symbol, period="30d", interval="15m")
except Exception as e:
    st.error(f"Failed to load price history for {symbol}: {e}")
    st.stop()

if daily_df.empty or hourly_df.empty or intraday_df.empty:
    st.error("Not enough data returned. Try another symbol.")
    st.stop()


# -----------------------------
# Build windows + analyze
# -----------------------------
lunch_start = time(int(lunch_start_h), int(lunch_start_m))
lunch_end = time(int(lunch_end_h), int(lunch_end_m))

windows, friday_midday_block = build_no_trade_windows(
    avoid_open_minutes=avoid_open_minutes,
    avoid_close_minutes=avoid_close_minutes,
    lunch_start=lunch_start,
    lunch_end=lunch_end,
    use_lunch_window=use_lunch_window,
    use_midday_friday=use_midday_friday,
)

result = analyze_setup(
    symbol=symbol,
    daily_df=daily_df,
    hourly_df=hourly_df,
    intraday_df=intraday_df,
    lookback=lookback,
    atr_period=atr_period,
    vol_period=vol_period,
    vol_spike_threshold=vol_spike_threshold,
    strike_width=strike_width,
    atr_mult=atr_mult,
    strike_increment=strike_increment,
    manual_trend_override=manual_trend_override,
    no_trade_windows=windows,
    friday_midday_block=friday_midday_block,
)

# -----------------------------
# Top dashboard
# -----------------------------
st.title("Liquidity Sweep Spread Finder")
st.caption(f"Time now: {now_et().strftime('%Y-%m-%d %I:%M:%S %p ET')}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker", result["symbol"])
c2.metric("Spot", f'{result["spot"]:.2f}')
c3.metric("Sweep", result["sweep"] or "None")
c4.metric("Confidence", f'{result["confidence"]}%')
c5.metric("ATR", f'{result["atr"]:.2f}' if result["atr"] else "N/A")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Support", f'{result["support"]:.2f}' if result["support"] else "N/A")
c7.metric("Resistance", f'{result["resistance"]:.2f}' if result["resistance"] else "N/A")
c8.metric("Rejection", result["rejection"].title())
c9.metric("Volume spike", "Yes" if result["volume_spike"] else "No")

if result["in_no_trade_window"]:
    st.warning(f"Do not trade right now: {result['reason']}")
elif result["is_valid"]:
    st.success(result["reason"])
else:
    st.info(result["reason"])


# -----------------------------
# Trend alignment
# -----------------------------
st.subheader("Trend Alignment")
mtf = result["mtf"]
st.write(
    f"Daily: **{mtf['daily']}** | "
    f"Hourly: **{mtf['hourly']}** | "
    f"Intraday: **{mtf['intraday']}** | "
    f"Aligned: **{mtf['aligned']}**"
)

if mtf.get("manual_override"):
    st.caption("Manual trend override is active.")


# -----------------------------
# Spread idea
# -----------------------------
st.subheader("Spread Suggestion")

spread = result["spread"]
if spread:
    st.write(
        f"**Trade Type:** {spread['type']}  \n"
        f"**Anchor Price:** {spread['anchor_price']:.2f}  \n"
        f"**Short Strike:** {spread['short_strike']:.2f}  \n"
        f"**Long Strike:** {spread['long_strike']:.2f}"
    )
else:
    st.write("No valid spread suggestion right now.")


# -----------------------------
# Options-chain idea lookup
# -----------------------------
if show_options_idea and spread:
    try:
        expirations, chains = fetch_options_chain(symbol)
        idea = find_option_idea_for_spread(symbol, spread, expirations, chains)

        st.subheader("Options Chain Snapshot")
        if idea:
            st.write(
                f"**Expiration:** {idea['expiration']}  \n"
                f"**Estimated Credit:** {idea['est_credit'] if idea['est_credit'] is not None else 'N/A'}  \n"
                f"**Width:** {idea['width']}  \n"
                f"**Estimated Max Loss:** {idea['est_max_loss'] if idea['est_max_loss'] is not None else 'N/A'}"
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Short leg**")
                st.json(idea["short_leg"])
            with right:
                st.markdown("**Long leg**")
                st.json(idea["long_leg"])
        else:
            st.write("Could not match the suggested spread cleanly to the current options chain.")
    except Exception as e:
        st.warning(f"Options chain lookup failed: {e}")


# -----------------------------
# Do-not-trade windows display
# -----------------------------
st.subheader("Active Do-Not-Trade Rules")
if windows:
    for label, start_t, end_t in windows:
        st.write(f"- {label}: {start_t.strftime('%I:%M %p')} to {end_t.strftime('%I:%M %p')} ET")
if friday_midday_block:
    st.write("- Friday midday chop window: 12:00 PM to 1:30 PM ET")


# -----------------------------
# Chart data
# -----------------------------
st.subheader("Intraday Chart Data")
chart_df = result["intraday_df"][["Close", "EMA20", "EMA50"]].copy()
st.line_chart(chart_df)

with st.expander("Recent intraday bars"):
    display_cols = ["Open", "High", "Low", "Close", "Volume", "ATR", "vol_ratio", "volume_spike", "EMA20", "EMA50"]
    st.dataframe(result["intraday_df"][display_cols].tail(50), use_container_width=True)


# -----------------------------
# Notes
# -----------------------------
st.subheader("How to use this")
st.write(
    """
- A bullish sweep means price ran below support and reclaimed it.
- A bearish sweep means price ran above resistance and failed back under it.
- This app only suggests a spread when sweep + rejection + volume + trend alignment agree,
  and when the current bar is not inside a blocked trading window.
- Manual trend override is there in case you want to force the system to match your read.
    """
)