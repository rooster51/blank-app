import time as pytime
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Spread Finder", layout="wide")

APP_TZ = ZoneInfo("America/New_York")
BASE_URL = "https://api.marketdata.app/v1"


# =========================================================
# SECRETS / AUTH
# =========================================================
def get_api_token() -> str:
    token = st.secrets.get("MARKETDATA_API_KEY", "")
    if not token:
        st.error("Missing MARKETDATA_API_KEY in Streamlit secrets.")
        st.stop()
    return token


API_TOKEN = get_api_token()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {API_TOKEN}"})
    return session


SESSION = make_session()


# =========================================================
# UTILS
# =========================================================
def now_et() -> datetime:
    return datetime.now(APP_TZ)


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def round_to_increment(price: float, increment: float) -> float:
    if increment <= 0:
        return round(price, 2)
    return round(round(price / increment) * increment, 2)


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(APP_TZ)
    return df.sort_index()


def fmt_num(value, digits=2, default="N/A"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return f"{float(value):,.{digits}f}"
    except Exception:
        return default


def fmt_pct(value, digits=1, default="N/A"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return default


def yes_no(value):
    return "Yes" if bool(value) else "No"


def quality_label(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def quality_text(score: int) -> str:
    label = quality_label(score)
    mapping = {
        "A": "A Setup",
        "B": "B Setup",
        "C": "C Setup",
        "D": "D Setup",
    }
    return mapping[label]


def setup_bias_text(profile: dict) -> str:
    return str(profile.get("direction", "neutral")).replace("_", " ").title()


def leg_summary_df(*legs):
    rows = []
    for label, leg in legs:
        if not leg:
            continue
        rows.append(
            {
                "Leg": label,
                "Strike": fmt_num(leg.get("strike")),
                "Bid": fmt_num(leg.get("bid")),
                "Ask": fmt_num(leg.get("ask")),
                "Mid": fmt_num(leg.get("mid")),
                "Delta": fmt_num(leg.get("delta"), 3),
                "IV": fmt_pct(leg.get("iv"), 1),
                "OI": fmt_num(leg.get("openInterest"), 0),
                "Vol": fmt_num(leg.get("volume"), 0),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# MARKETDATA API
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def md_get(url: str, params: dict | None = None):
    resp = SESSION.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def md_get_with_retry(url: str, params: dict | None = None, retries: int = 3, sleep_seconds: int = 2):
    last_error = None
    for attempt in range(retries):
        try:
            return md_get(url, params)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                pytime.sleep(sleep_seconds * (attempt + 1))
            else:
                raise last_error
    raise last_error


def build_candles_df(data: dict) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    t = data.get("t", [])
    o = data.get("o", [])
    h = data.get("h", [])
    l = data.get("l", [])
    c = data.get("c", [])
    v = data.get("v", [])

    if not t or not c:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(t, unit="s", utc=True),
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v if v else [0] * len(t),
        }
    ).set_index("Date")

    return normalize_datetime_index(df)


def md_candles(symbol: str, resolution: str, countback: int = 120) -> pd.DataFrame:
    url = f"{BASE_URL}/stocks/candles/{resolution}/{symbol}/"
    params = {
        "countback": countback,
        "format": "json",
    }
    data = md_get_with_retry(url, params=params)
    return build_candles_df(data)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_core_data(symbol: str):
    daily_df = md_candles(symbol, "D", countback=180)
    hourly_df = md_candles(symbol, "60", countback=180)
    intraday_df = md_candles(symbol, "15", countback=180)
    return daily_df, hourly_df, intraday_df


@st.cache_data(ttl=300, show_spinner=False)
def md_expirations(symbol: str) -> list[str]:
    url = f"{BASE_URL}/options/expirations/{symbol}/"
    params = {"format": "json"}
    data = md_get_with_retry(url, params=params)

    exps = data.get("expirations", [])
    if not exps and "expiration" in data:
        exps = data.get("expiration", [])

    return sorted([str(x) for x in exps])


@st.cache_data(ttl=120, show_spinner=False)
def md_filtered_chain(
    symbol: str,
    expiration: str,
    side: str,
    strike_low: float,
    strike_high: float,
):
    url = f"{BASE_URL}/options/chain/{symbol}/"
    params = {
        "expiration": expiration,
        "side": side,
        "strike": f"{strike_low}-{strike_high}",
        "strikeLimit": 12,
        "minOpenInterest": 50,
        "minVolume": 1,
        "format": "json",
    }
    return md_get_with_retry(url, params=params)


def chain_json_to_df(data: dict) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    if "s" in data and isinstance(data["s"], list):
        return pd.DataFrame(data["s"])

    length = 0
    for value in data.values():
        if isinstance(value, list):
            length = max(length, len(value))

    if length == 0:
        return pd.DataFrame()

    out = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == length:
            out[key] = value

    return pd.DataFrame(out)


# =========================================================
# TECHNICALS
# =========================================================
def add_ema(df: pd.DataFrame, span: int, col_name: str) -> pd.DataFrame:
    df = df.copy()
    df[col_name] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def add_volume_metrics(df: pd.DataFrame, vol_period: int = 20, spike_threshold: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df["vol_ma"] = df["Volume"].rolling(vol_period).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma"]
    df["volume_spike"] = df["vol_ratio"] > spike_threshold
    return df


def get_trend_from_emas(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    if df.empty or len(df) < slow + 5:
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
        return None, None
    recent = df.iloc[-lookback - 1:-1]
    support = recent["Low"].min()
    resistance = recent["High"].max()
    return float(support), float(resistance)


def detect_sweep(df: pd.DataFrame, support: float | None, resistance: float | None):
    if df.empty or support is None or resistance is None:
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


# =========================================================
# DO-NOT-TRADE WINDOWS
# =========================================================
def build_no_trade_windows(
    avoid_open_minutes: int,
    avoid_close_minutes: int,
    lunch_start: time,
    lunch_end: time,
    use_lunch_window: bool,
    use_midday_friday: bool,
):
    windows = []

    market_open = time(9, 30)
    market_close = time(16, 0)

    if avoid_open_minutes > 0:
        end_minutes = (9 * 60 + 30) + avoid_open_minutes
        windows.append(
            (
                "Open volatility window",
                market_open,
                time(end_minutes // 60, end_minutes % 60),
            )
        )

    if avoid_close_minutes > 0:
        start_minutes = (16 * 60) - avoid_close_minutes
        windows.append(
            (
                "Close volatility window",
                time(start_minutes // 60, start_minutes % 60),
                market_close,
            )
        )

    if use_lunch_window:
        windows.append(("Lunch drift window", lunch_start, lunch_end))

    return windows, use_midday_friday


def is_in_no_trade_window(ts: datetime, windows, use_midday_friday: bool) -> tuple[bool, str]:
    local_ts = ts.astimezone(APP_TZ)
    current_time = local_ts.time()
    weekday = local_ts.weekday()

    for label, start_t, end_t in windows:
        if start_t <= current_time <= end_t:
            return True, label

    if use_midday_friday and weekday == 4 and time(12, 0) <= current_time <= time(13, 30):
        return True, "Friday midday chop window"

    return False, ""


# =========================================================
# STRATEGY LOGIC
# =========================================================
def get_auto_direction(result: dict) -> str:
    sweep = result.get("sweep")
    mtf = result.get("mtf", {}).get("aligned", "neutral")

    if sweep == "bullish_sweep" and mtf == "bullish":
        return "bullish"
    if sweep == "bearish_sweep" and mtf == "bearish":
        return "bearish"
    return "neutral"


def strategy_profile(strategy_choice: str, auto_direction: str | None = None):
    if strategy_choice == "Auto":
        if auto_direction == "bullish":
            return {
                "strategy": "Put Credit Spread",
                "side": "put",
                "structure": "credit",
                "direction": "bullish",
            }
        if auto_direction == "bearish":
            return {
                "strategy": "Call Credit Spread",
                "side": "call",
                "structure": "credit",
                "direction": "bearish",
            }
        return {
            "strategy": "Iron Condor",
            "side": "both",
            "structure": "credit",
            "direction": "neutral",
        }

    mapping = {
        "Put Credit Spread": {
            "strategy": "Put Credit Spread",
            "side": "put",
            "structure": "credit",
            "direction": "bullish",
        },
        "Call Credit Spread": {
            "strategy": "Call Credit Spread",
            "side": "call",
            "structure": "credit",
            "direction": "bearish",
        },
        "Iron Condor": {
            "strategy": "Iron Condor",
            "side": "both",
            "structure": "credit",
            "direction": "neutral",
        },
        "Put Debit Spread": {
            "strategy": "Put Debit Spread",
            "side": "put",
            "structure": "debit",
            "direction": "bearish",
        },
        "Call Debit Spread": {
            "strategy": "Call Debit Spread",
            "side": "call",
            "structure": "debit",
            "direction": "bullish",
        },
        "Cash Secured Put": {
            "strategy": "Cash Secured Put",
            "side": "put",
            "structure": "short_single",
            "direction": "bullish",
        },
        "Covered Call": {
            "strategy": "Covered Call",
            "side": "call",
            "structure": "short_single",
            "direction": "neutral_bullish",
        },
    }

    return mapping[strategy_choice]


def suggest_trade_levels(result: dict, profile: dict, spot: float, width: float, atr_mult: float, strike_increment: float):
    atr = result.get("atr")
    support = result.get("support")
    resistance = result.get("resistance")
    spread = result.get("spread")

    if atr is None:
        return None

    if profile["strategy"] == "Put Credit Spread":
        if spread and spread.get("type") == "put_credit":
            short_strike = spread["short_strike"]
            long_strike = spread["long_strike"]
        else:
            short_strike = round_to_increment(support - atr * atr_mult, strike_increment)
            long_strike = round_to_increment(short_strike - width, strike_increment)
        return {"option_side": "put", "short_strike": short_strike, "long_strike": long_strike}

    if profile["strategy"] == "Call Credit Spread":
        if spread and spread.get("type") == "call_credit":
            short_strike = spread["short_strike"]
            long_strike = spread["long_strike"]
        else:
            short_strike = round_to_increment(resistance + atr * atr_mult, strike_increment)
            long_strike = round_to_increment(short_strike + width, strike_increment)
        return {"option_side": "call", "short_strike": short_strike, "long_strike": long_strike}

    if profile["strategy"] == "Put Debit Spread":
        long_strike = round_to_increment(spot, strike_increment)
        short_strike = round_to_increment(long_strike - width, strike_increment)
        return {"option_side": "put", "long_strike": long_strike, "short_strike": short_strike}

    if profile["strategy"] == "Call Debit Spread":
        long_strike = round_to_increment(spot, strike_increment)
        short_strike = round_to_increment(long_strike + width, strike_increment)
        return {"option_side": "call", "long_strike": long_strike, "short_strike": short_strike}

    if profile["strategy"] == "Cash Secured Put":
        short_strike = round_to_increment(support - atr * atr_mult, strike_increment)
        return {"option_side": "put", "short_strike": short_strike}

    if profile["strategy"] == "Covered Call":
        short_strike = round_to_increment(resistance + atr * atr_mult, strike_increment)
        return {"option_side": "call", "short_strike": short_strike}

    if profile["strategy"] == "Iron Condor":
        put_short = round_to_increment(support - atr * atr_mult, strike_increment)
        put_long = round_to_increment(put_short - width, strike_increment)
        call_short = round_to_increment(resistance + atr * atr_mult, strike_increment)
        call_long = round_to_increment(call_short + width, strike_increment)
        return {
            "option_side": "both",
            "put_short": put_short,
            "put_long": put_long,
            "call_short": call_short,
            "call_long": call_long,
        }

    return None


def suggest_base_spread(df: pd.DataFrame, sweep: str | None, strike_width: float, atr_mult: float, strike_increment: float):
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
        spread = suggest_base_spread(
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
        "support": round(support, 2) if support is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
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


# =========================================================
# OPTION PICKING
# =========================================================
def choose_expiration(expirations: list[str], target_dte: int = 14) -> str | None:
    if not expirations:
        return None

    today = now_et().date()
    candidates = []

    for exp in expirations:
        try:
            exp_date = pd.to_datetime(exp).date()
            dte = (exp_date - today).days
            if dte >= 1:
                candidates.append((abs(dte - target_dte), dte, exp))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def get_target_dte(expiration_mode: str) -> int:
    mapping = {
        "Auto": 14,
        "Nearest Weekly": 7,
        "14 DTE": 14,
        "30 DTE": 30,
    }
    return mapping.get(expiration_mode, 14)


def match_nearest_strike(df: pd.DataFrame, strike_value: float):
    if df.empty or "strike" not in df.columns:
        return None
    work = df.copy()
    work["dist"] = (work["strike"].astype(float) - float(strike_value)).abs()
    return work.sort_values("dist").iloc[0].to_dict()


def fetch_strategy_options(symbol: str, expiration: str, profile: dict, levels: dict):
    if not expiration or not levels:
        return {}

    result = {}

    if profile["side"] == "put":
        low = min(levels.get("short_strike", levels.get("long_strike")), levels.get("long_strike", levels.get("short_strike"))) - 10
        high = max(levels.get("short_strike", levels.get("long_strike")), levels.get("long_strike", levels.get("short_strike"))) + 10
        result["puts"] = chain_json_to_df(md_filtered_chain(symbol, expiration, "put", low, high))

    elif profile["side"] == "call":
        low = min(levels.get("short_strike", levels.get("long_strike")), levels.get("long_strike", levels.get("short_strike"))) - 10
        high = max(levels.get("short_strike", levels.get("long_strike")), levels.get("long_strike", levels.get("short_strike"))) + 10
        result["calls"] = chain_json_to_df(md_filtered_chain(symbol, expiration, "call", low, high))

    elif profile["side"] == "both":
        put_low = min(levels["put_short"], levels["put_long"]) - 10
        put_high = max(levels["put_short"], levels["put_long"]) + 10
        call_low = min(levels["call_short"], levels["call_long"]) - 10
        call_high = max(levels["call_short"], levels["call_long"]) + 10

        result["puts"] = chain_json_to_df(md_filtered_chain(symbol, expiration, "put", put_low, put_high))
        result["calls"] = chain_json_to_df(md_filtered_chain(symbol, expiration, "call", call_low, call_high))

    return result


def build_trade_from_chain(profile: dict, levels: dict, options_data: dict):
    if not levels or not options_data:
        return None

    if profile["strategy"] in {"Put Credit Spread", "Put Debit Spread"}:
        puts = options_data.get("puts", pd.DataFrame())
        if puts.empty:
            return None

        short_leg = match_nearest_strike(puts, levels["short_strike"])
        long_leg = match_nearest_strike(puts, levels["long_strike"])

        if not short_leg or not long_leg:
            return None

        short_mid = safe_float(short_leg.get("mid"), 0.0)
        long_mid = safe_float(long_leg.get("mid"), 0.0)

        if profile["structure"] == "credit":
            est_value = round(short_mid - long_mid, 2)
        else:
            est_value = round(long_mid - short_mid, 2)

        width = abs(float(short_leg["strike"]) - float(long_leg["strike"]))
        max_loss = round(width - est_value, 2) if profile["structure"] == "credit" else round(est_value, 2)

        return {
            "strategy": profile["strategy"],
            "expiration": short_leg.get("expiration"),
            "short_leg": short_leg,
            "long_leg": long_leg,
            "estimated_value": est_value,
            "width": round(width, 2),
            "estimated_max_loss": max_loss,
        }

    if profile["strategy"] in {"Call Credit Spread", "Call Debit Spread"}:
        calls = options_data.get("calls", pd.DataFrame())
        if calls.empty:
            return None

        short_leg = match_nearest_strike(calls, levels["short_strike"])
        long_leg = match_nearest_strike(calls, levels["long_strike"])

        if not short_leg or not long_leg:
            return None

        short_mid = safe_float(short_leg.get("mid"), 0.0)
        long_mid = safe_float(long_leg.get("mid"), 0.0)

        if profile["structure"] == "credit":
            est_value = round(short_mid - long_mid, 2)
        else:
            est_value = round(long_mid - short_mid, 2)

        width = abs(float(short_leg["strike"]) - float(long_leg["strike"]))
        max_loss = round(width - est_value, 2) if profile["structure"] == "credit" else round(est_value, 2)

        return {
            "strategy": profile["strategy"],
            "expiration": short_leg.get("expiration"),
            "short_leg": short_leg,
            "long_leg": long_leg,
            "estimated_value": est_value,
            "width": round(width, 2),
            "estimated_max_loss": max_loss,
        }

    if profile["strategy"] == "Cash Secured Put":
        puts = options_data.get("puts", pd.DataFrame())
        if puts.empty:
            return None

        short_leg = match_nearest_strike(puts, levels["short_strike"])
        if not short_leg:
            return None

        return {
            "strategy": profile["strategy"],
            "expiration": short_leg.get("expiration"),
            "short_leg": short_leg,
            "estimated_value": safe_float(short_leg.get("mid")),
        }

    if profile["strategy"] == "Covered Call":
        calls = options_data.get("calls", pd.DataFrame())
        if calls.empty:
            return None

        short_leg = match_nearest_strike(calls, levels["short_strike"])
        if not short_leg:
            return None

        return {
            "strategy": profile["strategy"],
            "expiration": short_leg.get("expiration"),
            "short_leg": short_leg,
            "estimated_value": safe_float(short_leg.get("mid")),
        }

    if profile["strategy"] == "Iron Condor":
        puts = options_data.get("puts", pd.DataFrame())
        calls = options_data.get("calls", pd.DataFrame())
        if puts.empty or calls.empty:
            return None

        put_short = match_nearest_strike(puts, levels["put_short"])
        put_long = match_nearest_strike(puts, levels["put_long"])
        call_short = match_nearest_strike(calls, levels["call_short"])
        call_long = match_nearest_strike(calls, levels["call_long"])

        if not all([put_short, put_long, call_short, call_long]):
            return None

        put_credit = safe_float(put_short.get("mid"), 0.0) - safe_float(put_long.get("mid"), 0.0)
        call_credit = safe_float(call_short.get("mid"), 0.0) - safe_float(call_long.get("mid"), 0.0)
        total_credit = round(put_credit + call_credit, 2)

        put_width = abs(float(put_short["strike"]) - float(put_long["strike"]))
        call_width = abs(float(call_short["strike"]) - float(call_long["strike"]))
        max_width = max(put_width, call_width)
        max_loss = round(max_width - total_credit, 2)

        return {
            "strategy": profile["strategy"],
            "expiration": put_short.get("expiration"),
            "put_short": put_short,
            "put_long": put_long,
            "call_short": call_short,
            "call_long": call_long,
            "estimated_value": total_credit,
            "width": round(max_width, 2),
            "estimated_max_loss": max_loss,
        }

    return None


# =========================================================
# DISPLAY HELPERS
# =========================================================
def show_top_snapshot(result: dict):
    st.subheader("Market Snapshot")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Ticker", result["symbol"])
    r1c2.metric("Spot", fmt_num(result["spot"]))
    r1c3.metric("Sweep", (result["sweep"] or "None").replace("_", " ").title())
    r1c4.metric("Confidence", f'{result["confidence"]}%')

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Support", fmt_num(result["support"]))
    r2c2.metric("Resistance", fmt_num(result["resistance"]))
    r2c3.metric("Rejection", result["rejection"].title())
    r2c4.metric("Volume Spike", yes_no(result["volume_spike"]))

    if result["in_no_trade_window"]:
        st.warning(f"Do not trade right now: {result['reason']}")
    elif result["is_valid"]:
        st.success(result["reason"])
    else:
        st.info(result["reason"])


def show_trade_quality(result: dict, profile: dict):
    st.subheader("Trade Quality")

    q1, q2, q3 = st.columns(3)
    q1.metric("Quality Grade", quality_label(result["confidence"]))
    q2.metric("Setup", quality_text(result["confidence"]))
    q3.metric("Bias", setup_bias_text(profile))

    if result["confidence"] >= 85:
        st.success("This is one of the stronger setups on your scoring model.")
    elif result["confidence"] >= 70:
        st.info("This is a decent setup, but still needs discipline on entry and risk.")
    elif result["confidence"] >= 55:
        st.warning("This is more of a marginal setup. Be careful forcing trades here.")
    else:
        st.warning("Low quality setup. This is usually where getting chopped up starts.")


def show_analysis_summary(result, profile, selected_expiration):
    st.subheader("Setup Summary")

    a1, a2, a3 = st.columns(3)
    a1.metric("Strategy", profile["strategy"])
    a2.metric("Direction Bias", setup_bias_text(profile))
    a3.metric("Expiration", selected_expiration or "N/A")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Spot", fmt_num(result["spot"]))
    b2.metric("Support", fmt_num(result["support"]))
    b3.metric("Resistance", fmt_num(result["resistance"]))
    b4.metric("ATR", fmt_num(result["atr"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sweep", (result["sweep"] or "None").replace("_", " ").title())
    c2.metric("Rejection", result["rejection"].title())
    c3.metric("Volume Spike", yes_no(result["volume_spike"]))
    c4.metric("Confidence", f"{result['confidence']}%")

    mtf = result["mtf"]
    st.caption(
        f"Trend alignment — Daily: {mtf['daily']} | Hourly: {mtf['hourly']} | "
        f"Intraday: {mtf['intraday']} | Aligned: {mtf['aligned']}"
    )

    if mtf.get("manual_override"):
        st.caption("Manual trend override active.")


def show_levels_clean(levels, profile):
    st.subheader("Suggested Trade Levels")

    if not levels:
        st.warning("No levels available for this setup.")
        return

    strategy = profile["strategy"]

    if strategy == "Iron Condor":
        left, right = st.columns(2)

        with left:
            st.markdown("**Put Side**")
            p1, p2 = st.columns(2)
            p1.metric("Short Put", fmt_num(levels.get("put_short")))
            p2.metric("Long Put", fmt_num(levels.get("put_long")))

        with right:
            st.markdown("**Call Side**")
            c1, c2 = st.columns(2)
            c1.metric("Short Call", fmt_num(levels.get("call_short")))
            c2.metric("Long Call", fmt_num(levels.get("call_long")))
        return

    if strategy in {"Put Credit Spread", "Call Credit Spread", "Put Debit Spread", "Call Debit Spread"}:
        l1, l2, l3 = st.columns(3)
        l1.metric("Option Side", str(levels.get("option_side", "")).title())
        l2.metric("Short Strike", fmt_num(levels.get("short_strike")))
        l3.metric("Long Strike", fmt_num(levels.get("long_strike")))
        return

    if strategy in {"Cash Secured Put", "Covered Call"}:
        l1, l2 = st.columns(2)
        l1.metric("Option Side", str(levels.get("option_side", "")).title())
        l2.metric("Strike", fmt_num(levels.get("short_strike")))


def show_options_result(trade_result):
    st.subheader("Options Result")

    if not trade_result:
        st.info("Press Load Options to pull contracts for your chosen setup.")
        return

    strategy = trade_result["strategy"]

    if strategy == "Iron Condor":
        top1, top2, top3 = st.columns(3)
        top1.metric("Estimated Credit", fmt_num(trade_result.get("estimated_value")))
        top2.metric("Width", fmt_num(trade_result.get("width")))
        top3.metric("Estimated Max Loss", fmt_num(trade_result.get("estimated_max_loss")))

        put_df = leg_summary_df(
            ("Short Put", trade_result.get("put_short")),
            ("Long Put", trade_result.get("put_long")),
        )
        call_df = leg_summary_df(
            ("Short Call", trade_result.get("call_short")),
            ("Long Call", trade_result.get("call_long")),
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Put Spread**")
            st.dataframe(put_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Call Spread**")
            st.dataframe(call_df, use_container_width=True, hide_index=True)

    elif strategy in {
        "Put Credit Spread",
        "Call Credit Spread",
        "Put Debit Spread",
        "Call Debit Spread",
    }:
        top1, top2, top3 = st.columns(3)
        label = "Estimated Credit" if "Credit" in strategy else "Estimated Debit"
        top1.metric(label, fmt_num(trade_result.get("estimated_value")))
        top2.metric("Width", fmt_num(trade_result.get("width")))
        top3.metric("Estimated Max Loss", fmt_num(trade_result.get("estimated_max_loss")))

        legs_df = leg_summary_df(
            ("Short Leg", trade_result.get("short_leg")),
            ("Long Leg", trade_result.get("long_leg")),
        )
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

    elif strategy in {"Cash Secured Put", "Covered Call"}:
        top1, top2 = st.columns(2)
        top1.metric("Estimated Premium", fmt_num(trade_result.get("estimated_value")))
        top2.metric("Expiration", str(trade_result.get("expiration", "N/A")))

        leg_df = leg_summary_df(("Short Leg", trade_result.get("short_leg")))
        st.dataframe(leg_df, use_container_width=True, hide_index=True)

    else:
        st.write(trade_result)


# =========================================================
# SESSION DEFAULTS
# =========================================================
defaults = {
    "analysis_result": None,
    "selected_profile": None,
    "selected_levels": None,
    "selected_expiration": None,
    "trade_result": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HEADER / MOBILE-FIRST CONTROLS
# =========================================================
st.title("Low-Call MarketData Spread Finder")
st.caption(f"Time now: {now_et().strftime('%Y-%m-%d %I:%M:%S %p ET')}")

top1, top2 = st.columns(2)
with top1:
    symbol = st.text_input("Ticker", value="SPY").upper().strip()
with top2:
    strategy_choice = st.selectbox(
        "Strategy",
        [
            "Auto",
            "Put Credit Spread",
            "Call Credit Spread",
            "Iron Condor",
            "Put Debit Spread",
            "Call Debit Spread",
            "Cash Secured Put",
            "Covered Call",
        ],
        index=0,
    )

mid1, mid2 = st.columns(2)
with mid1:
    expiration_mode = st.selectbox(
        "Expiration Selection",
        ["Auto", "Nearest Weekly", "14 DTE", "30 DTE"],
        index=0,
    )
with mid2:
    manual_trend_override = st.selectbox(
        "Trend Override",
        ["auto", "bullish", "bearish", "neutral"],
        index=0,
    )

run1, run2, run3 = st.columns(3)
with run1:
    analyze_clicked = st.button("Analyze", use_container_width=True)
with run2:
    load_options_clicked = st.button("Load Options", use_container_width=True)
with run3:
    clear_cache_clicked = st.button("Clear Cache", use_container_width=True)


# =========================================================
# ADVANCED SETTINGS
# =========================================================
with st.expander("Advanced Settings"):
    adv1, adv2 = st.columns(2)
    with adv1:
        lookback = st.slider("Sweep lookback bars", 10, 60, 20)
        atr_period = st.slider("ATR period", 5, 30, 14)
        vol_period = st.slider("Volume MA period", 5, 40, 20)
        vol_spike_threshold = st.slider("Volume spike threshold", 1.0, 3.0, 1.5, 0.1)
    with adv2:
        atr_mult = st.slider("ATR multiplier", 0.25, 2.0, 0.75, 0.05)
        strike_increment = st.selectbox("Strike increment", [0.5, 1.0, 2.5, 5.0], index=1)
        spread_width = st.number_input(
            "Spread width",
            min_value=float(strike_increment),
            value=5.0,
            step=float(strike_increment),
        )

with st.expander("Do-Not-Trade Windows"):
    wnd1, wnd2 = st.columns(2)
    with wnd1:
        avoid_open_minutes = st.slider("Avoid after open (minutes)", 0, 90, 15)
        avoid_close_minutes = st.slider("Avoid before close (minutes)", 0, 120, 60)
        use_lunch_window = st.checkbox("Block lunch window", value=False)
    with wnd2:
        lunch_start_h = st.number_input("Lunch start hour", min_value=10, max_value=14, value=12)
        lunch_start_m = st.number_input("Lunch start minute", min_value=0, max_value=59, value=0)
        lunch_end_h = st.number_input("Lunch end hour", min_value=10, max_value=15, value=13)
        lunch_end_m = st.number_input("Lunch end minute", min_value=0, max_value=59, value=0)
        use_midday_friday = st.checkbox("Block Friday midday", value=True)


# =========================================================
# CACHE CLEAR
# =========================================================
if clear_cache_clicked:
    st.cache_data.clear()
    st.session_state["analysis_result"] = None
    st.session_state["selected_profile"] = None
    st.session_state["selected_levels"] = None
    st.session_state["selected_expiration"] = None
    st.session_state["trade_result"] = None
    st.success("Cache cleared.")


# =========================================================
# ANALYZE
# =========================================================
if analyze_clicked:
    try:
        daily_df, hourly_df, intraday_df = fetch_core_data(symbol)

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
            strike_width=spread_width,
            atr_mult=atr_mult,
            strike_increment=strike_increment,
            manual_trend_override=manual_trend_override,
            no_trade_windows=windows,
            friday_midday_block=friday_midday_block,
        )

        auto_direction = get_auto_direction(result)
        profile = strategy_profile(strategy_choice, auto_direction)
        levels = suggest_trade_levels(
            result=result,
            profile=profile,
            spot=result["spot"],
            width=spread_width,
            atr_mult=atr_mult,
            strike_increment=strike_increment,
        )

        expirations = md_expirations(symbol)
        selected_expiration = choose_expiration(expirations, get_target_dte(expiration_mode))

        st.session_state["analysis_result"] = result
        st.session_state["selected_profile"] = profile
        st.session_state["selected_levels"] = levels
        st.session_state["selected_expiration"] = selected_expiration
        st.session_state["trade_result"] = None

    except Exception as e:
        st.error(f"Analyze failed: {e}")


# =========================================================
# WAIT STATE
# =========================================================
if st.session_state["analysis_result"] is None:
    st.info("Press Analyze to load price data and generate strategy levels.")
    st.stop()


result = st.session_state["analysis_result"]
profile = st.session_state["selected_profile"]
levels = st.session_state["selected_levels"]
selected_expiration = st.session_state["selected_expiration"]


# =========================================================
# DISPLAY
# =========================================================
show_top_snapshot(result)
show_trade_quality(result, profile)
show_analysis_summary(result, profile, selected_expiration)
show_levels_clean(levels, profile)


# =========================================================
# LOAD OPTIONS
# =========================================================
if load_options_clicked:
    try:
        if not selected_expiration:
            st.error("No expiration available.")
        elif not levels:
            st.error("No strategy levels available.")
        else:
            options_data = fetch_strategy_options(
                symbol=symbol,
                expiration=selected_expiration,
                profile=profile,
                levels=levels,
            )
            trade_result = build_trade_from_chain(profile, levels, options_data)
            st.session_state["trade_result"] = trade_result
    except Exception as e:
        st.error(f"Options load failed: {e}")


trade_result = st.session_state.get("trade_result")
show_options_result(trade_result)


# =========================================================
# CHART / TABLE
# =========================================================
st.subheader("Intraday Trend")
chart_df = result["intraday_df"][["Close", "EMA20", "EMA50"]].copy()
st.line_chart(chart_df, use_container_width=True)

with st.expander("View Recent Bars"):
    display_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ATR",
        "vol_ratio",
        "volume_spike",
        "EMA20",
        "EMA50",
    ]
    recent_df = result["intraday_df"][display_cols].tail(30).copy()
    st.dataframe(recent_df, use_container_width=True)


# =========================================================
# NOTES
# =========================================================
with st.expander("How This Saves API Calls"):
    st.write(
        """
- Price data only loads when you press **Analyze**
- Options only load when you press **Load Options**
- The app only pulls the side you asked for
- It does not download full chains on every rerun
"""
    )