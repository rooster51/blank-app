import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Zone-Based Spread App", layout="wide")
st.title("Zone-Based Spread App")
st.caption("Trend + support/resistance + nearest spread suggestion. Manual chain refresh. No sidebar.")


# =========================================================
# DEFAULTS
# =========================================================
DEFAULT_ACCOUNT_SIZE = 500.0
DEFAULT_MAX_RISK_PCT = 0.20
DEFAULT_TOTAL_RISK_CAP_PCT = 0.50
DEFAULT_MIN_DTE = 14
DEFAULT_MAX_DTE = 35
DEFAULT_MAX_WIDTH = 1.0
DEFAULT_MIN_OI = 25
DEFAULT_MIN_VOL = 1
DEFAULT_MAX_BID_ASK_PCT = 0.25
DEFAULT_ATR_ZONE_BUFFER = 0.25
DEFAULT_HISTORY_PERIOD = "1y"


# =========================================================
# HELPERS
# =========================================================
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def parse_date_like(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return pd.NaT


def calc_dte(expiration_value):
    exp = parse_date_like(expiration_value)
    if pd.isna(exp):
        return np.nan
    return (exp - date.today()).days


def option_mid(bid, ask, last=None):
    bid = safe_float(bid, np.nan)
    ask = safe_float(ask, np.nan)
    last = safe_float(last, np.nan)

    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and ask > 0:
        return (bid + ask) / 2.0
    if np.isfinite(last) and last > 0:
        return last
    if np.isfinite(bid) and bid > 0:
        return bid
    if np.isfinite(ask) and ask > 0:
        return ask
    return np.nan


def pct_bid_ask_spread(bid, ask):
    bid = safe_float(bid, np.nan)
    ask = safe_float(ask, np.nan)
    if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and ask >= bid:
        return (ask - bid) / ask
    return np.nan


def liquidity_ok(row, max_bid_ask_pct=DEFAULT_MAX_BID_ASK_PCT, min_oi=DEFAULT_MIN_OI, min_volume=DEFAULT_MIN_VOL):
    oi_ok = safe_float(row.get("oi"), 0) >= min_oi
    vol_ok = safe_float(row.get("volume"), 0) >= min_volume
    ba = row.get("spread_pct", np.nan)
    ba_ok = np.isnan(ba) or ba <= max_bid_ask_pct
    mid_ok = np.isfinite(row.get("mid", np.nan)) and row.get("mid", np.nan) > 0
    return oi_ok and vol_ok and ba_ok and mid_ok


def contracts_allowed(max_risk_per_trade, account_size, total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT):
    max_risk_per_trade = safe_float(max_risk_per_trade, np.nan)
    account_size = safe_float(account_size, np.nan)
    if not np.isfinite(max_risk_per_trade) or not np.isfinite(account_size) or max_risk_per_trade <= 0:
        return 0
    cap = account_size * total_risk_cap_pct
    return max(0, int(cap // max_risk_per_trade))


def project_growth_fixed(start_balance: float, avg_win: float, avg_loss: float, win_rate: float, num_trades: int):
    balance = safe_float(start_balance, 0.0)
    win_rate = safe_float(win_rate, 0.0)

    rows = []
    for i in range(1, num_trades + 1):
        expected_trade_pl = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        balance += expected_trade_pl
        rows.append(
            {
                "trade_num": i,
                "expected_trade_pl": round(expected_trade_pl, 2),
                "expected_balance": round(balance, 2),
            }
        )
    return pd.DataFrame(rows)


def price_location_vs_zones(price: float, zones: dict) -> str:
    demand_low = safe_float(zones.get("demand_low"), np.nan)
    demand_high = safe_float(zones.get("demand_high"), np.nan)
    supply_low = safe_float(zones.get("supply_low"), np.nan)
    supply_high = safe_float(zones.get("supply_high"), np.nan)

    if np.isfinite(demand_low) and np.isfinite(demand_high) and demand_low <= price <= demand_high:
        return "Inside Demand"
    if np.isfinite(supply_low) and np.isfinite(supply_high) and supply_low <= price <= supply_high:
        return "Inside Supply"
    return "Outside Zones"


# =========================================================
# YFINANCE DATA
# =========================================================
@st.cache_data(ttl=1800)
def fetch_stock_candles(symbol: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, auto_adjust=False)

    if hist is None or hist.empty:
        return pd.DataFrame()

    df = hist.reset_index().copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "date": "t",
        "datetime": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = ["t", "o", "h", "l", "c", "v"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[needed].sort_values("t").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_filtered_expirations(symbol: str, min_dte: int, max_dte: int) -> list[str]:
    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options or [])
    if not expirations:
        return []

    filtered = []
    for exp in expirations:
        dte = calc_dte(exp)
        if np.isfinite(dte) and min_dte <= dte <= max_dte:
            filtered.append(exp)

    return filtered


@st.cache_data(ttl=900)
def fetch_option_chain_for_expiration(symbol: str, expiration: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiration)

    parts = []

    if hasattr(chain, "calls") and chain.calls is not None and not chain.calls.empty:
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        parts.append(calls)

    if hasattr(chain, "puts") and chain.puts is not None and not chain.puts.empty:
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        parts.append(puts)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)

    rename_map = {
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "lastPrice": "last",
        "lastprice": "last",
        "impliedVolatility": "iv",
        "impliedvolatility": "iv",
        "openInterest": "oi",
        "openinterest": "oi",
        "volume": "volume",
        "contractSymbol": "optionSymbol",
        "contractsymbol": "optionSymbol",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["symbol"] = symbol.upper()
    df["expiration"] = expiration
    df["dte"] = calc_dte(expiration)

    for col in ["strike", "bid", "ask", "last", "iv", "oi", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "delta" not in df.columns:
        df["delta"] = np.nan

    if "optionSymbol" not in df.columns:
        df["optionSymbol"] = None

    df["mid"] = df.apply(lambda r: option_mid(r["bid"], r["ask"], r["last"]), axis=1)
    df["spread_pct"] = df.apply(lambda r: pct_bid_ask_spread(r["bid"], r["ask"]), axis=1)
    df["delta_abs"] = df["delta"].abs()

    keep_cols = [
        "symbol",
        "expiration",
        "dte",
        "option_type",
        "strike",
        "bid",
        "ask",
        "mid",
        "last",
        "delta",
        "delta_abs",
        "iv",
        "oi",
        "volume",
        "spread_pct",
        "optionSymbol",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols].sort_values(["option_type", "strike"]).reset_index(drop=True)


# =========================================================
# TREND + ZONES
# =========================================================
def add_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles

    df = candles.copy()
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()

    prev_close = df["c"].shift(1)
    tr1 = df["h"] - df["l"]
    tr2 = (df["h"] - prev_close).abs()
    tr3 = (df["l"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()

    return df


def classify_trend(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50:
        return {
            "trend": "Unknown",
            "strength": "Unknown",
            "last_close": np.nan,
            "ema20": np.nan,
            "ema50": np.nan,
            "atr14": np.nan,
        }

    last = df.iloc[-1]
    last_close = safe_float(last["c"], np.nan)
    ema20 = safe_float(last["ema20"], np.nan)
    ema50 = safe_float(last["ema50"], np.nan)
    atr14 = safe_float(last["atr14"], np.nan)

    if np.isfinite(last_close) and np.isfinite(ema20) and np.isfinite(ema50):
        if last_close > ema20 > ema50:
            trend = "Uptrend"
        elif last_close < ema20 < ema50:
            trend = "Downtrend"
        else:
            trend = "Neutral"
    else:
        trend = "Unknown"

    spread = abs(ema20 - ema50) if np.isfinite(ema20) and np.isfinite(ema50) else np.nan
    if np.isfinite(spread) and np.isfinite(last_close) and last_close > 0:
        ratio = spread / last_close
        if ratio >= 0.02:
            strength = "Strong"
        elif ratio >= 0.007:
            strength = "Moderate"
        else:
            strength = "Weak"
    else:
        strength = "Unknown"

    return {
        "trend": trend,
        "strength": strength,
        "last_close": round(last_close, 2) if np.isfinite(last_close) else np.nan,
        "ema20": round(ema20, 2) if np.isfinite(ema20) else np.nan,
        "ema50": round(ema50, 2) if np.isfinite(ema50) else np.nan,
        "atr14": round(atr14, 2) if np.isfinite(atr14) else np.nan,
    }


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    if df.empty or len(df) < left + right + 5:
        return pd.DataFrame()

    rows = []
    highs = df["h"].values
    lows = df["l"].values

    for i in range(left, len(df) - right):
        high_window = highs[i - left : i + right + 1]
        low_window = lows[i - left : i + right + 1]

        if highs[i] == np.max(high_window):
            rows.append({"idx": i, "type": "high", "price": highs[i], "time": df.iloc[i]["t"]})
        if lows[i] == np.min(low_window):
            rows.append({"idx": i, "type": "low", "price": lows[i], "time": df.iloc[i]["t"]})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def detect_supply_demand_zones(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50:
        return {
            "demand_low": np.nan,
            "demand_high": np.nan,
            "supply_low": np.nan,
            "supply_high": np.nan,
            "atr14": np.nan,
        }

    pivots = find_pivots(df)
    atr14 = safe_float(df["atr14"].iloc[-1], np.nan)
    if pivots.empty or not np.isfinite(atr14):
        return {
            "demand_low": np.nan,
            "demand_high": np.nan,
            "supply_low": np.nan,
            "supply_high": np.nan,
            "atr14": atr14,
        }

    demand_low = np.nan
    demand_high = np.nan
    supply_low = np.nan
    supply_high = np.nan

    recent_lows = pivots[pivots["type"] == "low"].tail(8)
    for _, row in recent_lows.iloc[::-1].iterrows():
        idx = int(row["idx"])
        pivot_price = safe_float(row["price"], np.nan)
        if not np.isfinite(pivot_price):
            continue
        future_slice = df.iloc[idx + 1 : min(idx + 11, len(df))]
        if future_slice.empty:
            continue
        rally = safe_float(future_slice["h"].max(), np.nan) - pivot_price
        if np.isfinite(rally) and rally >= atr14 * 1.5:
            demand_low = pivot_price - (atr14 * 0.20)
            demand_high = pivot_price + (atr14 * 0.35)
            break

    recent_highs = pivots[pivots["type"] == "high"].tail(8)
    for _, row in recent_highs.iloc[::-1].iterrows():
        idx = int(row["idx"])
        pivot_price = safe_float(row["price"], np.nan)
        if not np.isfinite(pivot_price):
            continue
        future_slice = df.iloc[idx + 1 : min(idx + 11, len(df))]
        if future_slice.empty:
            continue
        drop = pivot_price - safe_float(future_slice["l"].min(), np.nan)
        if np.isfinite(drop) and drop >= atr14 * 1.5:
            supply_low = pivot_price - (atr14 * 0.35)
            supply_high = pivot_price + (atr14 * 0.20)
            break

    return {
        "demand_low": round(demand_low, 2) if np.isfinite(demand_low) else np.nan,
        "demand_high": round(demand_high, 2) if np.isfinite(demand_high) else np.nan,
        "supply_low": round(supply_low, 2) if np.isfinite(supply_low) else np.nan,
        "supply_high": round(supply_high, 2) if np.isfinite(supply_high) else np.nan,
        "atr14": round(atr14, 2) if np.isfinite(atr14) else np.nan,
    }


# =========================================================
# SUGGESTED STRIKE LOGIC
# =========================================================
def find_nearest_zone_based_spread(
    chain: pd.DataFrame,
    trend: str,
    zones: dict,
    width: float = 1.0,
    atr_buffer_mult: float = 0.25,
    require_liquidity: bool = False,
):
    if chain.empty:
        return None

    df = chain.copy()
    if require_liquidity:
        df = df[df.apply(liquidity_ok, axis=1)].copy()

    if df.empty:
        return None

    atr14 = safe_float(zones.get("atr14"), np.nan)
    demand_low = safe_float(zones.get("demand_low"), np.nan)
    demand_high = safe_float(zones.get("demand_high"), np.nan)
    supply_low = safe_float(zones.get("supply_low"), np.nan)
    supply_high = safe_float(zones.get("supply_high"), np.nan)

    if trend == "Uptrend":
        if not (np.isfinite(demand_low) and np.isfinite(atr14)):
            return None

        target_short = demand_low - (atr14 * atr_buffer_mult)
        puts = df[df["option_type"] == "put"].copy().sort_values("strike", ascending=False)

        short_candidates = puts[puts["strike"] < target_short].copy()
        if short_candidates.empty:
            return {
                "strategy": "Bull Put",
                "zone_type": "Support / Demand",
                "zone_low": demand_low,
                "zone_high": demand_high,
                "target_short_strike": round(target_short, 2),
                "short_strike": np.nan,
                "long_strike": np.nan,
                "short_delta": np.nan,
                "credit": np.nan,
                "max_risk": np.nan,
                "expiration": None,
                "dte": np.nan,
                "reason": "No listed put short strike below support buffer for the selected expiration.",
            }

        short_row = short_candidates.iloc[0]
        short_strike = safe_float(short_row["strike"], np.nan)

        long_candidates = puts[puts["strike"] < short_strike].copy().sort_values("strike", ascending=False)
        if long_candidates.empty:
            return None

        exact = long_candidates[np.isclose(short_strike - long_candidates["strike"], width)]
        long_row = exact.iloc[0] if not exact.empty else long_candidates.iloc[0]
        long_strike = safe_float(long_row["strike"], np.nan)

        if abs(short_strike - long_strike) > width:
            return None

        short_mid = safe_float(short_row["mid"], np.nan)
        long_mid = safe_float(long_row["mid"], np.nan)
        if not np.isfinite(short_mid) or not np.isfinite(long_mid):
            return None

        credit = (short_mid - long_mid) * 100.0
        max_risk = (abs(short_strike - long_strike) - (short_mid - long_mid)) * 100.0

        return {
            "strategy": "Bull Put",
            "zone_type": "Support / Demand",
            "zone_low": demand_low,
            "zone_high": demand_high,
            "target_short_strike": round(target_short, 2),
            "short_strike": short_strike,
            "long_strike": long_strike,
            "short_delta": safe_float(short_row.get("delta"), np.nan),
            "credit": round(credit, 2),
            "max_risk": round(max_risk, 2),
            "expiration": short_row.get("expiration"),
            "dte": short_row.get("dte"),
            "reason": "Nearest listed put short strike below support buffer.",
        }

    if trend == "Downtrend":
        if not (np.isfinite(supply_high) and np.isfinite(atr14)):
            return None

        target_short = supply_high + (atr14 * atr_buffer_mult)
        calls = df[df["option_type"] == "call"].copy().sort_values("strike", ascending=True)

        short_candidates = calls[calls["strike"] > target_short].copy()
        if short_candidates.empty:
            return {
                "strategy": "Bear Call",
                "zone_type": "Resistance / Supply",
                "zone_low": supply_low,
                "zone_high": supply_high,
                "target_short_strike": round(target_short, 2),
                "short_strike": np.nan,
                "long_strike": np.nan,
                "short_delta": np.nan,
                "credit": np.nan,
                "max_risk": np.nan,
                "expiration": None,
                "dte": np.nan,
                "reason": "No listed call short strike above resistance buffer for the selected expiration.",
            }

        short_row = short_candidates.iloc[0]
        short_strike = safe_float(short_row["strike"], np.nan)

        long_candidates = calls[calls["strike"] > short_strike].copy().sort_values("strike", ascending=True)
        if long_candidates.empty:
            return None

        exact = long_candidates[np.isclose(long_candidates["strike"] - short_strike, width)]
        long_row = exact.iloc[0] if not exact.empty else long_candidates.iloc[0]
        long_strike = safe_float(long_row["strike"], np.nan)

        if abs(short_strike - long_strike) > width:
            return None

        short_mid = safe_float(short_row["mid"], np.nan)
        long_mid = safe_float(long_row["mid"], np.nan)
        if not np.isfinite(short_mid) or not np.isfinite(long_mid):
            return None

        credit = (short_mid - long_mid) * 100.0
        max_risk = (abs(short_strike - long_strike) - (short_mid - long_mid)) * 100.0

        return {
            "strategy": "Bear Call",
            "zone_type": "Resistance / Supply",
            "zone_low": supply_low,
            "zone_high": supply_high,
            "target_short_strike": round(target_short, 2),
            "short_strike": short_strike,
            "long_strike": long_strike,
            "short_delta": safe_float(short_row.get("delta"), np.nan),
            "credit": round(credit, 2),
            "max_risk": round(max_risk, 2),
            "expiration": short_row.get("expiration"),
            "dte": short_row.get("dte"),
            "reason": "Nearest listed call short strike above resistance buffer.",
        }

    return None


def find_backup_zone_strikes(
    chain: pd.DataFrame,
    trend: str,
    zones: dict,
    atr_buffer_mult: float,
    require_liquidity: bool = False,
    limit: int = 5,
):
    if chain.empty:
        return pd.DataFrame()

    df = chain.copy()
    if require_liquidity:
        df = df[df.apply(liquidity_ok, axis=1)].copy()

    atr14 = safe_float(zones.get("atr14"), np.nan)
    demand_low = safe_float(zones.get("demand_low"), np.nan)
    supply_high = safe_float(zones.get("supply_high"), np.nan)

    if trend == "Uptrend" and np.isfinite(demand_low) and np.isfinite(atr14):
        target_short = demand_low - (atr14 * atr_buffer_mult)
        puts = df[df["option_type"] == "put"].copy()
        puts = puts[puts["strike"] < target_short].sort_values("strike", ascending=False).head(limit)
        if puts.empty:
            return pd.DataFrame()
        return puts[["expiration", "dte", "strike", "delta", "mid", "oi", "volume"]].rename(
            columns={"strike": "candidate_short_strike", "delta": "candidate_delta", "mid": "candidate_mid"}
        )

    if trend == "Downtrend" and np.isfinite(supply_high) and np.isfinite(atr14):
        target_short = supply_high + (atr14 * atr_buffer_mult)
        calls = df[df["option_type"] == "call"].copy()
        calls = calls[calls["strike"] > target_short].sort_values("strike", ascending=True).head(limit)
        if calls.empty:
            return pd.DataFrame()
        return calls[["expiration", "dte", "strike", "delta", "mid", "oi", "volume"]].rename(
            columns={"strike": "candidate_short_strike", "delta": "candidate_delta", "mid": "candidate_mid"}
        )

    return pd.DataFrame()


# =========================================================
# CACHE LOADER
# =========================================================
@st.cache_data(ttl=1800)
def load_dashboard_symbol(symbol: str):
    candles_raw = fetch_stock_candles(symbol, period=DEFAULT_HISTORY_PERIOD)
    candles = add_indicators(candles_raw)
    trend_info = classify_trend(candles)
    zones = detect_supply_demand_zones(candles)
    expirations = get_filtered_expirations(symbol, min_dte=DEFAULT_MIN_DTE, max_dte=DEFAULT_MAX_DTE)
    return candles, trend_info, zones, expirations


# =========================================================
# SETTINGS PANEL
# =========================================================
with st.expander("Scanner Settings", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        symbol_input = st.text_input("Symbols (comma separated)", value="SPY,QQQ,IWM,AMD,AAPL")
        account_size = st.number_input("Account Size", min_value=100.0, value=DEFAULT_ACCOUNT_SIZE, step=50.0)
        max_risk_pct = st.slider("Max Risk Per Trade (% of acct)", 0.05, 0.40, DEFAULT_MAX_RISK_PCT, 0.01)
        require_liquidity = st.checkbox("Require Liquidity Filter", value=False)

    with c2:
        min_dte = st.number_input("Min DTE", min_value=1, value=DEFAULT_MIN_DTE, step=1)
        max_dte = st.number_input("Max DTE", min_value=1, value=DEFAULT_MAX_DTE, step=1)
        max_width = st.selectbox("Max Spread Width", options=[1.0, 2.0, 3.0], index=0)
        atr_zone_buffer = st.slider("ATR Buffer Outside Zone", 0.10, 1.00, DEFAULT_ATR_ZONE_BUFFER, 0.05)

run_scan = st.button("Load Data / Scan", use_container_width=True)

# session state for selected chain reuse
if "selected_symbol_chain" not in st.session_state:
    st.session_state.selected_symbol_chain = pd.DataFrame()
if "selected_symbol_chain_symbol" not in st.session_state:
    st.session_state.selected_symbol_chain_symbol = None
if "selected_symbol_chain_expiration" not in st.session_state:
    st.session_state.selected_symbol_chain_expiration = None
if "selected_chain_loaded" not in st.session_state:
    st.session_state.selected_chain_loaded = False


# =========================================================
# MAIN
# =========================================================
tabs = st.tabs(["Dashboard", "Suggested Spread", "Growth", "Debug"])
symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

if not run_scan:
    with tabs[0]:
        st.info("Enter your symbols and click 'Load Data / Scan'.")
else:
    try:
        all_candles = {}
        all_trends = {}
        all_zones = {}
        all_expirations = {}

        for sym in symbols:
            candles, trend_info, zones, expirations = load_dashboard_symbol(sym)
            expirations = [e for e in expirations if np.isfinite(calc_dte(e)) and min_dte <= calc_dte(e) <= max_dte]
            all_candles[sym] = candles
            all_trends[sym] = trend_info
            all_zones[sym] = zones
            all_expirations[sym] = expirations

        # DASHBOARD
        with tabs[0]:
            st.subheader("Trend + Zone Dashboard")

            rows = []
            for sym in symbols:
                trend_info = all_trends.get(sym, {})
                zones = all_zones.get(sym, {})
                price = safe_float(trend_info.get("last_close"), np.nan)

                rows.append(
                    {
                        "symbol": sym,
                        "trend": trend_info.get("trend"),
                        "strength": trend_info.get("strength"),
                        "last_close": trend_info.get("last_close"),
                        "ema20": trend_info.get("ema20"),
                        "ema50": trend_info.get("ema50"),
                        "demand_low": zones.get("demand_low"),
                        "demand_high": zones.get("demand_high"),
                        "supply_low": zones.get("supply_low"),
                        "supply_high": zones.get("supply_high"),
                        "location": price_location_vs_zones(price, zones),
                    }
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            selected_symbol = st.selectbox("View Ticker Detail", options=symbols, index=0, key="dashboard_symbol")
            candles = all_candles.get(selected_symbol, pd.DataFrame())
            trend_info = all_trends.get(selected_symbol, {})
            zones = all_zones.get(selected_symbol, {})

            if candles.empty:
                st.info("No candle data available.")
            else:
                chart_df = candles[["t", "c", "ema20", "ema50"]].copy().set_index("t")
                st.line_chart(chart_df)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Trend", trend_info.get("trend", "Unknown"))
                with c2:
                    st.metric("Strength", trend_info.get("strength", "Unknown"))
                with c3:
                    st.metric("ATR14", trend_info.get("atr14", np.nan))

                z1, z2 = st.columns(2)
                with z1:
                    st.markdown("**Demand / Support Zone**")
                    st.write(f"{zones.get('demand_low')} to {zones.get('demand_high')}")
                with z2:
                    st.markdown("**Supply / Resistance Zone**")
                    st.write(f"{zones.get('supply_low')} to {zones.get('supply_high')}")

        # SUGGESTED SPREAD
        with tabs[1]:
            st.subheader("Suggested Spread to Look At")

            spread_symbol = st.selectbox("Ticker", options=symbols, index=0, key="spread_symbol")
            trend_info = all_trends.get(spread_symbol, {})
            zones = all_zones.get(spread_symbol, {})
            expirations = all_expirations.get(spread_symbol, [])

            trend = trend_info.get("trend", "Unknown")
            strength = trend_info.get("strength", "Unknown")
            price = safe_float(trend_info.get("last_close"), np.nan)

            st.write(f"**Trend:** {trend} ({strength})")
            st.write(f"**Current Price:** {price}")
            st.write(f"**Demand / Support Zone:** {zones.get('demand_low')} to {zones.get('demand_high')}")
            st.write(f"**Supply / Resistance Zone:** {zones.get('supply_low')} to {zones.get('supply_high')}")

            if not expirations:
                st.info("No expirations found in the selected DTE range.")
            else:
                selected_expiration = st.selectbox(
                    "Expiration",
                    options=expirations,
                    index=0,
                    key="selected_expiration",
                )

                c_refresh, c_clear = st.columns(2)
                with c_refresh:
                    refresh_chain = st.button("Refresh Selected Chain", use_container_width=True)
                with c_clear:
                    clear_chain = st.button("Clear Cached Chain", use_container_width=True)

                if clear_chain:
                    st.session_state.selected_symbol_chain = pd.DataFrame()
                    st.session_state.selected_symbol_chain_symbol = None
                    st.session_state.selected_symbol_chain_expiration = None
                    st.session_state.selected_chain_loaded = False

                symbol_changed = st.session_state.selected_symbol_chain_symbol != spread_symbol
                expiration_changed = st.session_state.selected_symbol_chain_expiration != selected_expiration
                needs_first_load = not st.session_state.selected_chain_loaded

                if refresh_chain or symbol_changed or expiration_changed or needs_first_load:
                    with st.spinner("Loading selected option chain..."):
                        chain = fetch_option_chain_for_expiration(spread_symbol, selected_expiration)
                    st.session_state.selected_symbol_chain = chain
                    st.session_state.selected_symbol_chain_symbol = spread_symbol
                    st.session_state.selected_symbol_chain_expiration = selected_expiration
                    st.session_state.selected_chain_loaded = True
                else:
                    chain = st.session_state.selected_symbol_chain.copy()

                if chain.empty:
                    st.info("No option chain returned for the selected expiration.")
                else:
                    st.caption(
                        f"Using cached chain for {st.session_state.selected_symbol_chain_symbol} "
                        f"{st.session_state.selected_symbol_chain_expiration}"
                    )

                    suggestion = find_nearest_zone_based_spread(
                        chain=chain,
                        trend=trend,
                        zones=zones,
                        width=max_width,
                        atr_buffer_mult=atr_zone_buffer,
                        require_liquidity=require_liquidity,
                    )

                    backups = find_backup_zone_strikes(
                        chain=chain,
                        trend=trend,
                        zones=zones,
                        atr_buffer_mult=atr_zone_buffer,
                        require_liquidity=require_liquidity,
                        limit=5,
                    )

                    diagnostics = {}
                    atr14 = safe_float(zones.get("atr14"), np.nan)

                    if trend == "Uptrend":
                        demand_low = safe_float(zones.get("demand_low"), np.nan)
                        if np.isfinite(demand_low) and np.isfinite(atr14):
                            diagnostics["target_short_below"] = round(demand_low - (atr14 * atr_zone_buffer), 2)
                        put_chain = chain[chain["option_type"] == "put"]
                        if not put_chain.empty:
                            diagnostics["lowest_put_returned"] = safe_float(put_chain["strike"].min(), np.nan)
                            diagnostics["highest_put_returned"] = safe_float(put_chain["strike"].max(), np.nan)

                    elif trend == "Downtrend":
                        supply_high = safe_float(zones.get("supply_high"), np.nan)
                        if np.isfinite(supply_high) and np.isfinite(atr14):
                            diagnostics["target_short_above"] = round(supply_high + (atr14 * atr_zone_buffer), 2)
                        call_chain = chain[chain["option_type"] == "call"]
                        if not call_chain.empty:
                            diagnostics["lowest_call_returned"] = safe_float(call_chain["strike"].min(), np.nan)
                            diagnostics["highest_call_returned"] = safe_float(call_chain["strike"].max(), np.nan)

                    st.write("Diagnostics:", diagnostics)

                    if suggestion is None:
                        st.info("No valid zone-based spread suggestion found.")
                    else:
                        if pd.isna(suggestion["short_strike"]):
                            st.warning(suggestion["reason"])
                        else:
                            qty = contracts_allowed(
                                max_risk_per_trade=suggestion["max_risk"],
                                account_size=account_size,
                                total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT,
                            )

                            st.markdown("### Main Suggestion")
                            st.write(
                                f"""
**Strategy:** {suggestion['strategy']}  
**Zone Type:** {suggestion['zone_type']}  
**Zone:** {suggestion['zone_low']} to {suggestion['zone_high']}  
**Target Short Strike:** {suggestion['target_short_strike']}  
**Suggested Short Strike:** {suggestion['short_strike']}  
**Suggested Long Strike:** {suggestion['long_strike']}  
**Expiration:** {suggestion['expiration']}  
**DTE:** {int(suggestion['dte']) if pd.notna(suggestion['dte']) else 'N/A'}  
**Short Delta:** {suggestion['short_delta']:.3f}  
**Credit:** ${suggestion['credit']:.2f}  
**Max Risk:** ${suggestion['max_risk']:.2f}  
**Reason:** {suggestion['reason']}  
**Contracts Allowed:** {qty}
"""
                            )

                    st.markdown("### Backup Short Strikes to Look At")
                    if backups.empty:
                        st.info("No backup strikes found.")
                    else:
                        st.dataframe(backups, use_container_width=True, hide_index=True)

        # GROWTH
        with tabs[2]:
            st.subheader("Growth Projection")

            growth_symbol = st.selectbox("Growth Model Ticker", options=symbols, index=0, key="growth_symbol")
            assumed_win_rate = st.slider("Assumed Win Rate", 0.40, 0.95, 0.75, 0.01)
            growth_trades = st.number_input("Number of Trades", min_value=1, value=25, step=1)

            trend_info = all_trends.get(growth_symbol, {})
            zones = all_zones.get(growth_symbol, {})
            trend = trend_info.get("trend", "Unknown")

            growth_chain = pd.DataFrame()
            if (
                st.session_state.selected_chain_loaded
                and st.session_state.selected_symbol_chain_symbol == growth_symbol
            ):
                growth_chain = st.session_state.selected_symbol_chain.copy()

            suggestion = find_nearest_zone_based_spread(
                chain=growth_chain,
                trend=trend,
                zones=zones,
                width=max_width,
                atr_buffer_mult=atr_zone_buffer,
                require_liquidity=require_liquidity,
            )

            if suggestion is None or pd.isna(suggestion.get("credit")) or pd.isna(suggestion.get("max_risk")):
                st.info("Select this ticker and load its chain in Suggested Spread first.")
            else:
                fixed_df = project_growth_fixed(
                    start_balance=account_size,
                    avg_win=suggestion["credit"],
                    avg_loss=suggestion["max_risk"],
                    win_rate=assumed_win_rate,
                    num_trades=int(growth_trades),
                )

                end_balance = fixed_df.iloc[-1]["expected_balance"] if not fixed_df.empty else account_size
                ev_per_trade = (assumed_win_rate * suggestion["credit"]) - ((1 - assumed_win_rate) * suggestion["max_risk"])

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Starting Balance", f"${account_size:,.2f}")
                with c2:
                    st.metric("Expected P/L Per Trade", f"${ev_per_trade:,.2f}")
                with c3:
                    st.metric("Projected End Balance", f"${end_balance:,.2f}")

                st.markdown("### Model Spread Used")
                st.write(
                    f"""
**{growth_symbol} — {suggestion['strategy']}**  
Short Strike: **{suggestion['short_strike']}**  
Long Strike: **{suggestion['long_strike']}**  
Credit: **${suggestion['credit']:.2f}**  
Max Risk: **${suggestion['max_risk']:.2f}**
"""
                )

                st.dataframe(fixed_df, use_container_width=True, hide_index=True)

        # DEBUG
        with tabs[3]:
            st.subheader("Debug")

            debug_symbol = st.selectbox("Debug Symbol", options=symbols, index=0, key="debug_symbol")
            st.write("Trend Info")
            st.json(all_trends.get(debug_symbol, {}))

            st.write("Zones")
            st.json(all_zones.get(debug_symbol, {}))

            expirations = all_expirations.get(debug_symbol, [])
            st.write("Filtered expirations:", expirations)

            if (
                st.session_state.selected_chain_loaded
                and st.session_state.selected_symbol_chain_symbol == debug_symbol
                and not st.session_state.selected_symbol_chain.empty
            ):
                chain = st.session_state.selected_symbol_chain.copy()
                st.write(
                    f"Using cached chain for {st.session_state.selected_symbol_chain_symbol} "
                    f"{st.session_state.selected_symbol_chain_expiration}"
                )
            else:
                chain = pd.DataFrame()
                st.write("No cached chain for this symbol yet. Load it in Suggested Spread first.")

            if chain.empty:
                st.info("No cached chain data.")
            else:
                debug_cols = [
                    c for c in [
                        "symbol", "expiration", "dte", "option_type", "strike",
                        "bid", "ask", "mid", "last", "delta", "delta_abs",
                        "iv", "oi", "volume", "spread_pct", "optionSymbol"
                    ] if c in chain.columns
                ]
                st.dataframe(chain[debug_cols], use_container_width=True, hide_index=True)

                st.write("Chain row count:", len(chain))
                st.write("Option types:", chain["option_type"].value_counts(dropna=False).to_dict())
                st.write(
                    "DTE range:",
                    {
                        "min": safe_float(chain["dte"].min(), np.nan),
                        "max": safe_float(chain["dte"].max(), np.nan),
                    },
                )

    except Exception as e:
        st.exception(e)