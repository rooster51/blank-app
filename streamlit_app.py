import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Trend + Zone Spread App", layout="wide")
st.title("Trend + Zone Spread App")
st.caption("Trend-first spread suggestions using supply/demand zones. No sidebar.")


# =========================================================
# DEFAULTS
# =========================================================
DEFAULT_ACCOUNT_SIZE = 500.0
DEFAULT_MAX_RISK_PCT = 0.20
DEFAULT_TOTAL_RISK_CAP_PCT = 0.50
DEFAULT_MIN_DTE = 14
DEFAULT_MAX_DTE = 35
DEFAULT_MAX_WIDTH = 1.0
DEFAULT_STRIKE_LIMIT = 30
DEFAULT_TAKE_PROFIT_PCT = 0.50
DEFAULT_MIN_OI = 25
DEFAULT_MIN_VOL = 1
DEFAULT_MAX_BID_ASK_PCT = 0.20
DEFAULT_ATR_ZONE_BUFFER = 0.25
DEFAULT_API_TIMEOUT = 30


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


def approx_pop_from_short_delta(delta_abs: float) -> float:
    if not np.isfinite(delta_abs):
        return np.nan
    return max(0.0, min(1.0, 1.0 - abs(delta_abs)))


def expected_value(credit_dollars, max_risk_dollars, pop):
    if not all(np.isfinite(v) for v in [credit_dollars, max_risk_dollars, pop]):
        return np.nan
    return (pop * credit_dollars) - ((1.0 - pop) * max_risk_dollars)


def spread_score(credit_dollars, max_risk_dollars, pop):
    if not all(np.isfinite(v) for v in [credit_dollars, max_risk_dollars, pop]):
        return np.nan
    if max_risk_dollars <= 0:
        return np.nan
    return (credit_dollars / max_risk_dollars) * pop


def contracts_allowed(max_risk_per_trade, account_size, total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT):
    max_risk_per_trade = safe_float(max_risk_per_trade, np.nan)
    account_size = safe_float(account_size, np.nan)
    if not np.isfinite(max_risk_per_trade) or not np.isfinite(account_size) or max_risk_per_trade <= 0:
        return 0
    cap = account_size * total_risk_cap_pct
    return max(0, int(cap // max_risk_per_trade))


def payload_to_frame(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or len(payload) == 0:
        return pd.DataFrame()

    normalized = {}
    max_len = 0

    for key, value in payload.items():
        if isinstance(value, list):
            normalized[key] = value
            max_len = max(max_len, len(value))
        else:
            normalized[key] = [value]
            max_len = max(max_len, 1)

    for key, value in normalized.items():
        if len(value) < max_len:
            normalized[key] = value + [None] * (max_len - len(value))

    return pd.DataFrame(normalized)


def project_growth_fixed(start_balance: float, avg_win: float, avg_loss: float, win_rate: float, num_trades: int):
    balance = safe_float(start_balance, 0.0)
    win_rate = safe_float(win_rate, 0.0)

    history = []
    for i in range(1, num_trades + 1):
        expected_trade_pl = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        balance += expected_trade_pl
        history.append(
            {
                "trade_num": i,
                "expected_balance": round(balance, 2),
                "expected_trade_pl": round(expected_trade_pl, 2),
            }
        )

    return pd.DataFrame(history)


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
# API - MARKETDATA.APP
# =========================================================
def fetch_stock_candles(symbol: str, api_key: str, countback: int = 120) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.marketdata.app/v1/stocks/candles/D/{symbol.upper()}/"
    params = {
        "countback": countback,
        "dateformat": "timestamp",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=DEFAULT_API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, dict) or data.get("s") != "ok":
        return pd.DataFrame()

    candles = pd.DataFrame(
        {
            "t": data.get("t", []),
            "o": data.get("o", []),
            "h": data.get("h", []),
            "l": data.get("l", []),
            "c": data.get("c", []),
            "v": data.get("v", []),
        }
    )

    if candles.empty:
        return candles

    candles["t"] = pd.to_datetime(candles["t"], unit="s", errors="coerce")
    for col in ["o", "h", "l", "c", "v"]:
        candles[col] = pd.to_numeric(candles[col], errors="coerce")

    return candles.sort_values("t").reset_index(drop=True)


def fetch_option_chain(
    symbol: str,
    api_key: str,
    min_dte: int = DEFAULT_MIN_DTE,
    max_dte: int = DEFAULT_MAX_DTE,
    strike_limit: int = DEFAULT_STRIKE_LIMIT,
) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.marketdata.app/v1/options/chain/{symbol.upper()}/"

    today = date.today()
    from_date = today + timedelta(days=int(min_dte))
    to_date = today + timedelta(days=int(max_dte))

    params = {
        "dateformat": "timestamp",
        "nonstandard": "false",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "strikeLimit": strike_limit,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=DEFAULT_API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    df = payload_to_frame(data)
    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "underlying": "symbol",
        "expiration": "expiration",
        "side": "option_type",
        "type": "option_type",
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "mid": "mid",
        "last": "last",
        "delta": "delta",
        "iv": "iv",
        "openInterest": "oi",
        "volume": "volume",
        "updated": "timestamp",
        "optionSymbol": "optionSymbol",
        "dte": "dte",
        "underlyingPrice": "underlyingPrice",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "symbol" not in df.columns:
        df["symbol"] = symbol.upper()

    needed_cols = [
        "symbol",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "mid",
        "last",
        "delta",
        "iv",
        "oi",
        "volume",
        "timestamp",
        "optionSymbol",
        "dte",
        "underlyingPrice",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"p": "put", "c": "call"})
    )

    return df[needed_cols].copy()


# =========================================================
# NORMALIZATION
# =========================================================
def normalize_chain(chain: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    df = chain.copy()

    rename_map = {
        "type": "option_type",
        "side": "option_type",
        "right": "option_type",
        "expiry": "expiration",
        "exp_date": "expiration",
        "daysToExpiration": "dte",
        "days_to_expiration": "dte",
        "openInterest": "oi",
        "open_interest": "oi",
        "vol": "volume",
        "mark_iv": "iv",
        "impliedVolatility": "iv",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = {
        "symbol": symbol or "",
        "expiration": "",
        "dte": np.nan,
        "option_type": "",
        "strike": np.nan,
        "bid": np.nan,
        "ask": np.nan,
        "mid": np.nan,
        "last": np.nan,
        "delta": np.nan,
        "iv": np.nan,
        "oi": 0,
        "volume": 0,
        "timestamp": None,
        "optionSymbol": None,
        "underlyingPrice": np.nan,
    }

    for col, default in needed.items():
        if col not in df.columns:
            df[col] = default

    if symbol is not None:
        df["symbol"] = symbol.upper()

    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"p": "put", "c": "call"})
    )

    num_cols = ["dte", "strike", "bid", "ask", "mid", "last", "delta", "iv", "oi", "volume", "underlyingPrice"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["expiration_dt"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["dte"] = np.where(df["dte"].notna(), df["dte"], df["expiration"].apply(calc_dte))
    df["mid"] = np.where(
        df["mid"].notna(),
        df["mid"],
        df.apply(lambda r: option_mid(r["bid"], r["ask"], r["last"]), axis=1),
    )
    df["spread_pct"] = df.apply(lambda r: pct_bid_ask_spread(r["bid"], r["ask"]), axis=1)
    df["delta_abs"] = df["delta"].abs()

    return df.sort_values(["symbol", "expiration_dt", "option_type", "strike"]).reset_index(drop=True)


def liquidity_ok(row, max_bid_ask_pct=DEFAULT_MAX_BID_ASK_PCT, min_oi=DEFAULT_MIN_OI, min_volume=DEFAULT_MIN_VOL):
    oi_ok = safe_float(row.get("oi"), 0) >= min_oi
    vol_ok = safe_float(row.get("volume"), 0) >= min_volume
    ba = row.get("spread_pct", np.nan)
    ba_ok = np.isnan(ba) or ba <= max_bid_ask_pct
    mid_ok = np.isfinite(row.get("mid", np.nan)) and row.get("mid", np.nan) > 0
    return oi_ok and vol_ok and ba_ok and mid_ok


# =========================================================
# PHASE 1 - TREND + ZONES
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
# PHASE 2 - SPREAD SUGGESTIONS
# =========================================================
def trend_allowed_strategies(trend: str) -> list[str]:
    if trend == "Uptrend":
        return ["bull_put"]
    if trend == "Downtrend":
        return ["bear_call"]
    return []


def build_zone_spreads_for_expiration(
    df_exp: pd.DataFrame,
    symbol: str,
    option_type: str,
    strategy: str,
    max_width: float,
    max_risk_dollars: float,
    require_liquidity: bool,
    zones: dict,
    atr_buffer_mult: float,
):
    rows = []

    sub = df_exp[
        (df_exp["symbol"] == symbol) &
        (df_exp["option_type"] == option_type)
    ].copy()

    if require_liquidity:
        sub = sub[sub.apply(liquidity_ok, axis=1)].copy()

    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("strike").reset_index(drop=True)

    atr14 = safe_float(zones.get("atr14"), np.nan)
    demand_low = safe_float(zones.get("demand_low"), np.nan)
    supply_high = safe_float(zones.get("supply_high"), np.nan)

    for _, short_row in sub.iterrows():
        short_delta_abs = safe_float(short_row["delta_abs"], np.nan)
        short_strike = safe_float(short_row["strike"], np.nan)
        short_mid = safe_float(short_row["mid"], np.nan)

        if not np.isfinite(short_strike) or not np.isfinite(short_mid):
            continue

        zone_ok = False
        zone_distance = np.nan
        zone_label = "Unknown"

        if strategy == "bull_put":
            if not (np.isfinite(demand_low) and np.isfinite(atr14)):
                continue

            safe_threshold = demand_low - (atr14 * atr_buffer_mult)
            zone_distance = demand_low - short_strike
            zone_ok = short_strike < safe_threshold
            zone_label = "Ideal" if zone_ok else "Near Zone"
            candidates = sub[sub["strike"] < short_strike].copy()

        elif strategy == "bear_call":
            if not (np.isfinite(supply_high) and np.isfinite(atr14)):
                continue

            safe_threshold = supply_high + (atr14 * atr_buffer_mult)
            zone_distance = short_strike - supply_high
            zone_ok = short_strike > safe_threshold
            zone_label = "Ideal" if zone_ok else "Near Zone"
            candidates = sub[sub["strike"] > short_strike].copy()

        else:
            continue

        if candidates.empty:
            continue

        for _, long_row in candidates.iterrows():
            long_strike = safe_float(long_row["strike"], np.nan)
            long_mid = safe_float(long_row["mid"], np.nan)

            if not np.isfinite(long_strike) or not np.isfinite(long_mid):
                continue

            width = abs(short_strike - long_strike)
            if width <= 0 or width > max_width:
                continue

            credit = short_mid - long_mid
            if not np.isfinite(credit) or credit <= 0:
                continue
            if credit >= width:
                continue

            credit_dollars = credit * 100.0
            max_risk = (width - credit) * 100.0
            if max_risk <= 0 or max_risk > max_risk_dollars:
                continue

            pop = approx_pop_from_short_delta(short_delta_abs) if np.isfinite(short_delta_abs) else np.nan
            ev = expected_value(credit_dollars, max_risk, pop) if np.isfinite(pop) else np.nan
            score = spread_score(credit_dollars, max_risk, pop) if np.isfinite(pop) else np.nan

            take_profit_dollars = credit_dollars * DEFAULT_TAKE_PROFIT_PCT
            buyback_target_dollars = credit_dollars * (1.0 - DEFAULT_TAKE_PROFIT_PCT)

            rows.append(
                {
                    "symbol": symbol,
                    "expiration": short_row["expiration"],
                    "dte": short_row["dte"],
                    "strategy": strategy,
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "width": width,
                    "short_delta": short_row["delta"],
                    "credit": round(credit_dollars, 2),
                    "max_risk": round(max_risk, 2),
                    "pop": round(pop, 4) if np.isfinite(pop) else np.nan,
                    "ev": round(ev, 2) if np.isfinite(ev) else np.nan,
                    "score": round(score, 4) if np.isfinite(score) else np.nan,
                    "roi_on_risk": round(credit_dollars / max_risk, 4),
                    "zone_distance": round(zone_distance, 2) if np.isfinite(zone_distance) else np.nan,
                    "zone_ok": zone_ok,
                    "zone_label": zone_label,
                    "take_profit_dollars": round(take_profit_dollars, 2),
                    "buyback_target_dollars": round(buyback_target_dollars, 2),
                    "short_mid": round(short_mid, 2),
                    "long_mid": round(long_mid, 2),
                }
            )

    return pd.DataFrame(rows)


def scan_zone_spreads(
    chain: pd.DataFrame,
    account_size: float,
    max_risk_pct: float,
    min_dte: int,
    max_dte: int,
    max_width: float,
    require_liquidity: bool,
    trend: str,
    zones: dict,
    atr_buffer_mult: float,
):
    allowed = trend_allowed_strategies(trend)
    if not allowed:
        return pd.DataFrame()

    df = chain.copy()
    max_risk_dollars = account_size * max_risk_pct
    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    out_parts = []

    for symbol in symbols:
        sym_df = df[
            (df["symbol"] == symbol) &
            (df["dte"] >= min_dte) &
            (df["dte"] <= max_dte)
        ].copy()

        if sym_df.empty:
            continue

        expirations = sym_df["expiration"].dropna().unique().tolist()

        for exp in expirations:
            df_exp = sym_df[sym_df["expiration"] == exp].copy()

            if "bull_put" in allowed:
                puts = build_zone_spreads_for_expiration(
                    df_exp=df_exp,
                    symbol=symbol,
                    option_type="put",
                    strategy="bull_put",
                    max_width=max_width,
                    max_risk_dollars=max_risk_dollars,
                    require_liquidity=require_liquidity,
                    zones=zones,
                    atr_buffer_mult=atr_buffer_mult,
                )
                if not puts.empty:
                    out_parts.append(puts)

            if "bear_call" in allowed:
                calls = build_zone_spreads_for_expiration(
                    df_exp=df_exp,
                    symbol=symbol,
                    option_type="call",
                    strategy="bear_call",
                    max_width=max_width,
                    max_risk_dollars=max_risk_dollars,
                    require_liquidity=require_liquidity,
                    zones=zones,
                    atr_buffer_mult=atr_buffer_mult,
                )
                if not calls.empty:
                    out_parts.append(calls)

    if not out_parts:
        return pd.DataFrame()

    out = pd.concat(out_parts, ignore_index=True)
    out["spread_name"] = np.where(
        out["strategy"] == "bull_put",
        out["short_strike"].astype(str) + "/" + out["long_strike"].astype(str) + " Bull Put",
        out["short_strike"].astype(str) + "/" + out["long_strike"].astype(str) + " Bear Call",
    )
    out["pop_pct"] = (out["pop"] * 100.0).round(1)
    out["roi_pct"] = (out["roi_on_risk"] * 100.0).round(1)

    return out.sort_values(
        ["zone_ok", "score", "zone_distance", "ev", "credit"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


# =========================================================
# PHASE 3 - SCORING
# =========================================================
def rate_trade(row, trend: str, zones: dict) -> dict:
    total = 0

    if trend == "Uptrend" and row["strategy"] == "bull_put":
        trend_score = 30
    elif trend == "Downtrend" and row["strategy"] == "bear_call":
        trend_score = 30
    else:
        trend_score = 0
    total += trend_score

    zone_distance = safe_float(row.get("zone_distance"), np.nan)
    atr14 = safe_float(zones.get("atr14"), np.nan)
    zone_ok = bool(row.get("zone_ok", False))

    if zone_ok and np.isfinite(zone_distance) and np.isfinite(atr14):
        ratio = zone_distance / atr14 if atr14 > 0 else 0
        if ratio >= 1.0:
            zone_score = 30
        elif ratio >= 0.5:
            zone_score = 24
        else:
            zone_score = 18
    elif np.isfinite(zone_distance) and np.isfinite(atr14):
        ratio = zone_distance / atr14 if atr14 > 0 else 0
        if ratio >= 0.10:
            zone_score = 10
        else:
            zone_score = 3
    else:
        zone_score = 0
    total += zone_score

    credit = safe_float(row.get("credit"), np.nan)
    max_risk = safe_float(row.get("max_risk"), np.nan)
    roi = credit / max_risk if np.isfinite(credit) and np.isfinite(max_risk) and max_risk > 0 else 0
    if roi >= 0.18:
        rr_score = 15
    elif roi >= 0.12:
        rr_score = 10
    elif roi >= 0.08:
        rr_score = 6
    else:
        rr_score = 2
    total += rr_score

    delta_abs = abs(safe_float(row.get("short_delta"), np.nan))
    if np.isfinite(delta_abs):
        if 0.10 <= delta_abs <= 0.22:
            delta_score = 10
        elif 0.23 <= delta_abs <= 0.30:
            delta_score = 7
        elif delta_abs < 0.10:
            delta_score = 5
        else:
            delta_score = 3
    else:
        delta_score = 0
    total += delta_score

    ev = safe_float(row.get("ev"), np.nan)
    if np.isfinite(ev):
        if ev >= 8:
            ev_score = 15
        elif ev >= 4:
            ev_score = 10
        elif ev >= 0:
            ev_score = 5
        else:
            ev_score = 1
    else:
        ev_score = 0
    total += ev_score

    if total >= 80:
        grade = "A Setup"
    elif total >= 60:
        grade = "B Setup"
    else:
        grade = "Avoid"

    return {
        "trade_score": total,
        "grade": grade,
        "trend_score": trend_score,
        "zone_score": zone_score,
        "rr_score": rr_score,
        "delta_score": delta_score,
        "ev_score": ev_score,
    }


# =========================================================
# CACHE LOADER
# =========================================================
@st.cache_data(ttl=300)
def load_symbol_data(symbol: str, api_key: str, min_dte: int, max_dte: int, strike_limit: int):
    candles_raw = fetch_stock_candles(symbol, api_key, countback=120)
    candles = add_indicators(candles_raw)
    trend_info = classify_trend(candles)
    zones = detect_supply_demand_zones(candles)

    chain_raw = fetch_option_chain(
        symbol,
        api_key,
        min_dte=min_dte,
        max_dte=max_dte,
        strike_limit=strike_limit,
    )
    chain = normalize_chain(chain_raw, symbol=symbol)

    return candles, trend_info, zones, chain


# =========================================================
# SETTINGS PANEL
# =========================================================
saved_api_key = st.secrets.get("MARKETDATA_API_KEY", "")

with st.expander("Scanner Settings", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        st.text_input(
            "API Key",
            type="password",
            value="",
            placeholder="Loaded from Streamlit secrets" if saved_api_key else "Add MARKETDATA_API_KEY to secrets",
            disabled=True if saved_api_key else False,
        )
        symbol_input = st.text_input("Symbols (comma separated)", value="SPY,QQQ,IWM,AMD,AAPL")
        account_size = st.number_input("Account Size", min_value=100.0, value=DEFAULT_ACCOUNT_SIZE, step=50.0)
        max_risk_pct = st.slider("Max Risk Per Trade (% of acct)", 0.05, 0.40, DEFAULT_MAX_RISK_PCT, 0.01)
        require_liquidity = st.checkbox("Require Liquidity Filter", value=False)

    with c2:
        min_dte = st.number_input("Min DTE", min_value=1, value=DEFAULT_MIN_DTE, step=1)
        max_dte = st.number_input("Max DTE", min_value=1, value=DEFAULT_MAX_DTE, step=1)
        max_width = st.selectbox("Max Spread Width", options=[1.0, 2.0, 3.0], index=0)
        strike_limit = st.selectbox("Strike Limit Near Money", options=[8, 12, 16, 20, 30, 40], index=4)
        atr_zone_buffer = st.slider("ATR Buffer Outside Zone", 0.10, 1.00, DEFAULT_ATR_ZONE_BUFFER, 0.05)

api_key = saved_api_key
run_scan = st.button("Load Data / Scan", use_container_width=True)


# =========================================================
# MAIN
# =========================================================
tabs = st.tabs(["Dashboard", "Spread Suggestions", "Growth", "Debug"])
symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

if not run_scan:
    with tabs[0]:
        st.info("Enter your symbols and click 'Load Data / Scan'.")
else:
    try:
        if not api_key:
            st.warning("Add MARKETDATA_API_KEY to Streamlit secrets.")
        else:
            all_candles = {}
            all_trends = {}
            all_zones = {}
            all_chains = {}

            for sym in symbols:
                candles, trend_info, zones, chain = load_symbol_data(
                    sym,
                    api_key,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    strike_limit=strike_limit,
                )
                all_candles[sym] = candles
                all_trends[sym] = trend_info
                all_zones[sym] = zones
                all_chains[sym] = chain

            # DASHBOARD
            with tabs[0]:
                st.subheader("Phase 1 — Trend + Supply/Demand Dashboard")

                dashboard_rows = []
                for sym in symbols:
                    trend_info = all_trends.get(sym, {})
                    zones = all_zones.get(sym, {})
                    price = safe_float(trend_info.get("last_close"), np.nan)

                    dashboard_rows.append(
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

                dashboard_df = pd.DataFrame(dashboard_rows)
                st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

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
                        st.markdown("**Demand Zone**")
                        st.write(f"{zones.get('demand_low')} to {zones.get('demand_high')}")
                    with z2:
                        st.markdown("**Supply Zone**")
                        st.write(f"{zones.get('supply_low')} to {zones.get('supply_high')}")

            # SPREAD SUGGESTIONS
            with tabs[1]:
                st.subheader("Phase 2 + 3 — Trend-Aligned Spread Suggestions")

                spread_symbol = st.selectbox("Ticker for Spread Suggestions", options=symbols, index=0, key="spread_symbol")
                trend_info = all_trends.get(spread_symbol, {})
                zones = all_zones.get(spread_symbol, {})
                chain = all_chains.get(spread_symbol, pd.DataFrame())

                trend = trend_info.get("trend", "Unknown")
                strength = trend_info.get("strength", "Unknown")
                price = safe_float(trend_info.get("last_close"), np.nan)

                st.write(f"**Trend:** {trend} ({strength})")
                st.write(f"**Current Price:** {price}")
                st.write(f"**Demand Zone:** {zones.get('demand_low')} to {zones.get('demand_high')}")
                st.write(f"**Supply Zone:** {zones.get('supply_low')} to {zones.get('supply_high')}")

                if chain.empty:
                    st.info("No option chain returned.")
                else:
                    spread_results = scan_zone_spreads(
                        chain=chain,
                        account_size=account_size,
                        max_risk_pct=max_risk_pct,
                        min_dte=min_dte,
                        max_dte=max_dte,
                        max_width=max_width,
                        require_liquidity=require_liquidity,
                        trend=trend,
                        zones=zones,
                        atr_buffer_mult=atr_zone_buffer,
                    )

                    if spread_results.empty:
                        st.info("No qualifying trend-aligned spreads outside the zones.")
                    else:
                        scores = spread_results.apply(lambda r: pd.Series(rate_trade(r, trend, zones)), axis=1)
                        spread_results = pd.concat([spread_results, scores], axis=1)

                        diagnostic = {}
                        if trend == "Downtrend":
                            diagnostic["needed_short_call_above"] = (
                                round(safe_float(zones.get("supply_high"), np.nan) + safe_float(zones.get("atr14"), 0) * atr_zone_buffer, 2)
                                if np.isfinite(safe_float(zones.get("supply_high"), np.nan)) and np.isfinite(safe_float(zones.get("atr14"), np.nan))
                                else np.nan
                            )
                            if not chain[chain["option_type"] == "call"].empty:
                                diagnostic["highest_call_strike_returned"] = safe_float(
                                    chain[chain["option_type"] == "call"]["strike"].max(), np.nan
                                )
                        elif trend == "Uptrend":
                            diagnostic["needed_short_put_below"] = (
                                round(safe_float(zones.get("demand_low"), np.nan) - safe_float(zones.get("atr14"), 0) * atr_zone_buffer, 2)
                                if np.isfinite(safe_float(zones.get("demand_low"), np.nan)) and np.isfinite(safe_float(zones.get("atr14"), np.nan))
                                else np.nan
                            )
                            if not chain[chain["option_type"] == "put"].empty:
                                diagnostic["lowest_put_strike_returned"] = safe_float(
                                    chain[chain["option_type"] == "put"]["strike"].min(), np.nan
                                )

                        st.write("Spread diagnostics:", diagnostic)

                        show_cols = [
                            "symbol",
                            "expiration",
                            "dte",
                            "spread_name",
                            "zone_label",
                            "short_delta",
                            "credit",
                            "max_risk",
                            "zone_distance",
                            "pop_pct",
                            "roi_pct",
                            "ev",
                            "trade_score",
                            "grade",
                            "take_profit_dollars",
                        ]
                        st.dataframe(spread_results[show_cols], use_container_width=True, hide_index=True)

                        top = spread_results.iloc[0]
                        qty = contracts_allowed(
                            max_risk_per_trade=top["max_risk"],
                            account_size=account_size,
                            total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT,
                        )

                        st.markdown("### Best Suggestion")
                        st.write(
                            f"""
**Ticker:** {top['symbol']}  
**Trend:** {trend}  
**Suggested Spread:** {top['spread_name']}  
**Zone Status:** {top['zone_label']}  
**Credit:** ${top['credit']:.2f}  
**Max Risk:** ${top['max_risk']:.2f}  
**Short Delta:** {top['short_delta']:.3f}  
**Zone Distance:** {top['zone_distance']:.2f}  
**POP:** {top['pop_pct']:.1f}%  
**Trade Score:** {top['trade_score']} / 100  
**Grade:** {top['grade']}  
**Take Profit:** ${top['take_profit_dollars']:.2f}  
**Buyback Target:** ${top['buyback_target_dollars']:.2f}  
**Contracts Allowed:** {qty}
"""
                        )

                        st.markdown("### Top 5 Cards")
                        for _, row in spread_results.head(5).iterrows():
                            st.markdown(
                                f"""
**{row['spread_name']}**  
Zone Status: **{row['zone_label']}**  
DTE: **{int(row['dte'])}** | Short Delta: **{row['short_delta']:.3f}**  
Credit: **${row['credit']:.2f}** | Max Risk: **${row['max_risk']:.2f}**  
Zone Distance: **{row['zone_distance']:.2f}** | POP: **{row['pop_pct']:.1f}%**  
Score: **{row['trade_score']}** | Grade: **{row['grade']}**
"""
                            )
                            st.divider()

            # GROWTH
            with tabs[2]:
                st.subheader("Growth Projection")

                growth_symbol = st.selectbox("Growth Model Ticker", options=symbols, index=0, key="growth_symbol")
                assumed_win_rate = st.slider("Assumed Win Rate", 0.40, 0.95, 0.75, 0.01)
                growth_trades = st.number_input("Number of Trades", min_value=1, value=25, step=1)

                trend_info = all_trends.get(growth_symbol, {})
                zones = all_zones.get(growth_symbol, {})
                chain = all_chains.get(growth_symbol, pd.DataFrame())
                trend = trend_info.get("trend", "Unknown")

                growth_spreads = scan_zone_spreads(
                    chain=chain,
                    account_size=account_size,
                    max_risk_pct=max_risk_pct,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    max_width=max_width,
                    require_liquidity=require_liquidity,
                    trend=trend,
                    zones=zones,
                    atr_buffer_mult=atr_zone_buffer,
                )

                if growth_spreads.empty:
                    st.info("No spreads available for growth projection.")
                else:
                    scores = growth_spreads.apply(lambda r: pd.Series(rate_trade(r, trend, zones)), axis=1)
                    growth_spreads = pd.concat([growth_spreads, scores], axis=1)

                    model_trade = growth_spreads.iloc[0]
                    fixed_df = project_growth_fixed(
                        start_balance=account_size,
                        avg_win=model_trade["credit"],
                        avg_loss=model_trade["max_risk"],
                        win_rate=assumed_win_rate,
                        num_trades=int(growth_trades),
                    )

                    end_balance = fixed_df.iloc[-1]["expected_balance"] if not fixed_df.empty else account_size
                    ev_per_trade = (assumed_win_rate * model_trade["credit"]) - ((1 - assumed_win_rate) * model_trade["max_risk"])

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Starting Balance", f"${account_size:,.2f}")
                    with c2:
                        st.metric("Expected P/L Per Trade", f"${ev_per_trade:,.2f}")
                    with c3:
                        st.metric("Projected End Balance", f"${end_balance:,.2f}")

                    st.markdown("### Model Trade")
                    st.write(
                        f"""
**{model_trade['symbol']} — {model_trade['spread_name']}**  
Trend: **{trend}**  
Zone Status: **{model_trade['zone_label']}**  
Credit: **${model_trade['credit']:.2f}**  
Max Risk: **${model_trade['max_risk']:.2f}**  
Trade Score: **{model_trade['trade_score']}**
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

                chain = all_chains.get(debug_symbol, pd.DataFrame())
                if chain.empty:
                    st.info("No chain data.")
                else:
                    debug_cols = [
                        c for c in [
                            "symbol", "expiration", "dte", "option_type", "strike",
                            "bid", "ask", "mid", "last", "delta", "delta_abs",
                            "iv", "oi", "volume", "timestamp", "underlyingPrice"
                        ] if c in chain.columns
                    ]
                    st.dataframe(chain[debug_cols], use_container_width=True, hide_index=True)

                    st.write("Chain row count:", len(chain))
                    st.write("Expirations:", sorted(chain["expiration"].dropna().astype(str).unique().tolist()))
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