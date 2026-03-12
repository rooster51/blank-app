import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Options Scanner", layout="wide")
st.title("Options Scanner - Single Legs + Credit Spreads")
st.caption("Mobile-friendly layout with settings on the page instead of the sidebar.")


# =========================================================
# DEFAULTS
# =========================================================
DEFAULT_ACCOUNT_SIZE = 500.0
DEFAULT_MAX_RISK_PCT = 0.25
DEFAULT_TOTAL_RISK_CAP_PCT = 0.50
DEFAULT_MIN_DTE = 20
DEFAULT_MAX_DTE = 45
DEFAULT_SHORT_DELTA_MIN = 0.20
DEFAULT_SHORT_DELTA_MAX = 0.30
DEFAULT_SINGLE_DELTA_MIN = 0.20
DEFAULT_SINGLE_DELTA_MAX = 0.40
DEFAULT_MIN_CREDIT_PCT_WIDTH = 0.25
DEFAULT_MAX_CREDIT_PCT_WIDTH = 0.40
DEFAULT_MAX_BID_ASK_PCT = 0.20
DEFAULT_TAKE_PROFIT_PCT = 0.50
DEFAULT_MIN_OI = 25
DEFAULT_MIN_VOL = 1
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


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def approx_prob_itm_call(spot, strike, iv, dte):
    spot = safe_float(spot, np.nan)
    strike = safe_float(strike, np.nan)
    iv = safe_float(iv, np.nan)
    dte = safe_float(dte, np.nan)

    if not all(np.isfinite(v) for v in [spot, strike, iv, dte]):
        return np.nan
    if spot <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return np.nan

    t = dte / 365.0
    sigma_t = iv * math.sqrt(t)
    if sigma_t <= 0:
        return np.nan

    d2 = (math.log(spot / strike) - 0.5 * iv * iv * t) / sigma_t
    return norm_cdf(d2)


def approx_prob_itm_put(spot, strike, iv, dte):
    p_call_itm = approx_prob_itm_call(spot, strike, iv, dte)
    if not np.isfinite(p_call_itm):
        return np.nan
    return 1.0 - p_call_itm


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


def fair_credit_heuristic(width_points, pop):
    if not all(np.isfinite(v) for v in [width_points, pop]):
        return np.nan
    loss_prob = 1.0 - pop
    return width_points * loss_prob


def contracts_allowed(max_risk_per_trade, account_size, total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT):
    max_risk_per_trade = safe_float(max_risk_per_trade, np.nan)
    account_size = safe_float(account_size, np.nan)
    if not np.isfinite(max_risk_per_trade) or not np.isfinite(account_size) or max_risk_per_trade <= 0:
        return 0
    cap = account_size * total_risk_cap_pct
    return max(0, int(cap // max_risk_per_trade))


def first_valid_value(payload, keys):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    return item
        elif value is not None:
            return value
    return None


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


def flatten_option_symbols(frame: pd.DataFrame) -> list[str]:
    for col in ["optionSymbol", "option_symbol", "symbol", "option"]:
        if col in frame.columns:
            vals = frame[col].dropna().astype(str).tolist()
            vals = [v for v in vals if v.strip()]
            if vals:
                return vals
    return []


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# =========================================================
# API - MARKETDATA.APP
# =========================================================
def fetch_underlying_quote(symbol: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.marketdata.app/v1/stocks/quotes/{symbol.upper()}/"
    params = {"dateformat": "timestamp"}

    resp = requests.get(url, headers=headers, params=params, timeout=DEFAULT_API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    spot = safe_float(first_valid_value(data, ["mid", "last", "price", "close"]), np.nan)
    timestamp = first_valid_value(data, ["updated", "timestamp", "s"])

    return {
        "symbol": symbol.upper(),
        "spot": spot,
        "timestamp": timestamp,
        "raw": data,
    }


def fetch_option_chain(symbol: str, api_key: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}

    # Step 1: chain endpoint for contract discovery
    chain_url = f"https://api.marketdata.app/v1/options/chain/{symbol.upper()}/"
    chain_params = {"dateformat": "timestamp"}

    chain_resp = requests.get(chain_url, headers=headers, params=chain_params, timeout=DEFAULT_API_TIMEOUT)
    chain_resp.raise_for_status()
    chain_data = chain_resp.json()

    chain_df = payload_to_frame(chain_data)
    if chain_df.empty:
        return pd.DataFrame()

    option_symbols = flatten_option_symbols(chain_df)
    option_symbols = dedupe_keep_order(option_symbols)

    if not option_symbols:
        return pd.DataFrame()

    # Step 2: quotes endpoint for real quote fields
    quote_frames = []
    chunk_size = 50

    for i in range(0, len(option_symbols), chunk_size):
        chunk = option_symbols[i:i + chunk_size]
        joined = ",".join(chunk)

        quotes_url = f"https://api.marketdata.app/v1/options/quotes/{joined}/"
        quote_params = {"dateformat": "timestamp"}

        q_resp = requests.get(quotes_url, headers=headers, params=quote_params, timeout=DEFAULT_API_TIMEOUT)
        q_resp.raise_for_status()
        q_data = q_resp.json()

        q_df = payload_to_frame(q_data)
        if not q_df.empty:
            quote_frames.append(q_df)

    if not quote_frames:
        return pd.DataFrame()

    quotes_df = pd.concat(quote_frames, ignore_index=True)

    rename_map = {
        "underlying": "symbol",
        "ticker": "symbol",
        "expiration": "expiration",
        "expirationDate": "expiration",
        "side": "option_type",
        "type": "option_type",
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "last": "last",
        "lastPrice": "last",
        "delta": "delta",
        "iv": "iv",
        "openInterest": "oi",
        "volume": "volume",
        "updated": "timestamp",
        "optionSymbol": "optionSymbol",
    }
    quotes_df = quotes_df.rename(columns={k: v for k, v in rename_map.items() if k in quotes_df.columns})

    if "symbol" not in quotes_df.columns:
        quotes_df["symbol"] = symbol.upper()

    needed_cols = [
        "symbol",
        "expiration",
        "option_type",
        "strike",
        "bid",
        "ask",
        "last",
        "delta",
        "iv",
        "oi",
        "volume",
        "timestamp",
        "optionSymbol",
    ]
    for col in needed_cols:
        if col not in quotes_df.columns:
            quotes_df[col] = np.nan

    quotes_df["option_type"] = (
        quotes_df["option_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"p": "put", "c": "call"})
    )

    return quotes_df[needed_cols].copy()


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
        "last": np.nan,
        "delta": np.nan,
        "iv": np.nan,
        "oi": 0,
        "volume": 0,
        "timestamp": None,
        "optionSymbol": None,
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

    num_cols = ["dte", "strike", "bid", "ask", "last", "delta", "iv", "oi", "volume"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["expiration_dt"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["dte"] = np.where(df["dte"].notna(), df["dte"], df["expiration"].apply(calc_dte))
    df["mid"] = df.apply(lambda r: option_mid(r["bid"], r["ask"], r["last"]), axis=1)
    df["spread_pct"] = df.apply(lambda r: pct_bid_ask_spread(r["bid"], r["ask"]), axis=1)
    df["delta_abs"] = df["delta"].abs()

    df = df.sort_values(["symbol", "expiration_dt", "option_type", "strike"]).reset_index(drop=True)
    return df


def liquidity_ok(row, max_bid_ask_pct=DEFAULT_MAX_BID_ASK_PCT, min_oi=DEFAULT_MIN_OI, min_volume=DEFAULT_MIN_VOL):
    oi_ok = safe_float(row.get("oi"), 0) >= min_oi
    vol_ok = safe_float(row.get("volume"), 0) >= min_volume
    ba = row.get("spread_pct", np.nan)
    ba_ok = np.isnan(ba) or ba <= max_bid_ask_pct
    mid_ok = np.isfinite(row.get("mid", np.nan)) and row.get("mid", np.nan) > 0
    return oi_ok and vol_ok and ba_ok and mid_ok


# =========================================================
# SINGLE LEG SCANNER
# =========================================================
def scan_single_legs(
    chain: pd.DataFrame,
    spot: float,
    direction: str,
    min_dte: int,
    max_dte: int,
    delta_min: float = DEFAULT_SINGLE_DELTA_MIN,
    delta_max: float = DEFAULT_SINGLE_DELTA_MAX,
    require_liquidity: bool = True,
):
    df = chain.copy()

    if direction == "bullish":
        option_type = "call"
    elif direction == "bearish":
        option_type = "put"
    else:
        return pd.DataFrame()

    df = df[
        (df["option_type"] == option_type) &
        (df["dte"] >= min_dte) &
        (df["dte"] <= max_dte)
    ].copy()

    if require_liquidity:
        df = df[df.apply(liquidity_ok, axis=1)].copy()

    if df.empty:
        return pd.DataFrame()

    df = df[(df["delta_abs"] >= delta_min) & (df["delta_abs"] <= delta_max)].copy()

    if df.empty:
        return pd.DataFrame()

    def compute_intrinsic(row):
        if row["option_type"] == "call":
            return max(0.0, spot - row["strike"])
        return max(0.0, row["strike"] - spot)

    def compute_extrinsic(row):
        intrinsic = compute_intrinsic(row)
        premium = row["mid"] * 100.0
        return max(0.0, premium - intrinsic * 100.0)

    def compute_pop(row):
        if row["option_type"] == "call":
            return approx_prob_itm_call(spot, row["strike"], row["iv"], row["dte"])
        return approx_prob_itm_put(spot, row["strike"], row["iv"], row["dte"])

    df["premium"] = (df["mid"] * 100.0).round(2)
    df["intrinsic"] = df.apply(compute_intrinsic, axis=1).round(4)
    df["extrinsic_dollars"] = df.apply(compute_extrinsic, axis=1).round(2)
    df["approx_itm_prob"] = df.apply(compute_pop, axis=1)
    df["score"] = (df["delta_abs"] / df["premium"].replace(0, np.nan)) * 1000.0
    df["moneyness"] = np.where(
        df["option_type"] == "call",
        (df["strike"] - spot),
        (spot - df["strike"])
    )

    cols = [
        "symbol", "expiration", "dte", "option_type", "strike",
        "bid", "ask", "last", "mid", "delta", "delta_abs", "iv", "oi", "volume",
        "premium", "extrinsic_dollars", "approx_itm_prob", "moneyness", "score"
    ]
    df = df[cols].sort_values(
        ["score", "approx_itm_prob", "volume", "oi"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return df


# =========================================================
# CREDIT SPREAD SCANNER
# =========================================================
def build_vertical_spreads_for_expiration(
    df_exp: pd.DataFrame,
    symbol: str,
    option_type: str,
    strategy: str,
    max_width: float,
    max_risk_dollars: float,
    short_delta_min: float,
    short_delta_max: float,
    min_credit_pct_width: float,
    max_credit_pct_width: float,
    require_liquidity: bool,
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

    for _, short_row in sub.iterrows():
        short_delta_abs = safe_float(short_row["delta_abs"], np.nan)
        if not np.isfinite(short_delta_abs):
            continue
        if not (short_delta_min <= short_delta_abs <= short_delta_max):
            continue

        short_strike = safe_float(short_row["strike"], np.nan)
        short_mid = safe_float(short_row["mid"], np.nan)
        if not np.isfinite(short_strike) or not np.isfinite(short_mid):
            continue

        if strategy == "bull_put":
            candidates = sub[sub["strike"] < short_strike].copy()
        elif strategy == "bear_call":
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

            credit_pct_width = credit / width
            if not (min_credit_pct_width <= credit_pct_width <= max_credit_pct_width):
                continue

            pop = approx_pop_from_short_delta(short_delta_abs)
            ev = expected_value(credit_dollars, max_risk, pop)
            score = spread_score(credit_dollars, max_risk, pop)

            fair_credit = fair_credit_heuristic(width, pop) * 100.0
            edge_dollars = credit_dollars - fair_credit
            lottery_flag = edge_dollars >= 7.5

            take_profit_dollars = credit_dollars * DEFAULT_TAKE_PROFIT_PCT
            buyback_target_dollars = credit_dollars * (1.0 - DEFAULT_TAKE_PROFIT_PCT)

            rows.append({
                "symbol": symbol,
                "expiration": short_row["expiration"],
                "dte": short_row["dte"],
                "strategy": strategy,
                "option_type": option_type,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "short_delta": short_row["delta"],
                "short_delta_abs": short_delta_abs,
                "short_bid": short_row["bid"],
                "short_ask": short_row["ask"],
                "short_mid": short_mid,
                "long_bid": long_row["bid"],
                "long_ask": long_row["ask"],
                "long_mid": long_mid,
                "credit": round(credit_dollars, 2),
                "max_risk": round(max_risk, 2),
                "credit_pct_width": round(credit_pct_width, 4),
                "pop": round(pop, 4),
                "ev": round(ev, 2),
                "score": round(score, 4),
                "fair_credit_heuristic": round(fair_credit, 2),
                "edge_dollars": round(edge_dollars, 2),
                "lottery_flag": lottery_flag,
                "take_profit_dollars": round(take_profit_dollars, 2),
                "buyback_target_dollars": round(buyback_target_dollars, 2),
                "roi_on_risk": round(credit_dollars / max_risk, 4),
            })

    return pd.DataFrame(rows)


def scan_credit_spreads(
    chain: pd.DataFrame,
    account_size: float,
    max_risk_pct: float,
    min_dte: int,
    max_dte: int,
    max_width: float,
    short_delta_min: float,
    short_delta_max: float,
    min_credit_pct_width: float,
    max_credit_pct_width: float,
    require_liquidity: bool,
):
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

            puts = build_vertical_spreads_for_expiration(
                df_exp=df_exp,
                symbol=symbol,
                option_type="put",
                strategy="bull_put",
                max_width=max_width,
                max_risk_dollars=max_risk_dollars,
                short_delta_min=short_delta_min,
                short_delta_max=short_delta_max,
                min_credit_pct_width=min_credit_pct_width,
                max_credit_pct_width=max_credit_pct_width,
                require_liquidity=require_liquidity,
            )

            calls = build_vertical_spreads_for_expiration(
                df_exp=df_exp,
                symbol=symbol,
                option_type="call",
                strategy="bear_call",
                max_width=max_width,
                max_risk_dollars=max_risk_dollars,
                short_delta_min=short_delta_min,
                short_delta_max=short_delta_max,
                min_credit_pct_width=min_credit_pct_width,
                max_credit_pct_width=max_credit_pct_width,
                require_liquidity=require_liquidity,
            )

            if not puts.empty:
                out_parts.append(puts)
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

    out = out.sort_values(
        ["score", "ev", "edge_dollars", "pop"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return out


# =========================================================
# DEBUG / VALIDATION
# =========================================================
def build_debug_view(chain: pd.DataFrame):
    cols = [
        c for c in [
            "symbol", "expiration", "dte", "option_type", "strike",
            "bid", "ask", "last", "mid", "spread_pct", "delta", "delta_abs",
            "iv", "oi", "volume", "timestamp", "optionSymbol"
        ] if c in chain.columns
    ]
    return chain[cols].copy()


def validate_chain(chain: pd.DataFrame):
    problems = []

    if chain.empty:
        problems.append("Chain is empty.")
        return problems

    if (chain["strike"].isna()).any():
        problems.append("Some strikes are missing or non-numeric.")

    if (chain["mid"].isna()).mean() > 0.25:
        problems.append("A lot of contracts have no usable bid/ask/last price.")

    bad_quotes = chain[
        chain["bid"].notna() &
        chain["ask"].notna() &
        (chain["ask"] < chain["bid"])
    ]
    if not bad_quotes.empty:
        problems.append(f"{len(bad_quotes)} contracts have ask < bid.")

    mixed_types = chain[~chain["option_type"].isin(["put", "call"])]
    if not mixed_types.empty:
        problems.append(f"{len(mixed_types)} rows have invalid option_type values.")

    if (chain["dte"].isna()).mean() > 0.25:
        problems.append("A lot of rows are missing DTE.")

    return problems


# =========================================================
# CACHE
# =========================================================
@st.cache_data(ttl=60)
def load_symbol_data(symbol: str, api_key: str):
    quote = fetch_underlying_quote(symbol, api_key)
    chain_raw = fetch_option_chain(symbol, api_key)
    chain = normalize_chain(chain_raw, symbol=symbol)
    return quote, chain


# =========================================================
# SETTINGS PANEL (NO SIDEBAR)
# =========================================================
with st.expander("Scanner Settings", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        api_key = st.text_input("API Key", type="password")
        symbol_input = st.text_input("Symbols (comma separated)", value="SPY,QQQ,AMD")
        account_size = st.number_input(
            "Account Size",
            min_value=100.0,
            value=DEFAULT_ACCOUNT_SIZE,
            step=50.0,
        )
        max_risk_pct = st.slider(
            "Max Risk Per Trade (% of acct)",
            0.05,
            0.40,
            DEFAULT_MAX_RISK_PCT,
            0.01,
        )
        require_liquidity = st.checkbox("Require Liquidity Filter", value=True)

    with c2:
        min_dte = st.number_input("Min DTE", min_value=1, value=DEFAULT_MIN_DTE, step=1)
        max_dte = st.number_input("Max DTE", min_value=1, value=DEFAULT_MAX_DTE, step=1)
        max_width = st.selectbox("Max Spread Width", options=[1.0, 2.0, 3.0, 5.0], index=1)
        short_delta_min = st.slider("Short Delta Min", 0.05, 0.50, DEFAULT_SHORT_DELTA_MIN, 0.01)
        short_delta_max = st.slider("Short Delta Max", 0.05, 0.50, DEFAULT_SHORT_DELTA_MAX, 0.01)
        single_delta_min = st.slider("Single-Leg Delta Min", 0.05, 0.80, DEFAULT_SINGLE_DELTA_MIN, 0.01)
        single_delta_max = st.slider("Single-Leg Delta Max", 0.05, 0.80, DEFAULT_SINGLE_DELTA_MAX, 0.01)
        min_credit_pct_width = st.slider("Min Credit % Width", 0.05, 0.80, DEFAULT_MIN_CREDIT_PCT_WIDTH, 0.01)
        max_credit_pct_width = st.slider("Max Credit % Width", 0.05, 0.95, DEFAULT_MAX_CREDIT_PCT_WIDTH, 0.01)

run_scan = st.button("Load Data / Scan", use_container_width=True)


# =========================================================
# MAIN
# =========================================================
tabs = st.tabs(["Overview", "Single Legs", "Credit Spreads", "Debug"])
symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

if not run_scan:
    with tabs[0]:
        st.info("Enter your symbols and click 'Load Data / Scan'.")
else:
    try:
        if not api_key:
            st.warning("Add your API key above.")
        else:
            all_quotes = {}
            chain_parts = []

            for sym in symbols:
                quote, chain = load_symbol_data(sym, api_key)
                all_quotes[sym] = quote
                if not chain.empty:
                    chain_parts.append(chain)

            if not chain_parts:
                st.error("No option chain data returned.")
            else:
                full_chain = pd.concat(chain_parts, ignore_index=True)
                validation_issues = validate_chain(full_chain)

                with tabs[0]:
                    st.subheader("Overview")

                    quote_rows = []
                    for sym in symbols:
                        q = all_quotes.get(sym, {})
                        quote_rows.append({
                            "symbol": sym,
                            "spot": q.get("spot"),
                            "quote_timestamp": q.get("timestamp"),
                        })

                    st.dataframe(pd.DataFrame(quote_rows), use_container_width=True, hide_index=True)

                    if validation_issues:
                        st.warning("Data issues found:")
                        for msg in validation_issues:
                            st.write(f"- {msg}")
                    else:
                        st.success("Chain looks structurally okay.")

                    st.write(f"Contracts loaded: **{len(full_chain):,}**")
                    st.write(f"Symbols loaded: **{', '.join(sorted(full_chain['symbol'].unique()))}**")

                with tabs[1]:
                    st.subheader("Single-Leg Scanner")

                    single_direction = st.radio("Direction", options=["bullish", "bearish"], horizontal=True)
                    single_symbol = st.selectbox("Symbol", options=symbols, index=0, key="single_symbol")

                    quote = all_quotes.get(single_symbol, {})
                    spot = safe_float(quote.get("spot"), np.nan)

                    if not np.isfinite(spot):
                        st.error(f"No valid spot price for {single_symbol}.")
                    else:
                        st.write(f"Spot: **{spot:.2f}**")

                        single_df = full_chain[full_chain["symbol"] == single_symbol].copy()
                        single_results = scan_single_legs(
                            chain=single_df,
                            spot=spot,
                            direction=single_direction,
                            min_dte=min_dte,
                            max_dte=max_dte,
                            delta_min=single_delta_min,
                            delta_max=single_delta_max,
                            require_liquidity=require_liquidity,
                        )

                        if single_results.empty:
                            st.info("No qualifying single-leg contracts found.")
                        else:
                            out = single_results.copy()
                            out["approx_itm_prob"] = (out["approx_itm_prob"] * 100.0).round(1)

                            show_cols = [
                                "symbol", "expiration", "dte", "option_type", "strike",
                                "bid", "ask", "last", "mid", "delta", "delta_abs", "iv",
                                "oi", "volume", "premium", "extrinsic_dollars",
                                "approx_itm_prob", "moneyness", "score"
                            ]
                            st.dataframe(out[show_cols], use_container_width=True, hide_index=True)

                            top = out.iloc[0]
                            st.markdown("### Top Single-Leg Candidate")
                            st.write(
                                f"""
**{top['symbol']} {top['expiration']} {top['option_type'].upper()} {top['strike']}**  
Bid/Ask: **{top['bid']} / {top['ask']}**  
Mid: **{top['mid']:.2f}**  
Premium: **${top['premium']:.2f}**  
Delta: **{top['delta']:.3f}**  
Approx ITM Prob: **{top['approx_itm_prob']:.1f}%**  
Score: **{top['score']:.3f}**
"""
                            )

                with tabs[2]:
                    st.subheader("Credit Spread Scanner")

                    spread_results = scan_credit_spreads(
                        chain=full_chain,
                        account_size=account_size,
                        max_risk_pct=max_risk_pct,
                        min_dte=min_dte,
                        max_dte=max_dte,
                        max_width=max_width,
                        short_delta_min=short_delta_min,
                        short_delta_max=short_delta_max,
                        min_credit_pct_width=min_credit_pct_width,
                        max_credit_pct_width=max_credit_pct_width,
                        require_liquidity=require_liquidity,
                    )

                    if spread_results.empty:
                        st.info("No qualifying spreads found.")
                    else:
                        disp = spread_results.copy()
                        disp["lottery_flag"] = disp["lottery_flag"].map({True: "YES", False: ""})

                        show_cols = [
                            "symbol", "expiration", "dte", "spread_name",
                            "credit", "max_risk", "pop_pct", "roi_pct",
                            "ev", "score", "edge_dollars",
                            "take_profit_dollars", "buyback_target_dollars", "lottery_flag"
                        ]
                        st.dataframe(disp[show_cols], use_container_width=True, hide_index=True)

                        top = spread_results.iloc[0]
                        qty = contracts_allowed(
                            max_risk_per_trade=top["max_risk"],
                            account_size=account_size,
                            total_risk_cap_pct=DEFAULT_TOTAL_RISK_CAP_PCT
                        )

                        st.markdown("### Top Spread Candidate")
                        st.write(
                            f"""
**{top['symbol']} - {top['spread_name']}**  
Expiration: **{top['expiration']}**  
DTE: **{int(top['dte'])}**  
Credit: **${top['credit']:.2f}**  
Max Risk: **${top['max_risk']:.2f}**  
POP: **{top['pop_pct']:.1f}%**  
EV: **${top['ev']:.2f}**  
Score: **{top['score']:.4f}**  
Take Profit @ 50%: **${top['take_profit_dollars']:.2f}**  
Buyback Target: **${top['buyback_target_dollars']:.2f}**  
Possible Contracts @ 50% total risk cap: **{qty}**
"""
                        )

                        st.markdown("### Top Spread Leg Detail")
                        st.write(
                            f"""
Short Leg Mid: **{top['short_mid']:.2f}**  
Long Leg Mid: **{top['long_mid']:.2f}**  
Width: **{top['width']:.2f}**  
Short Delta: **{top['short_delta']:.3f}**
"""
                        )

                with tabs[3]:
                    st.subheader("Debug")

                    debug_symbol = st.selectbox("Debug Symbol", options=symbols, index=0, key="debug_symbol")
                    dbg = full_chain[full_chain["symbol"] == debug_symbol].copy()

                    quote = all_quotes.get(debug_symbol, {})
                    st.write("Underlying Quote")
                    st.json(quote)

                    st.write("Normalized Chain")
                    st.dataframe(build_debug_view(dbg), use_container_width=True, hide_index=True)

                    st.markdown("### Quote Diagnostics")
                    bad = dbg[
                        dbg["bid"].notna() &
                        dbg["ask"].notna() &
                        (dbg["ask"] < dbg["bid"])
                    ]
                    st.write(f"Bad ask < bid rows: **{len(bad)}**")

                    missing_mid = dbg[dbg["mid"].isna()]
                    st.write(f"Rows missing usable price: **{len(missing_mid)}**")

                    st.markdown("### Why prices can look off")
                    st.write(
                        """
- using `last` instead of `mid`
- stale option chain vs fresh underlying quote
- mixing expirations
- bad leg pairing
- delta sign issues on puts
- string strike sorting
"""
                    )

    except Exception as e:
        st.exception(e)