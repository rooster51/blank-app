import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Premium Selling Engine", layout="wide")

# =========================
# DATA
# =========================
@st.cache_data(ttl=300)
def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(how="all")
    return df


# =========================
# MARKET LOGIC
# =========================
def get_trend(df: pd.DataFrame) -> str:
    work = df.copy()
    work["ema9"] = work["Close"].ewm(span=9).mean()
    work["ema21"] = work["Close"].ewm(span=21).mean()
    work["ema50"] = work["Close"].ewm(span=50).mean()

    ema9 = float(work["ema9"].iloc[-1])
    ema21 = float(work["ema21"].iloc[-1])
    ema50 = float(work["ema50"].iloc[-1])
    price = float(work["Close"].iloc[-1])

    if ema9 > ema21 > ema50 and price > ema21:
        return "bullish"
    if ema9 < ema21 < ema50 and price < ema21:
        return "bearish"
    return "neutral"


def get_regime(df: pd.DataFrame) -> str:
    work = df.copy()
    work["range"] = work["High"] - work["Low"]
    work["atr10"] = work["range"].rolling(10).mean()

    if work["atr10"].isna().iloc[-1]:
        return "NORMAL_VOL"

    atr_pct = float(work["atr10"].iloc[-1] / work["Close"].iloc[-1]) * 100

    if atr_pct >= 2.2:
        return "HIGH_VOL"
    if atr_pct <= 1.0:
        return "LOW_VOL"
    return "NORMAL_VOL"


def get_levels(df: pd.DataFrame) -> tuple[float, float]:
    recent = df.tail(20)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


def get_supply_demand_zones(df: pd.DataFrame) -> dict:
    recent = df.tail(20)

    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    zone_width = max((resistance - support) * 0.08, float(df["Close"].iloc[-1]) * 0.003)

    demand_low = support
    demand_high = support + zone_width

    supply_low = resistance - zone_width
    supply_high = resistance

    return {
        "demand_low": round(demand_low, 2),
        "demand_high": round(demand_high, 2),
        "supply_low": round(supply_low, 2),
        "supply_high": round(supply_high, 2),
    }


def is_range_bound(df: pd.DataFrame) -> bool:
    work = df.tail(20).copy()
    total_range = float(work["High"].max() - work["Low"].min())
    avg_close = float(work["Close"].mean())

    if avg_close == 0:
        return False

    range_pct = (total_range / avg_close) * 100

    work["ema9"] = work["Close"].ewm(span=9).mean()
    work["ema21"] = work["Close"].ewm(span=21).mean()

    ema_spread_pct = abs(float(work["ema9"].iloc[-1] - work["ema21"].iloc[-1])) / avg_close * 100

    return range_pct < 6.0 and ema_spread_pct < 0.8


def get_price_location(price: float, support: float, resistance: float, zones: dict) -> str:
    if resistance <= support:
        return "unknown"

    range_width = resistance - support
    middle_low = support + (range_width * 0.4)
    middle_high = support + (range_width * 0.6)

    if zones["demand_low"] <= price <= zones["demand_high"]:
        return "near_support"

    if zones["supply_low"] <= price <= zones["supply_high"]:
        return "near_resistance"

    if middle_low <= price <= middle_high:
        return "middle"

    if price < zones["demand_low"]:
        return "below_support"

    if price > zones["supply_high"]:
        return "above_resistance"

    return "in_between"


# =========================
# TRADE RULES
# =========================
def get_trade_rules(df: pd.DataFrame, trend: str, regime: str, price_location: str) -> dict:
    rules = {
        "allow_trade": True,
        "reasons": [],
        "warnings": []
    }

    if df is None or df.empty or "Close" not in df.columns or len(df) < 20:
        rules["allow_trade"] = False
        rules["reasons"].append("Not enough price data")
        return rules

    work = df.copy()

    last_close = float(work["Close"].iloc[-1])
    prev_close = float(work["Close"].iloc[-2])
    day_move_pct = abs((last_close - prev_close) / prev_close) * 100

    work["range"] = work["High"] - work["Low"]
    atr10 = float(work["range"].rolling(10).mean().iloc[-1])
    today_range = float(work["range"].iloc[-1])

    ema9 = work["Close"].ewm(span=9).mean()
    ema21 = work["Close"].ewm(span=21).mean()
    ema_spread_pct = abs((float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / last_close) * 100

    last5 = work.tail(5).copy()
    up_days = int((last5["Close"] > last5["Open"]).sum())
    down_days = int((last5["Close"] < last5["Open"]).sum())

    if day_move_pct > 2.0:
        rules["allow_trade"] = False
        rules["reasons"].append(f"Daily move too large ({day_move_pct:.2f}%)")

    if today_range > atr10 * 1.75:
        rules["allow_trade"] = False
        rules["reasons"].append("Today's range is too large versus recent average")

    if up_days >= 2 and down_days >= 2 and regime == "HIGH_VOL":
        rules["allow_trade"] = False
        rules["reasons"].append("Recent price action is too choppy for premium selling")

    if trend == "neutral" and price_location != "middle":
        rules["warnings"].append("Trend is neutral")

    if ema_spread_pct < 0.15:
        rules["warnings"].append("Trend strength is weak")

    current_hour = datetime.now().hour
    if current_hour >= 15:
        rules["allow_trade"] = False
        rules["reasons"].append("Do not open new premium trades late in the day")

    if regime == "HIGH_VOL":
        rules["warnings"].append("High volatility can create overnight gap risk")

    if price_location == "middle":
        rules["warnings"].append("Middle of the range is better for condors than one-sided spreads")

    return rules


# =========================
# STRATEGY LOGIC
# =========================
def choose_premium_strategy(trend: str, regime: str, range_bound: bool, price_location: str) -> str:
    if range_bound and price_location == "middle":
        return "IRON_CONDOR"

    if trend == "bullish" and price_location == "near_support":
        return "PUT_CREDIT"

    if trend == "bearish" and price_location == "near_resistance":
        return "CALL_CREDIT"

    if trend == "neutral" and range_bound and price_location in ["near_support", "near_resistance", "middle"]:
        return "IRON_CONDOR"

    return "NO_TRADE"


def round_to_strike(value: float, increment: float = 1.0) -> float:
    return round(round(value / increment) * increment, 2)


def choose_strike_increment(price: float) -> float:
    if price < 25:
        return 0.5
    if price < 200:
        return 1.0
    return 5.0


def build_put_credit_spread(price: float, zones: dict, support: float, atr: float) -> dict:
    increment = choose_strike_increment(price)

    buffer = max(atr * 0.35, price * 0.005)
    short_target = min(zones["demand_low"] - buffer, support - (atr * 0.20))
    long_target = short_target - max(atr * 0.50, price * 0.01)

    short_strike = round_to_strike(short_target, increment)
    long_strike = round_to_strike(long_target, increment)

    if long_strike >= short_strike:
        long_strike = round_to_strike(short_strike - increment, increment)

    return {
        "label": f"Put Credit Spread → Sell {short_strike} / Buy {long_strike}",
        "short_strike": short_strike,
        "long_strike": long_strike,
        "zone_used": f"Demand {zones['demand_low']} - {zones['demand_high']}",
        "bias": "bullish"
    }


def build_call_credit_spread(price: float, zones: dict, resistance: float, atr: float) -> dict:
    increment = choose_strike_increment(price)

    buffer = max(atr * 0.35, price * 0.005)
    short_target = max(zones["supply_high"] + buffer, resistance + (atr * 0.20))
    long_target = short_target + max(atr * 0.50, price * 0.01)

    short_strike = round_to_strike(short_target, increment)
    long_strike = round_to_strike(long_target, increment)

    if long_strike <= short_strike:
        long_strike = round_to_strike(short_strike + increment, increment)

    return {
        "label": f"Call Credit Spread → Sell {short_strike} / Buy {long_strike}",
        "short_strike": short_strike,
        "long_strike": long_strike,
        "zone_used": f"Supply {zones['supply_low']} - {zones['supply_high']}",
        "bias": "bearish"
    }


def build_iron_condor(price: float, zones: dict, support: float, resistance: float, atr: float) -> dict:
    increment = choose_strike_increment(price)

    put_short_target = support - max(atr * 0.30, price * 0.005)
    put_long_target = put_short_target - max(atr * 0.50, price * 0.01)

    call_short_target = resistance + max(atr * 0.30, price * 0.005)
    call_long_target = call_short_target + max(atr * 0.50, price * 0.01)

    put_short = round_to_strike(put_short_target, increment)
    put_long = round_to_strike(put_long_target, increment)
    call_short = round_to_strike(call_short_target, increment)
    call_long = round_to_strike(call_long_target, increment)

    if put_long >= put_short:
        put_long = round_to_strike(put_short - increment, increment)

    if call_long <= call_short:
        call_long = round_to_strike(call_short + increment, increment)

    return {
        "label": f"Iron Condor → Buy {put_long} / Sell {put_short} | Sell {call_short} / Buy {call_long}",
        "put_short": put_short,
        "put_long": put_long,
        "call_short": call_short,
        "call_long": call_long,
        "zone_used": f"Demand {zones['demand_low']} - {zones['demand_high']} / Supply {zones['supply_low']} - {zones['supply_high']}",
        "bias": "neutral"
    }


def score_trade_quality(trend: str, regime: str, range_bound: bool, price_location: str, strategy: str, rules: dict) -> int:
    score = 50

    if strategy in ["PUT_CREDIT", "CALL_CREDIT", "IRON_CONDOR"]:
        score += 10

    if trend in ["bullish", "bearish"]:
        score += 10

    if regime == "NORMAL_VOL":
        score += 10
    elif regime == "LOW_VOL":
        score += 5
    elif regime == "HIGH_VOL":
        score -= 5

    if price_location in ["near_support", "near_resistance"]:
        score += 15

    if range_bound and strategy == "IRON_CONDOR":
        score += 10

    if not rules["allow_trade"]:
        score = 0

    score -= len(rules["warnings"]) * 3
    return max(0, min(score, 100))


# =========================
# EXECUTION PLANS
# =========================
def get_execution_plan(strategy: str) -> dict:
    plans = {
        "PUT_CREDIT": {
            "entry": "Enter only when price is at or near demand/support, not after a large bullish extension.",
            "profit_target": "Take profits at 40% to 60% of max profit.",
            "stop_rule": "Exit if spread value reaches 1.5x to 2x entry credit or support fails hard.",
            "holding_rule": "Avoid holding through major news or unstable overnight conditions."
        },
        "CALL_CREDIT": {
            "entry": "Enter only when price is at or near supply/resistance, not after a large bearish extension.",
            "profit_target": "Take profits at 40% to 60% of max profit.",
            "stop_rule": "Exit if spread value reaches 1.5x to 2x entry credit or resistance breaks cleanly.",
            "holding_rule": "Avoid holding through major news or unstable overnight conditions."
        },
        "IRON_CONDOR": {
            "entry": "Use only when price is range-bound and positioned away from both short strikes.",
            "profit_target": "Take profits at 35% to 50% of max profit.",
            "stop_rule": "Exit or reduce the threatened side early. Do not wait for max loss.",
            "holding_rule": "Avoid holding if price starts trending strongly out of range."
        }
    }
    return plans.get(strategy, {})


# =========================
# ENGINE
# =========================
def run_engine(ticker: str) -> dict:
    df = get_data(ticker)

    if df is None or df.empty:
        return {"error": "No price data returned"}

    if "Close" not in df.columns:
        return {"error": "Missing Close column"}

    if len(df) < 20:
        return {"error": "Not enough data"}

    try:
        price = float(df["Close"].iloc[-1])
    except Exception:
        return {"error": "Failed to read price data"}

    trend = get_trend(df)
    regime = get_regime(df)
    support, resistance = get_levels(df)
    zones = get_supply_demand_zones(df)
    range_bound = is_range_bound(df)
    price_location = get_price_location(price, support, resistance, zones)

    work = df.copy()
    work["range"] = work["High"] - work["Low"]
    atr10 = float(work["range"].rolling(10).mean().iloc[-1])

    rules = get_trade_rules(df, trend, regime, price_location)
    strategy = choose_premium_strategy(trend, regime, range_bound, price_location)

    if strategy == "NO_TRADE":
        rules["allow_trade"] = False
        rules["reasons"].append("Price is not in a premium-selling location")

    if not rules["allow_trade"]:
        score = score_trade_quality(trend, regime, range_bound, price_location, strategy, rules)
        return {
            "blocked": True,
            "price": price,
            "trend": trend,
            "regime": regime,
            "strategy": strategy,
            "rules": rules,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "zones": zones,
            "price_location": price_location,
            "score": score
        }

    if strategy == "PUT_CREDIT":
        trade = build_put_credit_spread(price, zones, support, atr10)
    elif strategy == "CALL_CREDIT":
        trade = build_call_credit_spread(price, zones, resistance, atr10)
    elif strategy == "IRON_CONDOR":
        trade = build_iron_condor(price, zones, support, resistance, atr10)
    else:
        return {"error": "No valid premium strategy found"}

    plan = get_execution_plan(strategy)
    score = score_trade_quality(trend, regime, range_bound, price_location, strategy, rules)

    return {
        "blocked": False,
        "price": price,
        "trend": trend,
        "regime": regime,
        "strategy": strategy,
        "trade": trade,
        "rules": rules,
        "plan": plan,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "zones": zones,
        "price_location": price_location,
        "range_bound": range_bound,
        "score": score
    }


# =========================
# UI
# =========================
st.title("💰 Premium Selling Engine")
st.caption("Premium-first logic: put credit spreads, call credit spreads, and iron condors based on location, trend, and regime.")

ticker = st.text_input("Enter Ticker", "SPY").strip().upper()

col1, col2 = st.columns(2)
run = col1.button("Run Strategy", use_container_width=True)
refresh = col2.button("🔄 Refresh Cache", use_container_width=True)

if refresh:
    st.cache_data.clear()
    st.success("Cache cleared")

if not ticker:
    st.warning("Enter a ticker.")
    st.stop()

if run:
    result = run_engine(ticker)

    if isinstance(result, dict) and "error" in result:
        st.error(f"⚠️ {result['error']}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{result['price']:.2f}")
    c2.metric("Trend", result["trend"])
    c3.metric("Regime", result["regime"])
    c4.metric("Score", f"{result['score']}/100")

    st.divider()

    z1, z2, z3 = st.columns(3)
    z1.metric("Support", f"{result['support']:.2f}")
    z2.metric("Resistance", f"{result['resistance']:.2f}")
    z3.metric("Location", result["price_location"])

    st.write(
        f"**Demand Zone:** {result['zones']['demand_low']} - {result['zones']['demand_high']}  \n"
        f"**Supply Zone:** {result['zones']['supply_low']} - {result['zones']['supply_high']}"
    )

    if result["blocked"]:
        st.error("🚫 NO TRADE")

        c5, c6 = st.columns(2)
        c5.metric("Suggested Strategy", result["strategy"])
        c6.metric("Status", "Blocked")

        st.subheader("Why")
        for reason in result["rules"]["reasons"]:
            st.write(f"- {reason}")

        if result["rules"]["warnings"]:
            st.subheader("Warnings")
            for warning in result["rules"]["warnings"]:
                st.write(f"- {warning}")

    else:
        c5, c6 = st.columns(2)
        c5.metric("Strategy", result["strategy"])
        c6.metric("Range Bound", "Yes" if result["range_bound"] else "No")

        st.success(f"✅ Trade Idea: {result['strategy']}")
        st.write(result["trade"]["label"])
        st.write(f"**Zone Used:** {result['trade']['zone_used']}")
        st.write(f"**Bias:** {result['trade']['bias']}")

        if result["rules"]["warnings"]:
            st.subheader("Warnings")
            for warning in result["rules"]["warnings"]:
                st.write(f"- {warning}")

        st.subheader("Execution Rules")
        plan = result["plan"]
        st.write(f"**Entry:** {plan.get('entry', 'N/A')}")
        st.write(f"**Profit Target:** {plan.get('profit_target', 'N/A')}")
        st.write(f"**Stop Rule:** {plan.get('stop_rule', 'N/A')}")
        st.write(f"**Holding Rule:** {plan.get('holding_rule', 'N/A')}")

        if datetime.now().hour >= 15:
            st.warning("⚠️ Avoid holding overnight.")