import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Options Strategy Engine", layout="wide")

# =========================
# DATA
# =========================
@st.cache_data(ttl=300)
def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(how="all")
    return df


# =========================
# MARKET LOGIC
# =========================
def get_trend(df: pd.DataFrame) -> str:
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()

    if float(df["ema9"].iloc[-1]) > float(df["ema21"].iloc[-1]):
        return "bullish"
    return "bearish"


def get_regime(df: pd.DataFrame) -> str:
    move = abs(float(df["Close"].pct_change().iloc[-1])) * 100
    return "HIGH_VOL" if move > 1.2 else "LOW_VOL"


def get_levels(df: pd.DataFrame) -> tuple[float, float]:
    recent = df.tail(20)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


def is_range(df: pd.DataFrame) -> bool:
    recent = df.tail(10)
    recent_range = float(recent["High"].max() - recent["Low"].min())
    avg_high = float(df["High"].rolling(10).mean().iloc[-1])
    return recent_range < avg_high * 1.2


def choose_strategy(trend: str, regime: str, range_bound: bool) -> str:
    if range_bound and regime == "LOW_VOL":
        return "IRON_CONDOR"

    if range_bound and regime == "HIGH_VOL":
        return "CALENDAR"

    if regime == "HIGH_VOL":
        return "DEBIT"

    if regime == "LOW_VOL":
        return "CREDIT"

    return "LEAPS"


# =========================
# TRADE RULES
# =========================
def get_trade_rules(df: pd.DataFrame, trend: str, regime: str) -> dict:
    rules = {
        "allow_trade": True,
        "reasons": [],
        "warnings": []
    }

    if df is None or df.empty or "Close" not in df.columns or len(df) < 20:
        rules["allow_trade"] = False
        rules["reasons"].append("Not enough price data")
        return rules

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    day_move_pct = abs((last_close - prev_close) / prev_close) * 100

    work = df.copy()
    work["range"] = work["High"] - work["Low"]
    atr10 = float(work["range"].rolling(10).mean().iloc[-1])
    today_range = float(work["range"].iloc[-1])

    ema9 = work["Close"].ewm(span=9).mean()
    ema21 = work["Close"].ewm(span=21).mean()
    ema_spread_pct = abs((float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / last_close) * 100

    last5 = work.tail(5).copy()
    up_days = int((last5["Close"] > last5["Open"]).sum())
    down_days = int((last5["Close"] < last5["Open"]).sum())

    if day_move_pct > 1.8:
        rules["allow_trade"] = False
        rules["reasons"].append(f"Daily move too large ({day_move_pct:.2f}%)")

    if today_range > atr10 * 1.6:
        rules["allow_trade"] = False
        rules["reasons"].append("Today’s range is too large versus recent average")

    if ema_spread_pct < 0.15 and regime == "LOW_VOL":
        rules["warnings"].append("Trend strength is weak")

    if up_days >= 2 and down_days >= 2 and regime == "HIGH_VOL":
        rules["allow_trade"] = False
        rules["reasons"].append("Recent price action is too choppy")

    current_hour = datetime.now().hour
    if current_hour >= 15:
        rules["allow_trade"] = False
        rules["reasons"].append("Do not open new trades late in the day")

    if regime == "HIGH_VOL":
        rules["warnings"].append("Avoid holding overnight in high volatility")

    return rules


# =========================
# STRATEGY BUILDERS
# =========================
def credit_spread(price: float, support: float, resistance: float, trend: str) -> str:
    if trend == "bullish":
        short_strike = round(support * 0.98, 2)
        long_strike = round(support * 0.95, 2)
        return f"Put Credit → Sell {short_strike} / Buy {long_strike}"
    short_strike = round(resistance * 1.02, 2)
    long_strike = round(resistance * 1.05, 2)
    return f"Call Credit → Sell {short_strike} / Buy {long_strike}"


def debit_spread(price: float, trend: str) -> str:
    if trend == "bullish":
        buy_strike = round(price * 0.99, 2)
        sell_strike = round(price * 1.03, 2)
        return f"Call Debit → Buy {buy_strike} / Sell {sell_strike}"
    buy_strike = round(price * 1.01, 2)
    sell_strike = round(price * 0.97, 2)
    return f"Put Debit → Buy {buy_strike} / Sell {sell_strike}"


def leaps(price: float, trend: str) -> str:
    if trend == "bullish":
        strike = round(price * 0.90, 2)
        return f"LEAPS Call → Buy ITM around {strike} (6–12 months)"
    strike = round(price * 1.10, 2)
    return f"LEAPS Put → Buy ITM around {strike} (6–12 months)"


def calendar(price: float) -> str:
    strike = round(price, 2)
    return f"Calendar → Sell near-term {strike}, Buy longer-term same strike"


def iron_condor(support: float, resistance: float) -> str:
    return f"Iron Condor → Expected range {round(support, 2)} to {round(resistance, 2)}"


# =========================
# EXECUTION PLANS
# =========================
def get_execution_plan(strategy: str) -> dict:
    plans = {
        "CREDIT": {
            "entry": "Only enter near support or resistance, not in the middle of the range.",
            "profit_target": "Take profits at 40% to 50% of max profit.",
            "stop_rule": "Exit if spread value reaches 2x entry credit.",
            "holding_rule": "Do not hold overnight in unstable conditions."
        },
        "DEBIT": {
            "entry": "Enter only with confirmed momentum, not after a huge extension bar.",
            "profit_target": "Take profits at 25% to 40% of max profit on a quick move.",
            "stop_rule": "Cut the trade if premium loses 35% to 40%.",
            "holding_rule": "Only hold overnight if the trend is strong and volatility is not expanding."
        },
        "LEAPS": {
            "entry": "Use only with strong higher-timeframe trend alignment.",
            "profit_target": "Scale out into strength instead of exiting all at once.",
            "stop_rule": "Exit on a clear trend break, not on normal short-term noise.",
            "holding_rule": "Designed for longer holding periods."
        },
        "CALENDAR": {
            "entry": "Use when price is near the expected pin area or center of the range.",
            "profit_target": "Take profits before front expiration behavior becomes erratic.",
            "stop_rule": "Exit if price moves too far away from the strike.",
            "holding_rule": "Monitor closely as the short leg nears expiration."
        },
        "IRON_CONDOR": {
            "entry": "Use only in stable, range-bound markets.",
            "profit_target": "Take profits at 35% to 50% of max profit.",
            "stop_rule": "Exit the threatened side early. Do not wait for max loss.",
            "holding_rule": "Avoid overnight exposure if the range starts breaking."
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
    range_bound = is_range(df)

    strategy = choose_strategy(trend, regime, range_bound)
    rules = get_trade_rules(df, trend, regime)

    if not rules["allow_trade"]:
        return {
            "blocked": True,
            "price": price,
            "trend": trend,
            "regime": regime,
            "strategy": strategy,
            "rules": rules
        }

    if strategy == "CREDIT":
        trade = credit_spread(price, support, resistance, trend)
    elif strategy == "DEBIT":
        trade = debit_spread(price, trend)
    elif strategy == "LEAPS":
        trade = leaps(price, trend)
    elif strategy == "CALENDAR":
        trade = calendar(price)
    elif strategy == "IRON_CONDOR":
        trade = iron_condor(support, resistance)
    else:
        return {"error": "No valid strategy found"}

    plan = get_execution_plan(strategy)

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
        "resistance": round(resistance, 2)
    }


# =========================
# UI
# =========================
st.title("🚀 Options Strategy Engine")
st.caption("This engine blocks trades in unstable conditions and adds execution rules to reduce bad entries.")

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"{result['price']:.2f}")
    c2.metric("Trend", result["trend"])
    c3.metric("Regime", result["regime"])

    st.divider()

    if result["blocked"]:
        st.error("🚫 NO TRADE")

        c4, c5 = st.columns(2)
        c4.metric("Suggested Strategy", result["strategy"])
        c5.metric("Status", "Blocked")

        st.subheader("Why")
        for reason in result["rules"]["reasons"]:
            st.write(f"- {reason}")

        if result["rules"]["warnings"]:
            st.subheader("Warnings")
            for warning in result["rules"]["warnings"]:
                st.write(f"- {warning}")

    else:
        c4, c5, c6 = st.columns(3)
        c4.metric("Strategy", result["strategy"])
        c5.metric("Support", f"{result['support']:.2f}")
        c6.metric("Resistance", f"{result['resistance']:.2f}")

        st.success(f"✅ Trade Idea: {result['strategy']}")
        st.write(result["trade"])

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