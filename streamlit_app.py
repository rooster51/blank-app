import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide")

# =========================
# DATA
# =========================
@st.cache_data(ttl=300)
def get_data(ticker):
    return yf.download(ticker, period="3mo", interval="1d")

# =========================
# TREND
# =========================
def get_trend(df):
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()

    return "bullish" if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] else "bearish"

# =========================
# VOL REGIME
# =========================
def get_regime(df):
    move = abs(df["Close"].pct_change().iloc[-1]) * 100
    return "HIGH_VOL" if move > 1.2 else "LOW_VOL"

# =========================
# SUPPORT / RESISTANCE
# =========================
def get_levels(df):
    recent = df.tail(20)
    return recent["Low"].min(), recent["High"].max()

# =========================
# RANGE DETECTION
# =========================
def is_range(df):
    recent = df.tail(10)
    return (recent["High"].max() - recent["Low"].min()) < df["High"].rolling(10).mean().iloc[-1] * 1.2

# =========================
# STRATEGY ROUTER
# =========================
def choose_strategy(trend, regime, is_range):
    if is_range and regime == "LOW_VOL":
        return "IRON_CONDOR"

    if is_range and regime == "HIGH_VOL":
        return "CALENDAR"

    if regime == "HIGH_VOL":
        return "DEBIT"

    if regime == "LOW_VOL":
        return "CREDIT"

    return "LEAPS"

# =========================
# STRATEGY BUILDERS
# =========================
def credit_spread(price, support, resistance, trend):
    if trend == "bullish":
        return f"Put Credit → Sell below {round(support*0.98,2)} / Buy lower"
    else:
        return f"Call Credit → Sell above {round(resistance*1.02,2)} / Buy higher"

def debit_spread(price, trend):
    if trend == "bullish":
        return f"Call Debit → Buy near {round(price*0.99,2)} / Sell above"
    else:
        return f"Put Debit → Buy near {round(price*1.01,2)} / Sell below"

def leaps(price, trend):
    if trend == "bullish":
        return f"LEAPS Call → Buy ITM around {round(price*0.9,2)} (6–12 months)"
    else:
        return f"LEAPS Put → Buy ITM around {round(price*1.1,2)} (6–12 months)"

def calendar(price):
    return f"Calendar → Sell near-term {round(price,2)}, Buy longer-term same strike"

def iron_condor(support, resistance):
    return f"Iron Condor → Range {round(support,2)} - {round(resistance,2)}"

# =========================
# ENGINE
# =========================
def run_engine(ticker):
    df = get_data(ticker)

    price = df["Close"].iloc[-1]
    trend = get_trend(df)
    regime = get_regime(df)
    support, resistance = get_levels(df)
    range_bound = is_range(df)

    strategy = choose_strategy(trend, regime, range_bound)

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

    return price, trend, regime, strategy, trade

# =========================
# UI
# =========================
st.title("🚀 Options Strategy Engine (Stable Mode)")

ticker = st.text_input("Enter Ticker", "SPY")

col1, col2 = st.columns(2)
run = col1.button("Run Strategy")
refresh = col2.button("🔄 Refresh Cache")

if refresh:
    st.cache_data.clear()
    st.success("Cache cleared")

if run:
    price, trend, regime, strategy, trade = run_engine(ticker)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", round(price,2))
    c2.metric("Trend", trend)
    c3.metric("Regime", regime)

    st.divider()

    st.success(f"Strategy: {strategy}")
    st.write(trade)

    if datetime.now().hour >= 15:
        st.warning("⚠️ Avoid overnight risk")