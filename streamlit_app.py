import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

st.set_page_config(layout="wide")

API_KEY = st.secrets["MARKETDATA_API_KEY"]

# =========================
# PRICE DATA (LIGHT)
# =========================
@st.cache_data(ttl=300)
def get_price_data(ticker):
    return yf.download(ticker, period="3mo", interval="1d")

# =========================
# MARKETDATA (HEAVY)
# =========================
@st.cache_data(ttl=600)
def get_option_chain(ticker):
    url = f"https://api.marketdata.app/v1/options/chain/{ticker}/?token={API_KEY}"
    res = requests.get(url)

    if res.status_code != 200:
        return None

    df = pd.DataFrame(res.json())
    return df

# =========================
# TREND + REGIME
# =========================
def get_trend(df):
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    return "bullish" if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] else "bearish"

def get_regime(df):
    move = abs(df["Close"].pct_change().iloc[-1]) * 100
    return "HIGH_VOL" if move > 1.2 else "LOW_VOL"

def is_range_bound(df):
    recent = df.tail(10)
    return (recent["High"].max() - recent["Low"].min()) < (df["High"].rolling(10).mean().iloc[-1] * 1.2)

# =========================
# FILTERS
# =========================
def liquidity_filter(df):
    df["spread"] = df["ask"] - df["bid"]
    return df[df["spread"] < 0.3]

def delta_filter(df, target=0.15, option_type="put"):
    df = df[df["optionType"] == option_type].copy()
    df["delta_diff"] = abs(abs(df["delta"]) - target)
    return df.sort_values("delta_diff").head(10)

# =========================
# POP
# =========================
def estimate_pop(delta):
    return round((1 - abs(delta)) * 100, 1)

# =========================
# STRATEGY ROUTER
# =========================
def choose_strategy(trend, regime, df_price):
    move = abs(df_price["Close"].pct_change().iloc[-1]) * 100

    if move < 0.7:
        return "CALENDAR"

    if regime == "HIGH_VOL":
        return "DEBIT"

    if regime == "LOW_VOL":
        return "CREDIT"

    if trend in ["bullish", "bearish"]:
        return "LEAPS"

    return "NO_TRADE"

# =========================
# CREDIT SPREAD
# =========================
def build_credit_spread(df, target_delta, width, option_type):
    candidates = delta_filter(df, target_delta, option_type)
    trades = []

    for _, short in candidates.iterrows():
        long_strike = short["strike"] - width if option_type == "put" else short["strike"] + width
        long = df[(df["strike"] == long_strike) & (df["optionType"] == option_type)]

        if not long.empty:
            long = long.iloc[0]
            credit = short["bid"] - long["ask"]
            risk = width - credit
            pop = estimate_pop(short["delta"])

            trades.append({
                "Short": short["strike"],
                "Long": long["strike"],
                "Credit": round(credit, 2),
                "Risk": round(risk, 2),
                "POP %": pop,
                "Delta": round(short["delta"], 2)
            })

    return pd.DataFrame(trades).sort_values("POP %", ascending=False).head(5)

# =========================
# DEBIT SPREAD
# =========================
def build_debit_spread(df, target_delta, width, option_type):
    candidates = delta_filter(df, target_delta, option_type)
    trades = []

    for _, buy in candidates.iterrows():
        sell_strike = buy["strike"] + width if option_type == "call" else buy["strike"] - width
        sell = df[(df["strike"] == sell_strike) & (df["optionType"] == option_type)]

        if not sell.empty:
            sell = sell.iloc[0]
            debit = buy["ask"] - sell["bid"]
            reward = width - debit

            trades.append({
                "Buy": buy["strike"],
                "Sell": sell["strike"],
                "Debit": round(debit, 2),
                "Max Profit": round(reward, 2),
                "Delta": round(buy["delta"], 2)
            })

    return pd.DataFrame(trades).head(5)

# =========================
# LEAPS
# =========================
def build_leaps(df, trend):
    long_dated = df[df["daysToExpiration"] > 120]

    if trend == "bullish":
        calls = long_dated[long_dated["optionType"] == "call"]
        return calls.sort_values("delta", ascending=False).head(5)[["strike","delta","ask"]]

    else:
        puts = long_dated[long_dated["optionType"] == "put"]
        return puts.sort_values("delta").head(5)[["strike","delta","ask"]]

# =========================
# CALENDAR
# =========================
def build_calendar(df, price):
    near = df[df["daysToExpiration"] < 30]
    far = df[df["daysToExpiration"] > 60]

    atm = min(df["strike"], key=lambda x: abs(x - price))

    near_leg = near[near["strike"] == atm].iloc[0]
    far_leg = far[far["strike"] == atm].iloc[0]

    return {
        "Strike": atm,
        "Sell Exp": near_leg["daysToExpiration"],
        "Buy Exp": far_leg["daysToExpiration"]
    }

# =========================
# MAIN ENGINE
# =========================
def run_engine(ticker):
    df_price = get_price_data(ticker)
    chain = get_option_chain(ticker)

    if chain is None or chain.empty:
        return None

    chain = liquidity_filter(chain)

    trend = get_trend(df_price)
    regime = get_regime(df_price)
    strategy = choose_strategy(trend, regime, df_price)

    if strategy == "CREDIT":
        result = build_credit_spread(chain, 0.15, 5, "put" if trend=="bullish" else "call")

    elif strategy == "DEBIT":
        result = build_debit_spread(chain, 0.5, 5, "call" if trend=="bullish" else "put")

    elif strategy == "LEAPS":
        result = build_leaps(chain, trend)

    elif strategy == "CALENDAR":
        result = build_calendar(chain, df_price["Close"].iloc[-1])

    else:
        result = None

    return trend, regime, strategy, result

# =========================
# UI
# =========================
st.title("🚀 Options Strategy Engine (Level 4.5 PRO)")

ticker = st.sidebar.text_input("Ticker", "SPY")

if st.sidebar.button("Run Strategy"):
    result = run_engine(ticker)

    if result is None:
        st.error("⚠️ API issue — try again in a few seconds")
    else:
        trend, regime, strategy, output = result

        col1, col2, col3 = st.columns(3)
        col1.metric("Trend", trend)
        col2.metric("Regime", regime)
        col3.metric("Strategy", strategy)

        st.divider()

        st.subheader("Trade Setup")

        if isinstance(output, pd.DataFrame):
            st.dataframe(output)
        else:
            st.write(output)

        if datetime.now().hour >= 15:
            st.warning("⚠️ Avoid holding overnight")