import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

st.set_page_config(layout="wide")

# =========================
# API KEY
# =========================
try:
    API_KEY = st.secrets["MARKETDATA_API_KEY"]
except:
    st.error("❌ Missing MarketData API key in Streamlit secrets")
    st.stop()

# =========================
# PRICE DATA
# =========================
@st.cache_data(ttl=300)
def get_price_data(ticker):
    return yf.download(ticker, period="3mo", interval="1d")

# =========================
# MARKETDATA (SAFE)
# =========================
@st.cache_data(ttl=600)
def get_option_chain(ticker):
    url = f"https://api.marketdata.app/v1/options/chain/{ticker}/?token={API_KEY}"

    try:
        res = requests.get(url)

        if res.status_code != 200:
            return {
                "error": f"HTTP {res.status_code}",
                "message": res.text[:300]
            }

        data = res.json()

        if isinstance(data, dict) and "error" in data:
            return {
                "error": "API Error",
                "message": str(data)
            }

        df = pd.DataFrame(data)

        required = ["strike", "optionType", "delta", "bid", "ask"]
        if not all(col in df.columns for col in required):
            return {
                "error": "Bad Data Format",
                "message": f"Columns received: {list(df.columns)}"
            }

        return df

    except Exception as e:
        return {
            "error": "Exception",
            "message": str(e)
        }

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
        return long_dated[long_dated["optionType"] == "call"].sort_values("delta", ascending=False).head(5)[["strike","delta","ask"]]
    else:
        return long_dated[long_dated["optionType"] == "put"].sort_values("delta").head(5)[["strike","delta","ask"]]

# =========================
# CALENDAR
# =========================
def build_calendar(df, price):
    near = df[df["daysToExpiration"] < 30]
    far = df[df["daysToExpiration"] > 60]

    if near.empty or far.empty:
        return {"error": "Not enough expirations for calendar"}

    atm = min(df["strike"], key=lambda x: abs(x - price))

    near_leg = near[near["strike"] == atm]
    far_leg = far[far["strike"] == atm]

    if near_leg.empty or far_leg.empty:
        return {"error": "No matching strikes for calendar"}

    return {
        "Strike": atm,
        "Sell Exp (days)": int(near_leg.iloc[0]["daysToExpiration"]),
        "Buy Exp (days)": int(far_leg.iloc[0]["daysToExpiration"])
    }

# =========================
# ENGINE
# =========================
def run_engine(ticker):
    df_price = get_price_data(ticker)
    chain = get_option_chain(ticker)

    # 🔴 ERROR HANDLING
    if isinstance(chain, dict) and "error" in chain:
        return chain

    if chain is None or chain.empty:
        return {"error": "No Data", "message": "Empty option chain"}

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
st.title("🚀 Options Strategy Engine")

ticker = st.text_input("Enter Ticker", "SPY")

col1, col2 = st.columns(2)
run = col1.button("Run Strategy")
refresh = col2.button("🔄 Refresh Cache")

if refresh:
    st.cache_data.clear()
    st.success("Cache cleared")

if run:
    result = run_engine(ticker)

    # 🔴 SHOW REAL ERRORS
    if isinstance(result, dict) and "error" in result:
        st.error(f"⚠️ {result['error']}")
        st.code(result["message"])
    else:
        trend, regime, strategy, output = result

        c1, c2, c3 = st.columns(3)
        c1.metric("Trend", trend)
        c2.metric("Regime", regime)
        c3.metric("Strategy", strategy)

        st.divider()
        st.subheader("Trade Setup")

        if isinstance(output, pd.DataFrame):
            st.dataframe(output, use_container_width=True)
        else:
            st.write(output)

        if datetime.now().hour >= 15:
            st.warning("⚠️ Avoid holding overnight")