import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, period="6mo", interval="1d")

# =========================
# LOAD OPTIONS
# =========================
@st.cache_data
def load_options(ticker):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    return tk, expirations

def get_option_chain(tk, expiry):
    chain = tk.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    return calls, puts

# =========================
# TREND
# =========================
def get_trend(df):
    df['ema9'] = df['Close'].ewm(span=9).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()

    if df['ema9'].iloc[-1] > df['ema21'].iloc[-1]:
        return "bullish"
    return "bearish"

# =========================
# VOL REGIME
# =========================
def get_regime(df):
    move = abs(df['Close'].pct_change().iloc[-1]) * 100
    return "HIGH_VOL" if move > 1.2 else "LOW_VOL"

# =========================
# DELTA PROXY
# =========================
def estimate_delta(price, strike):
    diff = abs(price - strike) / price
    return round(max(0.05, 1 - diff * 5), 2)

# =========================
# FIND STRIKES
# =========================
def find_strikes(options, price, target="OTM"):
    options = options.copy()
    options['distance'] = abs(options['strike'] - price)

    otm = options[options['strike'] > price] if target == "CALL" else options[options['strike'] < price]

    otm = otm.sort_values("distance")

    return otm.head(5)

# =========================
# BUILD CREDIT SPREAD
# =========================
def build_credit_spread(price, calls, puts, trend):
    if trend == "bullish":
        candidates = find_strikes(puts, price, "PUT")
        short = candidates.iloc[1]
        long = candidates.iloc[3]

        return f"PUT CREDIT → Sell {short['strike']} / Buy {long['strike']}"

    else:
        candidates = find_strikes(calls, price, "CALL")
        short = candidates.iloc[1]
        long = candidates.iloc[3]

        return f"CALL CREDIT → Sell {short['strike']} / Buy {long['strike']}"

# =========================
# BUILD DEBIT SPREAD
# =========================
def build_debit_spread(price, calls, puts, trend):
    if trend == "bullish":
        candidates = find_strikes(calls, price, "CALL")
        buy = candidates.iloc[0]
        sell = candidates.iloc[2]

        return f"CALL DEBIT → Buy {buy['strike']} / Sell {sell['strike']}"

    else:
        candidates = find_strikes(puts, price, "PUT")
        buy = candidates.iloc[0]
        sell = candidates.iloc[2]

        return f"PUT DEBIT → Buy {buy['strike']} / Sell {sell['strike']}"

# =========================
# LEAPS
# =========================
def build_leaps(price, calls, puts, trend):
    if trend == "bullish":
        itm = calls[calls['strike'] < price].sort_values("strike", ascending=False)
        return f"LEAPS CALL → Buy {itm.iloc[0]['strike']} (long-term)"

    else:
        itm = puts[puts['strike'] > price].sort_values("strike")
        return f"LEAPS PUT → Buy {itm.iloc[0]['strike']} (long-term)"

# =========================
# MAIN ENGINE
# =========================
def run_engine(ticker, expiry):
    df = load_data(ticker)
    tk, expirations = load_options(ticker)

    price = df['Close'].iloc[-1]
    trend = get_trend(df)
    regime = get_regime(df)

    calls, puts = get_option_chain(tk, expiry)

    if regime == "LOW_VOL":
        trade = build_credit_spread(price, calls, puts, trend)
        strategy = "Credit Spread"
    else:
        trade = build_debit_spread(price, calls, puts, trend)
        strategy = "Debit Spread"

    leaps = build_leaps(price, calls, puts, trend)

    return price, trend, regime, strategy, trade, leaps

# =========================
# UI
# =========================
st.title("📊 Level 3 Options Engine (REAL CONTRACTS)")

ticker = st.sidebar.text_input("Ticker", "SPY")

tk = yf.Ticker(ticker)
expirations = tk.options

expiry = st.sidebar.selectbox("Expiration", expirations)

run = st.sidebar.button("Run Engine")

if run:
    price, trend, regime, strategy, trade, leaps = run_engine(ticker, expiry)

    col1, col2, col3 = st.columns(3)

    col1.metric("Price", round(price, 2))
    col2.metric("Trend", trend)
    col3.metric("Regime", regime)

    st.divider()

    st.success(f"Primary Strategy: {strategy}")
    st.write(trade)

    st.divider()

    st.subheader("Alternative (Long-Term)")
    st.write(leaps)

    if datetime.now().hour >= 15:
        st.warning("⚠️ Avoid holding overnight in high volatility markets")