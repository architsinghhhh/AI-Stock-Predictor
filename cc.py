import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import math

# --- STOCKS LIST ---
STOCKS = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX"
}

st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 AI-Powered Stock Prediction App")
st.markdown(
    """
    Select a stock, pick a date range, and get AI-driven 7-day price forecasts with confidence scores.
    """
)

# --- SIDEBAR ---
st.sidebar.header("1. Select Stock & Date Range")
stock_name = st.sidebar.selectbox("Choose a stock", list(STOCKS.keys()))
ticker = STOCKS[stock_name]

date_options = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 180 days": 180,
    "Last 1 year": 365,
    "Custom Range": None
}
date_choice = st.sidebar.selectbox("Select period", list(date_options.keys()))
if date_options[date_choice]:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=date_options[date_choice])
else:
    start_date = st.sidebar.date_input("Start date", datetime.today() - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", datetime.today())
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")

# --- DATA FETCH ---
@st.cache_data(show_spinner=True)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

with st.spinner("Fetching stock data..."):
    df = fetch_data(ticker, start_date, end_date + timedelta(days=1))

if df.empty:
    st.error("No data found for the selected period.")
    st.stop()

# --- REAL-TIME METRICS ---
latest = yf.Ticker(ticker).info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${latest.get('regularMarketPrice', 'N/A')}")
col2.metric("Volume", f"{latest.get('volume', 'N/A')}")
col3.metric("Market Cap", f"${latest.get('marketCap', 'N/A'):,}")
col4.metric("52W High/Low", f"${latest.get('fiftyTwoWeekHigh', 'N/A')} / ${latest.get('fiftyTwoWeekLow', 'N/A')}")

# --- ML PREDICTION (move this above the chart section) ---
# Prepare data for ML
df_ml = df[['Close']].copy()
df_ml['Target'] = df_ml['Close'].shift(-1)
df_ml = df_ml.dropna()
X = np.arange(len(df_ml)).reshape(-1, 1)
y = df_ml['Target'].values

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# Confidence score (inverse RMSE, scaled)
confidence = max(0, 100 - rmse)

# Predict next 7 days
future_X = np.arange(len(df_ml), len(df_ml)+7).reshape(-1, 1)
future_preds = model.predict(future_X)
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(7)]

# --- CHARTS WITH TABS ---
st.subheader(f"{stock_name} Price Analysis")
tabs = st.tabs(["Historical", "Predictions", "Combined View"])

with tabs[0]:
    # Historical
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='royalblue'),
        fill='tozeroy',
        fillcolor='rgba(65,105,225,0.2)'
    ))
    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tabs[1]:
    # Predictions
    fig_pred = go.Figure()
    
    # Example for predictions
    hover_text = [f"Predicted: ${p:.2f}<br>Confidence: {confidence:.1f}%" for p in future_preds]
    fig_pred.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode='lines+markers',
        name='Prediction',
        line=dict(color='orange', dash='dash'),
        text=hover_text,
        hoverinfo='text'
    ))
    fig_pred.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with tabs[2]:
    # Combined
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='royalblue'),
        fill='tozeroy',
        fillcolor='rgba(65,105,225,0.2)'
    ))
    fig_combined.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode='lines+markers',
        name='Prediction',
        line=dict(color='orange', dash='dash')
    ))
    fig_combined.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_combined, use_container_width=True)

# --- AI 7-Day Price Prediction (keep this below the chart section) ---
st.subheader("🔮 AI 7-Day Price Prediction")
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
st.write("**7-Day Forecast:**")
st.dataframe(pred_df.set_index('Date').style.format({"Predicted Close": "${:.2f}"}))

st.info(f"Prediction Confidence Score: **{confidence:.1f}/100** (lower RMSE is better)")

# --- Trend Analysis ---
st.subheader("📊 Trend Analysis")

if len(df['Close']) >= 2:
    first_close = float(df['Close'].iloc[0])
    last_close = float(df['Close'].iloc[-1])
    if not (math.isnan(first_close) or math.isnan(last_close)):
        change = (last_close - first_close) / first_close * 100
        trend = "upward 📈" if change > 0 else "downward 📉"
        st.write(f"Over the selected period, the stock price changed by **{change:.2f}%** ({trend}).")
    else:
        st.write("Not enough valid data for trend analysis.")
else:
    st.write("Not enough data for trend analysis.")

# --- ML Prediction Analysis Section ---
import streamlit.components.v1 as components

st.subheader("🤖 ML Prediction Analysis")

# Next predicted price and date
next_pred_price = future_preds[0]
next_pred_date = future_dates[0].strftime("%Y-%m-%d")

# Confidence as percent
conf_percent = min(max(confidence, 0), 100)

# Ensure last_close is a scalar float
last_close = float(df['Close'].iloc[-1])
bullish = "Bullish" if next_pred_price > last_close else "Bearish"
bullish_color = "green" if bullish == "Bullish" else "red"

# Calculate moving averages, RSI, MACD (simple versions)
ma50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
ma200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi_series = calc_rsi(df['Close']) if len(df) >= 15 else None
rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

def calc_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

macd_val, macd_signal = calc_macd(df['Close']) if len(df) >= 26 else (None, None)

def to_scalar(val):
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val) if val is not None and not pd.isna(val) else None

macd_val = to_scalar(macd_val)
macd_signal = to_scalar(macd_signal)

# Analysis summary
current_price = float(df['Close'].iloc[-1])
prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else current_price
if prev_close != 0:
    price_change = ((current_price - prev_close) / prev_close * 100)
else:
    price_change = 0
    
st.markdown(f"""
**Next Price Prediction**  
<span style="font-size:1.5em;font-weight:bold;">{conf_percent:.1f}% Confidence</span>  
<span style="font-size:2em;font-weight:bold;">${next_pred_price:.2f}</span>  
<span style="color:{bullish_color};font-size:1.2em;font-weight:bold;">{bullish}</span>  
Expected for <b>{next_pred_date}</b>
""", unsafe_allow_html=True)

st.markdown("**Analysis Summary**")
summary = f"""
{stock_name} ({ticker}) is currently trading at ${current_price:.2f}, reflecting a {'increase' if price_change >= 0 else 'decrease'} of {abs(price_change):.3f}% from the previous close. 
"""

if (
    ma50 is not None and ma200 is not None
    and not math.isnan(ma50) and not math.isnan(ma200)
):
    summary += f"The 50-day moving average (MA50) is ${ma50:.2f} and the 200-day moving average (MA200) is ${ma200:.2f}, indicating {'bullish' if ma50 > ma200 else 'bearish'} momentum. "
if rsi is not None and not math.isnan(rsi):
    summary += f"Relative Strength Index (RSI) is {rsi:.2f}, suggesting a {'Buy' if rsi < 70 and rsi > 50 else 'Neutral' if rsi <= 70 else 'Overbought'} signal. "
if (
    macd_val is not None and macd_signal is not None
    and not math.isnan(macd_val) and not math.isnan(macd_signal)
):
    summary += f"MACD is {macd_val:.2f}, {'above' if macd_val > macd_signal else 'below'} the signal line ({macd_signal:.2f}), indicating a {'Buy' if macd_val > macd_signal else 'Sell'} signal. "

summary += f"\n\nGiven these factors, the stock is expected to continue its {'upward' if bullish == 'Bullish' else 'downward'} trajectory over the next seven days, with a predicted price of ${future_preds[-1]:.2f} by {future_dates[-1].strftime('%B %d, %Y')}, and a confidence level of {conf_percent:.1f}%."

st.markdown(summary)

st.markdown(
    "[Technical indicators source: investing.com](https://www.investing.com/equities/facebook-inc-technical?utm_source=openai)"
)

st.caption("Powered by Streamlit, yfinance, scikit-learn, and Plotly. For educational use only.")