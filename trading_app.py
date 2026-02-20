import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. UI SETUP (Dark Theme, Professional Layout) ---
st.set_page_config(page_title="Quantitative Alpha Model", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to make it look "Ivy League" and sleek
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #E0E6ED; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1E2530; padding: 15px; border-radius: 8px; border: 1px solid #2B3544; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Institutional Systematic Equity Model")
st.markdown("### Accounting for Slippage, Execution Fees, and Short-Term Tax Drag")

# --- 2. THE ENGINE (Data & Friction Math) ---
@st.cache_data
def load_and_simulate(ticker, start_date):
    # Fetch Data
    df = yf.download(ticker, start=start_date, progress=False)
    df['Return'] = df['Close'].pct_change()
    
    # Fake a signal for demonstration (e.g., Buy if above 200-Day Moving Average)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = np.where(df['Close'] > df['SMA_200'], 1, 0) # 1 = Invested, 0 = Cash
    
    # Identify Trades (When signal changes)
    df['Trade'] = df['Signal'].diff().abs() 
    
    # --- APPLYING REAL WORLD FRICTION ---
    SLIPPAGE = 0.0005  # 5 basis points lost on every trade
    COMMISSION = 0.0001 # 1 basis point broker fee
    TAX_RATE = 0.30     # 30% short term capital gains tax
    
    # Calculate Gross Returns
    df['Strategy_Gross'] = df['Signal'].shift(1) * df['Return']
    
    # Deduct Slippage and Commissions on trade days
    df['Friction_Cost'] = df['Trade'] * (SLIPPAGE + COMMISSION)
    df['Strategy_Net'] = df['Strategy_Gross'] - df['Friction_Cost'].fillna(0)
    
    # Calculate Cumulative (Pre-Tax)
    df['Equity_Curve_Gross'] = (1 + df['Strategy_Gross'].fillna(0)).cumprod()
    
    # --- TAX DRAG SIMULATION (Annualized) ---
    # To model tax, we reduce the net return by the tax rate
    df['Strategy_Post_Tax'] = np.where(df['Strategy_Net'] > 0, 
                                       df['Strategy_Net'] * (1 - TAX_RATE), 
                                       df['Strategy_Net'])
    df['Equity_Curve_Real'] = (1 + df['Strategy_Post_Tax'].fillna(0)).cumprod()
    
    return df.dropna()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Model Parameters")
    ticker = st.text_input("Asset Ticker", "QQQ")
    start_year = st.slider("Backtest Start Year", 2000, 2020, 2010)
    st.markdown("---")
    st.markdown("**Real World Assumptions:**\n* Slippage: 5 bps\n* Commissions: 1 bps\n* Tax Rate: 30% on gains")

# Run the engine
df = load_and_simulate(ticker, f"{start_year}-01-01")

# --- 4. TABS SETUP ---
tab_past, tab_future = st.tabs(["📜 Historical Backtest (Real-World)", "🔮 Future Outlook (Probabilities)"])

# ==========================================
# TAB 1: THE PAST (Brutal Reality Backtest)
# ==========================================
with tab_past:
    # Summary Metrics
    total_gross = (df['Equity_Curve_Gross'].iloc[-1] - 1) * 100
    total_net = (df['Equity_Curve_Real'].iloc[-1] - 1) * 100
    money_lost_to_friction = total_gross - total_net
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Return (Illusion)", f"{total_gross:.1f}%")
    col2.metric("Post-Tax & Fee Return (Reality)", f"{total_net:.1f}%", f"-{money_lost_to_friction:.1f}% lost to friction", delta_color="inverse")
    col3.metric("Trades Executed", int(df['Trade'].sum()))
    col4.metric("Current Regime", "Bull (Invested)" if df['Signal'].iloc[-1] == 1 else "Bear (Cash)")

    # The Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Curve_Gross'], name='Academic Model (No Fees)', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Curve_Real'], name='Real World Model (Taxes + Fees)', line=dict(color='#00FFAA', width=2)))
    
    fig.update_layout(
        title="Equity Curve: Academic vs. Real World",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333333'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: THE FUTURE (Monte Carlo & Tomorrow's Signal)
# ==========================================
with tab_future:
    st.markdown("### 🤖 Tomorrow's Execution Plan")
    
    last_price = df['Close'].iloc[-1]
    sma = df['SMA_200'].iloc[-1]
    signal_today = df['Signal'].iloc[-1]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Action for Tomorrow:** {'HOLD LONG' if signal_today == 1 else 'STAY IN CASH'}")
        st.write(f"*Current Price:* ${last_price:.2f}")
        st.write(f"*Critical Threshold:* ${sma:.2f}")
    with col_b:
        st.warning("**Risk Warning:** Output is a probabilistic expectation based on historical data, not a guarantee.")

    st.markdown("---")
    st.markdown("### 🎲 1-Year Monte Carlo Simulation (100 Paths)")
    
    # Generate random walks based on the asset's historical volatility
    daily_vol = df['Return'].std()
    days_ahead = 252 # Trading days in a year
    simulations = 100
    
    # Vectorized Monte Carlo
    random_returns = np.random.normal(0, daily_vol, (days_ahead, simulations))
    price_paths = last_price * np.exp(np.cumsum(random_returns, axis=0))
    
    # Plotting the simulations
    fig_mc = go.Figure()
    for i in range(simulations):
        fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(color='rgba(0, 255, 170, 0.05)'), showlegend=False))
        
    fig_mc.add_trace(go.Scatter(y=np.median(price_paths, axis=1), mode='lines', line=dict(color='white', width=3), name='Median Expected Path'))
    
    fig_mc.update_layout(
        title=f"Probabilistic Forecast for {ticker}",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(title="Trading Days into the Future", showgrid=False),
        yaxis=dict(title="Projected Price ($)", showgrid=True, gridcolor='#333333')
    )
    st.plotly_chart(fig_mc, use_container_width=True)
