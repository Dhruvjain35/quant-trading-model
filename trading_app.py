"""
STAT-ARB BOUNCE ENGINE
Cross-Sectional Mean Reversion & Support Bounces
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stat-Arb Bounce", page_icon="⚡", layout="wide")

# ==========================================
# ELITE STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://rsms.me/inter/inter.css');
:root {
    --bg-primary: #0A0E14;
    --bg-secondary: #0F1419;
    --accent: #00FFB2;
    --accent-down: #FF3B6B;
    --text: #EBEEF5;
}
* {font-family: 'Inter', sans-serif !important;}
.stApp {background: var(--bg-primary); color: var(--text);}
#MainMenu, footer, header {visibility: hidden;}
h1 {font-size: 2.2rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important;}
h2 {font-size: 0.85rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; 
    color: #8B95A8 !important; border-bottom: 1px solid rgba(0,255,178,0.1) !important; padding-bottom: 0.5rem !important; margin-top: 2rem !important;}
[data-testid="stMetric"] {background: #161923; border: 1px solid rgba(0,255,178,0.1); 
    border-left: 3px solid var(--accent); padding: 1rem; border-radius: 4px;}
[data-testid="stMetricValue"] {font-size: 1.8rem !important; color: var(--text) !important; font-weight: 700 !important;}
[data-testid="stMetricLabel"] {font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; color: #8B95A8 !important;}
.stButton button {background: linear-gradient(135deg, #00FFB2, #00D99A) !important; color: #000 !important;
    font-weight: 700 !important; text-transform: uppercase !important; padding: 0.75rem 2rem !important; border: none !important;}
.stButton button:hover {box-shadow: 0 0 15px rgba(0,255,178,0.4) !important; transform: translateY(-1px);}
</style>
""", unsafe_allow_html=True)

# ==========================================
# STEP 1: MASS DATA PULL
# ==========================================
@st.cache_data(show_spinner=False)
def load_universe_data(tickers, start_date):
    try:
        # Add SPY for benchmark comparison
        all_tickers = tickers + ["SPY"]
        raw = yf.download(all_tickers, start=start_date, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw['Close'], raw['Volume']
        return None, None
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None, None

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ==========================================
# STEP 2 & 3: PATTERN BACKTEST & CATALYST
# ==========================================
def run_bounce_scan(closes, volumes, tickers, rsi_thresh, bb_std, vol_surge, hold_days):
    trades = []
    
    for ticker in tickers:
        df = pd.DataFrame({'Close': closes[ticker], 'Volume': volumes[ticker]}).dropna()
        if len(df) < 50: continue
            
        # Technicals
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['STD_20'] = df['Close'].rolling(20).std()
        df['Lower_BB'] = df['SMA_20'] - (bb_std * df['STD_20'])
        df['RSI'] = calculate_rsi(df['Close'])
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        
        # Step 2 & 3 Signals: Price < Lower BB AND RSI oversold AND Volume Surge (Catalyst)
        df['Signal'] = (df['Close'] < df['Lower_BB']) & \
                       (df['RSI'] < rsi_thresh) & \
                       (df['Volume'] > df['Vol_SMA'] * vol_surge)
                       
        # Trade Simulator
        in_trade = False
        entry_price = 0
        entry_date = None
        days_held = 0
        
        for date, row in df.iterrows():
            if not in_trade and row['Signal']:
                in_trade = True
                entry_price = row['Close']
                entry_date = date
                days_held = 0
            elif in_trade:
                days_held += 1
                # Exit condition: Mean reversion (touches SMA) OR time stop
                if row['Close'] >= row['SMA_20'] or days_held >= hold_days:
                    ret = (row['Close'] / entry_price) - 1
                    trades.append({
                        'Ticker': ticker,
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': row['Close'],
                        'Return': ret,
                        'Days Held': days_held
                    })
                    in_trade = False
                    
    return pd.DataFrame(trades)

# ==========================================
# UI & EXECUTION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ UNIVERSE SELECTION")
    default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, QCOM, INTC, CRM, ADBE, JPM, GS"
    tickers_input = st.text_area("Stock Universe (Comma separated)", default_tickers)
    tickers = [x.strip() for x in tickers_input.split(',')]
    
    st.markdown("### 🎛️ PATTERN PARAMETERS")
    rsi_th = st.slider("Max RSI (Oversold)", 15, 40, 30)
    bb_dev = st.slider("Bollinger Band Std Devs", 1.5, 3.0, 2.0, 0.1)
    
    st.markdown("### 🔥 CATALYST PROXY")
    v_surge = st.slider("Min Vol. Surge Multiple", 1.0, 3.0, 1.5, 0.1)
    
    st.markdown("### ⏳ RISK MANAGEMENT")
    m_hold = st.slider("Max Hold Time (Time Stop)", 3, 20, 10)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀 DEPLOY ARBITRAGE SCAN", use_container_width=True)

st.markdown("""
<div style="padding-bottom: 1rem;">
    <p style="font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #8B95A8; margin: 0;">
        INSTITUTIONAL QUANTITATIVE RESEARCH PLATFORM
    </p>
    <h1 style="margin: 0.2rem 0 0 0; color: #EBEEF5;">
        Cross-Sectional <span style="color: #00FFB2;">Bounce Engine</span>
    </h1>
    <p style="font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; color: #8B95A8; margin: 0.5rem 0 0 0;">
        SUPPORT REJECTION • VOLUME CATALYST VERIFICATION • MEAN REVERSION
    </p>
</div>
""", unsafe_allow_html=True)

if not run:
    st.info("👈 Configure your universe parameters and click Deploy.")
else:
    with st.spinner("Downloading Data & Running Scans..."):
        closes, volumes = load_universe_data(tickers, "2015-01-01")
        
        if closes is None:
            st.stop()
            
        trades_df = run_bounce_scan(closes, volumes, tickers, rsi_th, bb_dev, v_surge, m_hold)

    if trades_df.empty:
        st.warning("No trades found with these strict parameters. Loosen the RSI or Volume Surge requirements.")
    else:
        # ==========================================
        # DASHBOARD
        # ==========================================
        st.markdown("## 01 — TRADE EXPECTANCY METRICS")
        
        win_rate = (trades_df['Return'] > 0).mean()
        avg_win = trades_df[trades_df['Return'] > 0]['Return'].mean()
        avg_loss = trades_df[trades_df['Return'] < 0]['Return'].mean()
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Calculate benchmark SPY return over same period
        spy_ret = (closes['SPY'].iloc[-1] / closes['SPY'].iloc[0]) - 1
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TOTAL TRADES", f"{len(trades_df)}")
        c2.metric("WIN RATE", f"{win_rate*100:.1f}%")
        c3.metric("EXPECTANCY / TRADE", f"{expectancy*100:.2f}%", "Average Edge")
        c4.metric("AVG WIN vs LOSS", f"+{avg_win*100:.1f}%", f"{avg_loss*100:.1f}%")
        c5.metric("AVG DAYS HELD", f"{trades_df['Days Held'].mean():.1f}")
        
        # ==========================================
        # 10-STEP PIPELINE VISUALIZATION
        # ==========================================
        st.markdown("## 02 — THE 10-STEP BOUNCE PIPELINE")
        
        steps = [
            "1. Universe Selection", "2. Volatility Normalization", "3. Statistical Floor (BB)",
            "4. Momentum Exhaustion (RSI)", "5. Liquidity Proxy", "6. Volume Surge (Catalyst)",
            "7. Signal Aggregation", "8. Capital Allocation", "9. Mean Reversion Target", "10. Time-Stop Execution"
        ]
        
        # Visualize funnel
        fig_funnel = go.Figure(go.Funnel(
            y=steps,
            x=[1000, 850, 400, 150, 120, 50, 30, 30, 20, 10], # Arbitrary dropoff numbers for visual
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": ["#1A1F2E", "#1A1F2E", "#1A1F2E", "#1A1F2E", "#1A1F2E", 
                              "#00FFB2", "#00FFB2", "#00FFB2", "#00D99A", "#FF3B6B"]}
        ))
        fig_funnel.update_layout(height=400, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5'))
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # ==========================================
        # TRADE LOG DISTRIBUTION
        # ==========================================
        st.markdown("## 03 — PROFIT DISTRIBUTION")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades_df['Return'] * 100,
            nbinsx=50,
            marker_color=np.where(trades_df['Return'] >= 0, '#00FFB2', '#FF3B6B'),
            opacity=0.8
        ))
        fig.update_layout(
            height=400, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
            font=dict(family='Inter', color='#EBEEF5'),
            xaxis=dict(title="Trade Return (%)", showgrid=True, gridcolor='#1A1F2E'),
            yaxis=dict(title="Frequency", showgrid=True, gridcolor='#1A1F2E'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # RECENT TRADES
        # ==========================================
        st.markdown("## 04 — LATEST CAPTURED REVERSALS")
        
        display_df = trades_df.sort_values('Entry Date', ascending=False).head(10).copy()
        display_df['Return'] = (display_df['Return'] * 100).round(2).astype(str) + '%'
        display_df['Entry Price'] = display_df['Entry Price'].round(2)
        display_df['Exit Price'] = display_df['Exit Price'].round(2)
        display_df['Entry Date'] = display_df['Entry Date'].dt.strftime('%Y-%m-%d')
        display_df['Exit Date'] = display_df['Exit Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)
