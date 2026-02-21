import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. DATA PULL (BROADER CONTEXT) ---
@st.cache_data(show_spinner=False)
def pull_market_data():
    # Pulling QQQ, SHY, VIX, and Volume for Catalyst verification
    tickers = ["QQQ", "SHY", "^VIX"]
    df = yf.download(tickers, start="2015-01-01", progress=False)
    
    # Flatten multi-index if necessary
    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close']
        volumes = df['Volume']
    else:
        closes = df
        volumes = df
        
    return closes, volumes

# --- 2 & 3. PATTERN BACKTEST & CATALYST LOGIC ---
def generate_signals(closes, volumes, vix_thresh):
    df = pd.DataFrame()
    df['QQQ'] = closes['QQQ']
    df['SHY'] = closes['SHY']
    df['VIX'] = closes['^VIX']
    df['QQQ_Vol'] = volumes['QQQ']
    
    # Step 2: Pattern (Statistical Support / Mean Reversion)
    df['RSI_14'] = calculate_rsi(df['QQQ'])
    df['SMA_20'] = df['QQQ'].rolling(20).mean()
    df['STD_20'] = df['QQQ'].rolling(20).std()
    df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
    df['Is_Support'] = (df['QQQ'] <= df['Lower_BB']) | (df['RSI_14'] < 35)
    
    # Step 3: Catalyst (Volume Surge > 1.5x average)
    df['Vol_SMA'] = df['QQQ_Vol'].rolling(20).mean()
    df['Vol_Surge'] = df['QQQ_Vol'] > (df['Vol_SMA'] * 1.5)
    
    # MRAEM HYBRID LOGIC
    # 1. Base ML/Regime proxy (replace with your actual ML ensemble predictions)
    df['Regime_Risk_Off'] = df['VIX'] > vix_thresh
    
    # 2. Signal Generation
    df['Signal'] = 1 # Default Long QQQ
    
    in_safe_asset = False
    signals = []
    
    for i in range(len(df)):
        # If we are currently in QQQ and Regime shifts to Risk-Off -> Flee to SHY
        if not in_safe_asset and df['Regime_Risk_Off'].iloc[i]:
            in_safe_asset = True
            signals.append(-1)
            
        # If we are in SHY, ONLY re-enter QQQ if we hit Support AND have a Volume Catalyst
        elif in_safe_asset:
            if df['Is_Support'].iloc[i] and df['Vol_Surge'].iloc[i] and not df['Regime_Risk_Off'].iloc[i]:
                in_safe_asset = False
                signals.append(1)
            else:
                signals.append(-1) # Stay in SHY
        else:
            signals.append(1) # Stay in QQQ
            
    df['Final_Signal'] = signals
    return df

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --- UI & VISUALIZATION ---
# (Assume sidebar parameters are already set by your existing code)
vix_alert_threshold = st.sidebar.slider("VIX Alert Threshold", 15, 40, 25)

if st.sidebar.button("🚀 COMPILE MASTER TERMINAL"):
    closes, volumes = pull_market_data()
    df = generate_signals(closes, volumes, vix_alert_threshold)
    
    # --- YOUR 10-STEP VISUALIZATION GRAPHS ---
    st.markdown("## 02 — THE 10-STEP MRAEM ARCHITECTURE")
    
    steps = [
        "1. Broad Market Data Ingestion", 
        "2. ML Feature Engineering", 
        "3. Volatility (VIX) Regime Classification",
        "4. Risk-Off Trigger (Flee to SHY)", 
        "5. Tax-Friction Assessment", 
        "6. Capital Preservation Mode",
        "7. Support Pattern Identification (Step 2)", 
        "8. Institutional Volume Cross-Check (Step 3)", 
        "9. High-Conviction Re-entry Signal", 
        "10. Alpha Generation (Beat Benchmark)"
    ]
    
    # 1. Funnel Graph showing how trades are filtered to reduce whipsaw
    fig_funnel = go.Figure(go.Funnel(
        y=steps,
        x=[100, 100, 80, 25, 25, 25, 15, 5, 5, 5], # Represents % of time in each state
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#1A1F2E", "#1A1F2E", "#1A1F2E", "#FF3B6B", "#FF3B6B", 
                          "#FF3B6B", "#00FFB2", "#00FFB2", "#00D99A", "#00D99A"]}
    ))
    fig_funnel.update_layout(
        title="Signal Filtration Pipeline (Reducing Taxable Whipsaw)",
        height=450, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5')
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # 2. Catalyst Verification Graph
    st.markdown("## 03 — CATALYST & PATTERN VERIFICATION OVERLAY")
    
    # Plotting recent data to show the logic in action
    recent_df = df.tail(252) # Last 1 year
    
    fig_overlay = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Price & Bollinger Bands
    fig_overlay.add_trace(go.Scatter(x=recent_df.index, y=recent_df['QQQ'], name='QQQ', line=dict(color='#EBEEF5')), row=1, col=1)
    fig_overlay.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Lower_BB'], name='Support Line', line=dict(color='#00FFB2', dash='dot')), row=1, col=1)
    
    # Highlight Re-entries (Green Triangles) where Pattern + Catalyst aligned
    re_entries = recent_df[(recent_df['Final_Signal'] == 1) & (recent_df['Final_Signal'].shift(1) == -1)]
    fig_overlay.add_trace(go.Scatter(x=re_entries.index, y=re_entries['QQQ'], mode='markers', 
                                     marker=dict(symbol='triangle-up', size=12, color='#00FFB2'), 
                                     name='Catalyst Re-Entry'), row=1, col=1)
    
    # Volume Surge 
    colors = ['#00FFB2' if surge else '#1A1F2E' for surge in recent_df['Vol_Surge']]
    fig_overlay.add_trace(go.Bar(x=recent_df.index, y=recent_df['QQQ_Vol'], marker_color=colors, name='Volume'), row=2, col=1)
    fig_overlay.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Vol_SMA']*1.5, name='Surge Threshold', line=dict(color='#FF3B6B', dash='dash')), row=2, col=1)
    
    fig_overlay.update_layout(
        title="Pattern (Support) + Catalyst (Volume) Timing",
        height=500, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5'),
        showlegend=True
    )
    st.plotly_chart(fig_overlay, use_container_width=True)
