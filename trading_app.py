import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Systematic Equity Architecture", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS DOM INJECTION (Hedge Fund / Terminal Aesthetic) ---
st.markdown("""
    <style>
    /* Import professional typefaces */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0E17;
        color: #D1D5DB;
    }
    
    /* Typography */
    h1, h2, h3 { color: #FFFFFF; font-weight: 600; letter-spacing: -0.5px; }
    h1 { font-size: 2.2rem; margin-bottom: 0px; padding-bottom: 0px; }
    p.subtitle { color: #8B949E; font-size: 0.95rem; margin-top: 5px; }

    /* Metric Cards */
    .stMetric {
        background-color: #161B22;
        padding: 20px;
        border-radius: 4px;
        border-left: 3px solid #2F81F7;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #E6EDF3;
        font-size: 1.8rem;
    }

    .stMetric [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        color: #8B949E !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.75rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161B22;
        border: 1px solid #30363D;
        border-bottom: none;
        border-radius: 4px 4px 0px 0px;
        color: #8B949E;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] { color: #FFFFFF; border-top: 2px solid #2F81F7; background-color: #0A0E17; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    hr { border-color: #30363D; }
    </style>
""", unsafe_allow_html=True)

# --- 3. HEADER ---
st.markdown("<h1>Systematic Equity Architecture: v4.2</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Accounting for Execution Slippage, Commission Drag, and Short-Term Capital Gains Taxation.</p>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# --- 4. ENGINE & MATHEMATICS ---
@st.cache_data
def load_and_simulate(ticker, start_date):
    # Fetch data and fix yfinance MultiIndex/Timezone errors
    raw_data = yf.download(ticker, start=start_date, progress=False)
    
    # Flatten MultiIndex if it exists (fixes the common yf error)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
        
    df = raw_data[['Close']].copy()
    df.index = df.index.tz_localize(None) # Strip timezone for Plotly compatibility
    
    df['Return'] = df['Close'].pct_change()
    
    # Quantitative Signal: Dual Moving Average Crossover (50d / 200d)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
    
    # Execution Logic
    df['Trade'] = df['Signal'].diff().abs() 
    
    # Real-World Friction Constraints
    SLIPPAGE = 0.0005  
    COMMISSION = 0.0001 
    TAX_RATE = 0.30     
    
    df['Strategy_Gross'] = df['Signal'].shift(1) * df['Return']
    df['Friction_Cost'] = df['Trade'] * (SLIPPAGE + COMMISSION)
    df['Strategy_Net'] = df['Strategy_Gross'] - df['Friction_Cost'].fillna(0)
    df['Equity_Gross'] = (1 + df['Strategy_Gross'].fillna(0)).cumprod()
    
    # Tax Drag Logic (Applied to profitable days)
    df['Strategy_Post_Tax'] = np.where(df['Strategy_Net'] > 0, 
                                       df['Strategy_Net'] * (1 - TAX_RATE), 
                                       df['Strategy_Net'])
    df['Equity_Real'] = (1 + df['Strategy_Post_Tax'].fillna(0)).cumprod()
    
    # Drawdown Calculation
    df['Peak'] = df['Equity_Real'].cummax()
    df['Drawdown'] = (df['Equity_Real'] - df['Peak']) / df['Peak']
    
    return df.dropna()

def calc_metrics(returns):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    neg_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_return / neg_vol if neg_vol != 0 else 0
    return ann_return, ann_vol, sharpe, sortino

# --- 5. SIDEBAR PARAMETERS ---
with st.sidebar:
    st.markdown("<h3 style='color:#FFFFFF; font-size:1.2rem;'>Model Parameters</h3>", unsafe_allow_html=True)
    ticker = st.text_input("Underlying Asset", "SPY")
    start_year = st.number_input("Backtest Start Year", min_value=2000, max_value=2023, value=2010)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; color:#8B949E;'><b>REAL WORLD ASSUMPTIONS:</b><br/>Slippage: 5 bps<br/>Commissions: 1 bps<br/>Tax Rate: 30% STCG</p>", unsafe_allow_html=True)

df = load_and_simulate(ticker, f"{start_year}-01-01")

# --- 6. TABS & UI ROUTING ---
tab_past, tab_future = st.tabs(["HISTORICAL BACKTEST (REALITY)", "FORWARD PROBABILITY FORECAST"])

# ==========================================
# TAB 1: THE PAST (Institutional Analysis)
# ==========================================
with tab_past:
    # Calculate Institutional Metrics
    net_ret, net_vol, sharpe, sortino = calc_metrics(df['Strategy_Post_Tax'])
    max_dd = df['Drawdown'].min()
    calmar = net_ret / abs(max_dd) if max_dd != 0 else 0
    
    total_gross = (df['Equity_Gross'].iloc[-1] - 1) * 100
    total_net = (df['Equity_Real'].iloc[-1] - 1) * 100
    
    # Top Row Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Post-Tax Net Return", f"{total_net:.1f}%", f"{total_gross - total_net:.1f}% Tax/Fee Drag", delta_color="inverse")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-Adjusted")
    c3.metric("Sortino Ratio", f"{sortino:.2f}", "Downside-Adjusted")
    c4.metric("Max Drawdown", f"{max_dd*100:.1f}%", f"Calmar: {calmar:.2f}", delta_color="off")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Advanced Subplot: Price/Equity & Drawdown
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Gross'], name='Academic (Gross)', line=dict(color='#8B949E', dash='dot', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Real'], name='Reality (Net)', line=dict(color='#2F81F7', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name='Drawdown', fill='tozeroy', line=dict(color='#F85149', width=1), fillcolor='rgba(248, 81, 73, 0.2)'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, gridcolor='#30363D')
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1, gridcolor='#30363D')
    fig.update_xaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: THE FUTURE (Monte Carlo)
# ==========================================
with tab_future:
    st.markdown("<h3 style='font-size:1.1rem; color:#8B949E; border-bottom: 1px solid #30363D; padding-bottom:10px;'>TOMORROW'S EXECUTION STATE</h3>", unsafe_allow_html=True)
    
    last_price = df['Close'].iloc[-1]
    signal_today = df['Signal'].iloc[-1]
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Current Asset Price", f"${last_price:.2f}")
    col_b.metric("Algorithmic Stance", "LONG EXPOSURE" if signal_today == 1 else "CASH (DE-RISKED)")
    col_c.metric("Confidence Interval", "95% (1.96σ)", "Based on historical var")

    st.markdown("<br/><h3 style='font-size:1.1rem; color:#8B949E; border-bottom: 1px solid #30363D; padding-bottom:10px;'>252-DAY MONTE CARLO PROBABILITY FORECAST</h3>", unsafe_allow_html=True)
    
    # Vectorized Monte Carlo Math
    daily_vol = df['Return'].std()
    days_ahead = 252 
    simulations = 500
    
    np.random.seed(42) # For reproducible interviews
    random_returns = np.random.normal(0, daily_vol, (days_ahead, simulations))
    price_paths = last_price * np.exp(np.cumsum(random_returns, axis=0))
    
    # Calculate Percentiles
    median_path = np.median(price_paths, axis=1)
    p_95 = np.percentile(price_paths, 95, axis=1)
    p_05 = np.percentile(price_paths, 5, axis=1)
    
    fig_mc = go.Figure()
    
    # Plot Confidence Intervals
    fig_mc.add_trace(go.Scatter(x=list(range(days_ahead)), y=p_95, line=dict(color='rgba(47, 129, 247, 0.2)', width=0), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=list(range(days_ahead)), y=p_05, line=dict(color='rgba(47, 129, 247, 0.2)', width=0), fill='tonexty', fillcolor='rgba(47, 129, 247, 0.1)', name='90% Confidence Interval'))
    
    # Plot Median Path
    fig_mc.add_trace(go.Scatter(x=list(range(days_ahead)), y=median_path, line=dict(color='#2F81F7', width=2), name='Median Expected Path'))
    
    fig_mc.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Trading Days Forward", showgrid=False),
        yaxis=dict(title="Projected Asset Price ($)", gridcolor='#30363D'),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_mc, use_container_width=True)
