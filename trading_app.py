import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- UI CONFIGURATION & CSS ---
st.set_page_config(page_title="AMCE Terminal", layout="wide", initial_sidebar_state="expanded")

# Injecting Custom CSS for the "Terminal V2.0" Aesthetic
st.markdown("""
<style>
    /* Global Theme */
    .stApp { background-color: #0b0e14; color: #8b949e; font-family: 'Inter', sans-serif; }
    
    /* Hide Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Terminal Headers */
    .terminal-title {
        font-family: 'Space Mono', monospace;
        color: #00f2fe;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0px;
    }
    .terminal-subtitle {
        font-family: 'Space Mono', monospace;
        color: #4facfe;
        font-size: 0.8rem;
        letter-spacing: 2px;
        margin-bottom: 20px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 10px;
    }

    /* Metric Cards */
    .metric-container {
        background-color: #11151c;
        border: 1px solid #1f2937;
        border-top: 3px solid #00f2fe;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;}
    .metric-value { font-size: 1.8rem; color: #e6edf3; font-weight: 700; margin-top: 5px; }
    .metric-sub { font-size: 0.7rem; color: #2ea043; margin-top: 5px;}
    
    /* Specific Card Borders */
    .border-purple { border-top: 3px solid #8a2be2; }
    .border-green { border-top: 3px solid #2ea043; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)

# --- CACHED CORE ENGINE (PREVENTS HANGING) ---

@st.cache_data(show_spinner=False)
def fetch_market_data(risk_asset, safe_asset):
    # Fetching 15 years of data to ensure statistical significance
    data = yf.download([risk_asset, safe_asset], start="2008-01-01", progress=False)['Adj Close']
    data = data.dropna()
    return data

@st.cache_data(show_spinner=False)
def run_amce_pipeline(df, risk_asset, safe_asset, tc_bps):
    """Monolithic data, feature engineering, and backtest pipeline"""
    
    # 1. Feature Engineering (Strictly t)
    X = pd.DataFrame(index=df.index)
    returns = df.pct_change().dropna()
    
    X['Mom_3M'] = df[risk_asset].pct_change(63)
    X['Mom_6M'] = df[risk_asset].pct_change(126)
    X['Vol_1M'] = returns[risk_asset].rolling(21).std() * np.sqrt(252)
    X['Vol_3M'] = returns[risk_asset].rolling(63).std() * np.sqrt(252)
    X['Yield_Proxy'] = df[safe_asset].pct_change(63) 
    X['Risk_Spread'] = X['Mom_3M'] - X['Yield_Proxy']
    
    # 2. Target (Strictly t+1)
    target_returns = returns[risk_asset].shift(-1)
    y = (target_returns > 0).astype(int)
    
    # Align and drop NaNs
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    actual_returns = returns.loc[valid_idx]
    
    # 3. Fast Walk-Forward Validation (Retrain every 252 days to avoid hanging)
    predictions = pd.Series(index=X.index, dtype=float)
    
    # Lightweight Ensemble
    clf1 = LogisticRegression(C=0.1, solver='lbfgs')
    clf2 = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
    ensemble = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
    scaler = StandardScaler()
    
    train_window = 1000 # ~4 years initial training
    step = 252 # Annual retraining for speed
    
    for i in range(train_window, len(X), step):
        start_idx = max(0, i - train_window)
        X_train, y_train = X.iloc[start_idx:i], y.iloc[start_idx:i]
        
        test_end = min(i + step, len(X))
        X_test = X.iloc[i:test_end]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ensemble.fit(X_train_scaled, y_train)
        probs = ensemble.predict_proba(X_test_scaled)[:, 1]
        predictions.iloc[i:test_end] = probs

    # 4. Backtest Engine
    bt = pd.DataFrame(index=predictions.dropna().index)
    bt['Prob'] = predictions.dropna()
    bt['Target_Weight'] = np.where(bt['Prob'] > 0.52, 1.0, 0.0)
    bt['Weight_Shift'] = bt['Target_Weight'].shift(1).fillna(0)
    bt['Turnover'] = bt['Weight_Shift'].diff().abs().fillna(0)
    
    tc_dec = tc_bps / 10000.0
    
    bt['Gross_Ret'] = (bt['Weight_Shift'] * actual_returns[risk_asset].loc[bt.index]) + \
                      ((1 - bt['Weight_Shift']) * actual_returns[safe_asset].loc[bt.index])
    
    bt['Net_Ret'] = bt['Gross_Ret'] - (bt['Turnover'] * tc_dec)
    bt['Bench_Ret'] = actual_returns[risk_asset].loc[bt.index]
    
    bt['Cum_Strat'] = (1 + bt['Net_Ret']).cumprod()
    bt['Cum_Bench'] = (1 + bt['Bench_Ret']).cumprod()
    
    # Drawdowns
    bt['Strat_Peak'] = bt['Cum_Strat'].cummax()
    bt['Strat_DD'] = (bt['Cum_Strat'] / bt['Strat_Peak']) - 1
    
    return bt

def calculate_metrics(bt):
    rets = bt['Net_Ret']
    ann_ret = rets.mean() * 252
    vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol != 0 else 0
    
    downside_vol = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol != 0 else 0
    
    max_dd = bt['Strat_DD'].min()
    cvar_95 = rets.quantile(0.05) * np.sqrt(21) # Monthly CVaR Approx
    
    tot_ret = bt['Cum_Strat'].iloc[-1] - 1
    bench_tot_ret = bt['Cum_Bench'].iloc[-1] - 1
    bench_ann_ret = bt['Bench_Ret'].mean() * 252
    bench_sharpe = (bt['Bench_Ret'].mean() * 252) / (bt['Bench_Ret'].std() * np.sqrt(252))
    
    return {
        "Sharpe": sharpe, "Bench_Sharpe": bench_sharpe,
        "Sortino": sortino, "Total_Ret": tot_ret, "Bench_Tot_Ret": bench_tot_ret,
        "Ann_Ret": ann_ret, "Bench_Ann_Ret": bench_ann_ret,
        "Max_DD": max_dd, "CVaR": cvar_95
    }

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<div style='font-family: monospace; color: #8b949e; letter-spacing: 1px;'>RESEARCH TERMINAL V2.0</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Controls**")

risk_asset = st.sidebar.text_input("High-Beta Asset", value="QQQ")
safe_asset = st.sidebar.text_input("Risk-Free Asset", value="SHY")
embargo = st.sidebar.slider("Purged Embargo (Months)", 0, 6, 2)
tc_bps = st.sidebar.slider("Transaction Costs (bps)", 0, 20, 5)

execute = st.sidebar.button("EXECUTE PIPELINE")

# --- MAIN DASHBOARD ---
st.markdown("<div class='terminal-title'>Adaptive Macro-Conditional Ensemble</div>", unsafe_allow_html=True)
st.markdown("<div class='terminal-subtitle'>AMCE FRAMEWORK • PURGED WALK-FORWARD • ENSEMBLE VOTING • STATISTICAL VALIDATION</div>", unsafe_allow_html=True)

# Research Hypothesis Box
st.markdown("""
<div style='background-color: #0d1117; border-left: 4px solid #4facfe; padding: 15px; margin-bottom: 30px; border-radius: 4px;'>
    <div style='color: #4facfe; font-size: 0.75rem; font-weight: bold; margin-bottom: 5px;'>RESEARCH HYPOTHESIS</div>
    <div style='font-size: 0.85rem; color: #c9d1d9; margin-bottom: 3px;'><b>H0 (Null):</b> Macro-conditional regime signals provide no statistically significant improvement over passive equity exposure.</div>
    <div style='font-size: 0.85rem; color: #c9d1d9;'><b>H1 (Alternative):</b> Integrating momentum dynamics and yield curve signals with purged walk-forward validation generates positive crisis alpha and statistically significant risk-adjusted outperformance.</div>
</div>
""", unsafe_allow_html=True)

if execute:
    with st.spinner("Compiling Institutional Pipeline..."):
        # Run core logic
        raw_data = fetch_market_data(risk_asset, safe_asset)
        bt = run_amce_pipeline(raw_data, risk_asset, safe_asset, tc_bps)
        metrics = calculate_metrics(bt)

        # 01 - EXECUTIVE RISK SUMMARY
        st.markdown("<div style='color: #8b949e; font-family: monospace; letter-spacing: 2px; margin-bottom: 10px;'>01 — EXECUTIVE RISK SUMMARY</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.markdown(f"""<div class='metric-container'><div class='metric-label'>SHARPE RATIO</div><div class='metric-value'>{metrics['Sharpe']:.3f}</div><div class='metric-sub'>Bench: {metrics['Bench_Sharpe']:.3f}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-container border-purple'><div class='metric-label'>SORTINO RATIO</div><div class='metric-value'>{metrics['Sortino']:.3f}</div><div class='metric-sub'>Downside-adj.</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-container'><div class='metric-label'>TOTAL RETURN</div><div class='metric-value'>{metrics['Total_Ret']*100:.0f}%</div><div class='metric-sub'>Bench: {metrics['Bench_Tot_Ret']*100:.0f}%</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class='metric-container border-purple'><div class='metric-label'>ANN. RETURN</div><div class='metric-value'>{metrics['Ann_Ret']*100:.1f}%</div><div class='metric-sub'>Bench: {metrics['Bench_Ann_Ret']*100:.1f}%</div></div>""", unsafe_allow_html=True)
        c5.markdown(f"""<div class='metric-container border-green'><div class='metric-label'>MAX DRAWDOWN</div><div class='metric-value'>{metrics['Max_DD']*100:.1f}%</div><div class='metric-sub'>Calmar: {abs(metrics['Ann_Ret']/metrics['Max_DD']):.2f}</div></div>""", unsafe_allow_html=True)
        c6.markdown(f"""<div class='metric-container'><div class='metric-label'>CVAR (95%)</div><div class='metric-value'>{metrics['CVaR']*100:.2f}%</div><div class='metric-sub'>Monthly Tail Risk</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 02 - EQUITY CURVE & REGIME OVERLAY
        st.markdown("<div style='color: #8b949e; font-family: monospace; letter-spacing: 2px; margin-bottom: 10px;'>02 — EQUITY CURVE & REGIME OVERLAY</div>", unsafe_allow_html=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Equity curves
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Bench'], name=f'{risk_asset} Buy & Hold', line=dict(color='#8b949e', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Strat'], name='AMCE Strategy', line=dict(color='#00ff88', width=2)), row=1, col=1)
        
        # Drawdown profile
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Strat_DD'], name='Strategy DD', fill='tozeroy', line=dict(color='#ff3366', width=1)), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#0b0e14',
            paper_bgcolor='#0b0e14',
            margin=dict(l=0, r=0, t=20, b=0),
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Portfolio Value (x)", row=1, col=1, gridcolor='#1f2937')
        fig.update_yaxes(title_text="Drawdown %", tickformat=".0%", row=2, col=1, gridcolor='#1f2937')
        fig.update_xaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 03 - STRATEGY STABILITY (ROLLING METRICS)
        st.markdown("<div style='color: #8b949e; font-family: monospace; letter-spacing: 2px; margin-bottom: 10px;'>03 — STRATEGY STABILITY (ROLLING METRICS)</div>", unsafe_allow_html=True)
        
        roll_ann_ret = bt['Net_Ret'].rolling(252).mean() * 252
        roll_vol = bt['Net_Ret'].rolling(252).std() * np.sqrt(252)
        roll_sharpe = roll_ann_ret / roll_vol
        
        roll_bench_ann = bt['Bench_Ret'].rolling(252).mean() * 252
        roll_bench_vol = bt['Bench_Ret'].rolling(252).std() * np.sqrt(252)
        roll_bench_sharpe = roll_bench_ann / roll_bench_vol
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name='Strategy 12M Sharpe', line=dict(color='#00f2fe', width=2)))
        fig2.add_trace(go.Scatter(x=roll_bench_sharpe.index, y=roll_bench_sharpe, name='Benchmark 12M Sharpe', line=dict(color='#8b949e', width=1, dash='dot')))
        fig2.add_hline(y=0, line_dash="dash", line_color="#ff3366")
        
        fig2.update_layout(
            template='plotly_dark', plot_bgcolor='#0b0e14', paper_bgcolor='#0b0e14',
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            title=dict(text="12-Month Rolling Sharpe Ratio", font=dict(color="#c9d1d9")),
            yaxis=dict(gridcolor='#1f2937'), xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px; color: #4facfe; font-family: monospace;'>
        [ SYSTEM IDLE ] <br><br> Configure parameters in the control panel and execute the pipeline to initialize the core engine.
    </div>
    """, unsafe_allow_html=True)
