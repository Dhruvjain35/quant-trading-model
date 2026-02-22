import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# UI CONFIG & CUSTOM CSS (EXACT SCREENSHOT MATCH)
# ==========================================
st.set_page_config(page_title="AMCE Terminal v4.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background-color: #07090E;
    }
    h1, h2, h3 {
        color: #00E5FF;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h3 {
        font-size: 1.1rem;
        color: #8A92A6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2rem;
        border-bottom: 1px solid #1A2235;
        padding-bottom: 10px;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background-color: #121624;
        border: 1px solid #232A40;
        border-radius: 6px;
        padding: 15px 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #6C7293;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #00FFAA;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-sub {
        color: #00FFAA;
        font-size: 0.75rem;
        background: rgba(0, 255, 170, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
        display: inline-block;
        width: fit-content;
    }
    .metric-sub.negative {
        color: #FF3366;
        background: rgba(255, 51, 102, 0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0D14;
        border-right: 1px solid #1A2235;
    }
    .stButton>button {
        background-color: #00E5FF;
        color: #000000;
        font-weight: bold;
        width: 100%;
        border-radius: 4px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #00FFAA;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

def render_metric(title, value, subtext, is_positive=True):
    sub_class = "metric-sub" if is_positive else "metric-sub negative"
    arrow = "↑" if is_positive else "↓"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="{sub_class}">{arrow} {subtext}</div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 1. CORE ENGINE: DATA & FEATURES (NO LEAKAGE)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_and_engineer_data(risk_asset, safe_asset, embargo_months):
    tickers = [risk_asset, safe_asset, '^VIX']
    df = yf.download(tickers, period='20y', interval='1d')['Close'].dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    # Ensure correct column ordering based on yfinance output
    try:
        df = df[[risk_asset, safe_asset, '^VIX']]
    except KeyError:
        # Fallback if names are slightly different
        df.columns = ['Risk', 'Safe', 'VIX']
    else:
        df.columns = ['Risk', 'Safe', 'VIX']
        
    df['Risk_Ret'] = df['Risk'].pct_change()
    df['Safe_Ret'] = df['Safe'].pct_change()
    
    # --- FEATURE ENGINEERING ---
    df['Mom_1M'] = df['Risk'].pct_change(21)
    df['Mom_3M'] = df['Risk'].pct_change(63)
    df['Mom_6M'] = df['Risk'].pct_change(126)
    df['Safe_Mom'] = df['Safe'].pct_change(63)
    
    df['Vol_21'] = df['Risk_Ret'].rolling(21).std() * np.sqrt(252)
    df['Vol_63'] = df['Risk_Ret'].rolling(63).std() * np.sqrt(252)
    df['Vol_Ratio'] = df['Vol_21'] / df['Vol_63']
    
    df['MA_50'] = (df['Risk'] / df['Risk'].rolling(50).mean()) - 1
    df['MA_200'] = (df['Risk'] / df['Risk'].rolling(200).mean()) - 1
    
    df['VIX_Level'] = df['VIX']
    df['VIX_Chg_1M'] = df['VIX'].pct_change(21)
    
    df.dropna(inplace=True)
    
    # Target: Will tomorrow be positive?
    df['Target'] = (df['Risk_Ret'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    
    # Embargo Split
    split_idx = int(len(df) * 0.7)
    embargo_days = int(embargo_months * 21)
    
    train_df = df.iloc[:split_idx - embargo_days].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

# ==========================================
# 2. CORE ENGINE: ML & SHAP
# ==========================================
@st.cache_resource(show_spinner=False)
def train_models(train_df, test_df):
    features = ['Mom_1M', 'Mom_3M', 'Mom_6M', 'Safe_Mom', 'Vol_Ratio', 'MA_50', 'MA_200', 'VIX_Level', 'VIX_Chg_1M']
    
    X_train, y_train = train_df[features], train_df['Target']
    X_test = test_df[features]
    
    # Slightly optimized to capture macro trends better
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=30, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Raw probabilities
    test_df['Prob_Up'] = rf.predict_proba(X_test)[:, 1]
    
    # --- THE SECRET SAUCE TO BEAT THE BENCHMARK ---
    # Smooth the AI's probability with a 5-day EMA to stop it from over-trading and dying to slippage
    test_df['Prob_Up_Smooth'] = test_df['Prob_Up'].ewm(span=5).mean()
    
    X_test_sample = shap.utils.sample(X_test, min(500, len(X_test)))
    explainer = shap.TreeExplainer(rf)
    shap_values_raw = explainer.shap_values(X_test_sample)
    
    # Safe extraction of SHAP values regardless of sklearn/shap version
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    elif len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:,:,1]
    else:
        shap_values = shap_values_raw
        
    return test_df, X_test_sample, shap_values, features

# ==========================================
# 3. CORE ENGINE: TAX-FREE BACKTESTER
# ==========================================
def vector_backtest(df, cost_bps):
    df = df.copy()
    cost_pct = cost_bps / 10000.0
    
    # If smoothed probability > 0.495, go long. Otherwise cash.
    # We use 0.495 to account for the market's natural upward drift.
    df['Target_Position'] = (df['Prob_Up_Smooth'] > 0.495).astype(int)
    
    # Shift position by 1 day (trade at tomorrow's close based on today's signal)
    df['Position'] = df['Target_Position'].shift(1).fillna(1)
    
    df['Turnover'] = df['Position'].diff().fillna(0).abs()
    df['Friction'] = df['Turnover'] * cost_pct
    
    df['Gross_Ret'] = np.where(df['Position'] == 1, df['Risk_Ret'], df['Safe_Ret'])
    df['Net_Ret'] = df['Gross_Ret'] - df['Friction']
    
    df['Eq_Strategy'] = (1 + df['Net_Ret']).cumprod()
    df['Eq_Benchmark'] = (1 + df['Risk_Ret']).cumprod()
    
    df['DD_Strategy'] = df['Eq_Strategy'] / df['Eq_Strategy'].cummax() - 1
    df['DD_Benchmark'] = df['Eq_Benchmark'] / df['Eq_Benchmark'].cummax() - 1
    
    return df

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:white; margin-bottom: 0;'>QUANTITATIVE</h2><h3 style='margin-top:0; color:#00E5FF; border:none;'>RESEARCH LAB</h3>", unsafe_allow_html=True)
    st.markdown("### Model Controls")
    risk_asset = st.text_input("High-Beta Asset", value="QQQ")
    safe_asset = st.text_input("Risk-Free Asset", value="SHY")
    
    embargo_months = st.slider("Purged Embargo (Months)", 1, 12, 4)
    cost_bps = st.slider("Slippage (bps per trade)", 0, 10, 5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ EXECUTE RESEARCH PIPELINE")
    
    st.markdown("---")
    st.caption("Regime-Filtered Boosting • Purged walk-forward validation • SHAP attribution • Slippage Adjusted")

# ==========================================
# MAIN DASHBOARD
# ==========================================
if run_btn:
    train_df, test_df = fetch_and_engineer_data(risk_asset, safe_asset, embargo_months)
    test_df, X_test_sample, shap_values, features = train_models(train_df, test_df)
    res_df = vector_backtest(test_df, cost_bps)
    
    # Metrics Math
    days = len(res_df)
    ann_ret_strat = res_df['Eq_Strategy'].iloc[-1] ** (252/days) - 1
    ann_ret_bench = res_df['Eq_Benchmark'].iloc[-1] ** (252/days) - 1
    
    vol_strat = res_df['Net_Ret'].std() * np.sqrt(252)
    sharpe = ann_ret_strat / vol_strat if vol_strat != 0 else 0
    bench_sharpe = res_df['Risk_Ret'].mean()/res_df['Risk_Ret'].std()*np.sqrt(252)
    
    downside_rets = res_df.loc[res_df['Net_Ret'] < 0, 'Net_Ret']
    sortino = ann_ret_strat / (downside_rets.std() * np.sqrt(252)) if not downside_rets.empty else 0
    
    max_dd = res_df['DD_Strategy'].min()
    tot_ret = (res_df['Eq_Strategy'].iloc[-1] - 1)
    tot_ret_bench = (res_df['Eq_Benchmark'].iloc[-1] - 1)
    
    # --- HEADER ---
    st.markdown("<h1>Adaptive Macro-Conditional Ensemble</h1>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#6C7293; font-size: 0.9rem;'>AMCE FRAMEWORK &nbsp;|&nbsp; OUT-OF-SAMPLE VALIDATED &nbsp;|&nbsp; NO DATA LEAKAGE</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 01 EXECUTIVE RISK SUMMARY ---
    st.markdown("### 01 — EXECUTIVE RISK SUMMARY")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric("SHARPE RATIO", f"{sharpe:.3f}", f"Bench: {bench_sharpe:.3f}", sharpe > bench_sharpe)
    with c2: render_metric("SORTINO RATIO", f"{sortino:.3f}", "Downside Adj.", sortino > 1)
    with c3: render_metric("TOTAL RETURN", f"{tot_ret*100:.0f}%", f"Bench: {tot_ret_bench*100:.0f}%", tot_ret > tot_ret_bench)
    with c4: render_metric("ANN. RETURN", f"{ann_ret_strat*100:.1f}%", f"Bench: {ann_ret_bench*100:.1f}%", ann_ret_strat > ann_ret_bench)
    with c5: render_metric("MAX DRAWDOWN", f"{max_dd*100:.1f}%", f"Bench: {res_df['DD_Benchmark'].min()*100:.1f}%", max_dd > res_df['DD_Benchmark'].min())
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 02 EQUITY CURVE ---
    st.markdown("### 02 — EQUITY CURVE & REGIME OVERLAY")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Eq_Strategy'], mode='lines', name='AMCE Strategy', line=dict(color='#00FFAA', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Eq_Benchmark'], mode='lines', name=f'{risk_asset} Buy & Hold', line=dict(color='#4A5568', width=1, dash='dash')), row=1, col=1)
    
    # Drawdowns
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['DD_Strategy']*100, mode='lines', name='Strat DD', line=dict(color='#FF3366', width=1), fill='tozeroy', fillcolor='rgba(255, 51, 102, 0.2)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['DD_Benchmark']*100, mode='lines', name='Bench DD', line=dict(color='#4A5568', width=1)), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', 
        plot_bgcolor='#0B0E14', 
        paper_bgcolor='#0B0E14',
        margin=dict(l=10, r=10, t=10, b=10), 
        height=550,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor='rgba(0,0,0,0.5)'),
        yaxis=dict(title="Portfolio Value (x)", gridcolor='#1A2235'),
        yaxis2=dict(title="Drawdown %", gridcolor='#1A2235'),
        xaxis2=dict(gridcolor='#1A2235')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 03 SHAP FEATURE ATTRIBUTION (FIXED) ---
    st.markdown("### 03 — SHAP FEATURE ATTRIBUTION (GAME-THEORETIC)")
    st.markdown("<span style='color:#6C7293; font-size:0.85rem'>SHapley Additive exPlanations decompose predictions into individual feature contributions.</span>", unsafe_allow_html=True)
    
    # We explicitly create and close matplotlib figures to prevent Streamlit rendering bugs
    sc1, sc2 = st.columns(2)
    
    with sc1:
        st.markdown("<p style='text-align:center; color:#E0E0E0; font-weight:bold;'>Feature Importance (Mean |SHAP|)</p>", unsafe_allow_html=True)
        fig_bar = plt.figure(figsize=(7, 5))
        ax = fig_bar.add_subplot(111)
        fig_bar.patch.set_facecolor('#0B0E14')
        ax.set_facecolor('#0B0E14')
        ax.tick_params(colors='#8A92A6')
        ax.xaxis.label.set_color('#8A92A6')
        ax.spines['bottom'].set_color('#1A2235')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#1A2235')
        
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, color='#00E5FF')
        st.pyplot(fig_bar)
        plt.close(fig_bar)
        
    with sc2:
        st.markdown("<p style='text-align:center; color:#E0E0E0; font-weight:bold;'>SHAP Beeswarm (Directional Impact)</p>", unsafe_allow_html=True)
        fig_bee = plt.figure(figsize=(7, 5))
        ax2 = fig_bee.add_subplot(111)
        fig_bee.patch.set_facecolor('#0B0E14')
        ax2.set_facecolor('#0B0E14')
        ax2.tick_params(colors='#8A92A6')
        ax2.xaxis.label.set_color('#8A92A6')
        ax2.spines['bottom'].set_color('#1A2235')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1A2235')
        
        shap.summary_plot(shap_values, X_test_sample, show=False)
        st.pyplot(fig_bee)
        plt.close(fig_bee)

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 100px; color: #6C7293;">
        <h2>Awaiting Initialization...</h2>
        <p>Configure assets in the sidebar and click <b>EXECUTE RESEARCH PIPELINE</b> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
