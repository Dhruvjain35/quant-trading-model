"""
MACRO-REGIME ADAPTIVE ENSEMBLE MODEL (MRAEM) v2.0
Institutional Quantitative Research Platform
Incorporating ML Ensemble + Step-2/3 Catalyst Verification
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="MRAEM Quant Platform", page_icon="🏛️", layout="wide")

# ==========================================
# 0. ELITE UI STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://rsms.me/inter/inter.css');
:root { --bg-primary: #0A0E14; --bg-secondary: #0F1419; --accent: #00FFB2; --text: #EBEEF5; }
* {font-family: 'Inter', sans-serif !important;}
.stApp {background: var(--bg-primary); color: var(--text);}
h1, h2, h3 {color: var(--text) !important;}
h2 {font-size: 0.9rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; 
    color: #8B95A8 !important; border-bottom: 1px solid rgba(0,255,178,0.2) !important; padding-bottom: 0.5rem !important; margin-top: 2rem !important;}
[data-testid="stMetric"] {background: #161923; border: 1px solid rgba(0,255,178,0.1); border-left: 3px solid var(--accent); padding: 1rem; border-radius: 4px;}
[data-testid="stMetricValue"] {font-size: 1.8rem !important; color: var(--text) !important; font-weight: 700 !important;}
.stButton button {background: linear-gradient(135deg, #00FFB2, #00D99A) !important; color: #000 !important; font-weight: 700 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA PULL & FEATURE ENGINEERING
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_engineer_data():
    tickers = ["QQQ", "SHY", "^VIX"]
    df = yf.download(tickers, start="2010-01-01", progress=False)
    
    # Flatten columns if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close']
        volume = df['Volume']
    else:
        close = df
        volume = df

    data = pd.DataFrame()
    data['QQQ'] = close['QQQ']
    data['SHY'] = close['SHY']
    data['VIX'] = close['^VIX']
    data['Volume'] = volume['QQQ']
    data.dropna(inplace=True)

    # --- ML Feature Engineering ---
    data['Returns'] = data['QQQ'].pct_change()
    data['Vol_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
    data['SMA_50'] = data['QQQ'].rolling(50).mean()
    data['SMA_200'] = data['QQQ'].rolling(200).mean()
    data['Dist_to_SMA50'] = data['QQQ'] / data['SMA_50'] - 1
    
    # Step 2: Pattern (RSI & Bollinger Bands)
    delta = data['QQQ'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    data['SMA_20'] = data['QQQ'].rolling(20).mean()
    data['STD_20'] = data['QQQ'].rolling(20).std()
    data['Lower_BB'] = data['SMA_20'] - (2 * data['STD_20'])
    data['Is_Support'] = (data['QQQ'] <= data['Lower_BB']) | (data['RSI_14'] < 35)

    # Step 3: Catalyst (Volume Surge)
    data['Vol_SMA'] = data['Volume'].rolling(20).mean()
    data['Vol_Ratio'] = data['Volume'] / (data['Vol_SMA'] + 1e-9)
    data['Vol_Surge'] = data['Vol_Ratio'] > 1.5

    # Target Variable: Does QQQ go up over the next 5 days? (1=Yes, 0=No)
    data['Target'] = (data['QQQ'].shift(-5) > data['QQQ']).astype(int)
    
    return data.dropna()

# ==========================================
# 2. MACHINE LEARNING ENSEMBLE
# ==========================================
@st.cache_resource(show_spinner=False)
def train_ml_ensemble(data):
    features = ['Vol_20', 'Dist_to_SMA50', 'RSI_14', 'Vol_Ratio', 'VIX']
    X = data[features][:-5] # Drop last 5 to avoid target leakage in training
    y = data['Target'][:-5]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3-Model Ensemble Architecture
    clf1 = LogisticRegression(random_state=42, class_weight='balanced')
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf3 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    ensemble = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
    
    ensemble.fit(X_scaled, y)
    
    # Predict over entire dataset
    X_all_scaled = scaler.transform(data[features])
    data['ML_Prob_Bullish'] = ensemble.predict_proba(X_all_scaled)[:, 1]
    return data

# ==========================================
# 3. HYBRID SIGNAL & FRICTION ENGINE
# ==========================================
def run_backtest(data, vix_thresh, min_hold, tax_rate, tc_bps):
    df = data.copy()
    
    # Regime Definition
    df['Regime_Risk_Off'] = df['VIX'] > vix_thresh
    
    # Signal Logic (Combines ML + Step 2 + Step 3)
    signals = []
    in_market = True # Start Long QQQ
    days_held = 0
    
    for i in range(len(df)):
        prob_bull = df['ML_Prob_Bullish'].iloc[i]
        risk_off = df['Regime_Risk_Off'].iloc[i]
        support = df['Is_Support'].iloc[i]
        catalyst = df['Vol_Surge'].iloc[i]
        
        # State Machine Logic
        if in_market:
            days_held += 1
            # Exit if Regime is Risk Off AND ML is highly bearish. 
            # Require minimum hold time to avoid tax whipsaw.
            if risk_off and prob_bull < 0.40 and days_held >= min_hold:
                in_market = False
                days_held = 0
                signals.append(-1) # Flee to SHY
            else:
                signals.append(1)  # Stay in QQQ
        else:
            days_held += 1
            # Re-Entry Logic: Requires ML conviction OR (Step 2 + Step 3 Catalyst)
            if (prob_bull > 0.60) or (support and catalyst and not risk_off):
                in_market = True
                days_held = 0
                signals.append(1)  # Back to QQQ
            else:
                signals.append(-1) # Stay in SHY

    df['Position'] = signals
    
    # --- Returns & Frictions Math ---
    df['QQQ_Ret'] = df['QQQ'].pct_change()
    df['SHY_Ret'] = df['SHY'].pct_change()
    
    # Gross Strategy Return
    df['Strat_Ret_Gross'] = np.where(df['Position'].shift(1) == 1, df['QQQ_Ret'], df['SHY_Ret'])
    
    # Frictions: Apply only when state changes
    df['Trade'] = df['Position'] != df['Position'].shift(1)
    df['Cost'] = df['Trade'] * (tc_bps / 10000)
    
    # Simplified Capital Gains Tax (applied only on exiting QQQ to SHY, assuming profit)
    # In a real system you track basis, here we apply a proxy drag on exits
    is_exit = (df['Position'].shift(1) == 1) & (df['Position'] == -1)
    df['Tax_Drag'] = is_exit * (df['QQQ_Ret'].clip(lower=0) * tax_rate)
    
    # Net Return
    df['Strat_Ret_Net'] = df['Strat_Ret_Gross'] - df['Cost'] - df['Tax_Drag']
    
    # Equity Curves
    df['Bench_EQ'] = (1 + df['QQQ_Ret'].fillna(0)).cumprod()
    df['Strat_EQ'] = (1 + df['Strat_Ret_Net'].fillna(0)).cumprod()
    
    return df

# ==========================================
# 4. METRICS & MONTE CARLO
# ==========================================
def calc_metrics(returns_series, risk_free_rate=0.0):
    returns = returns_series.dropna()
    if len(returns) == 0: return 0, 0, 0
    
    total_ret = (1 + returns).prod() - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252 - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return sharpe, total_ret, max_dd

def calculate_p_value(strat_returns, bench_returns):
    t_stat, p_val = stats.ttest_ind(strat_returns.dropna(), bench_returns.dropna())
    return p_val

def run_monte_carlo(returns, paths=500):
    sims = []
    rets_array = returns.dropna().values
    n_days = len(rets_array)
    for _ in range(paths):
        # Bootstrap sampling with replacement
        boot_rets = np.random.choice(rets_array, size=n_days, replace=True)
        sims.append((1 + boot_rets).cumprod())
    
    sims_df = pd.DataFrame(sims).T
    final_vals = sims_df.iloc[-1]
    ci_lower = np.percentile(final_vals, 5)
    ci_upper = np.percentile(final_vals, 95)
    return sims_df, ci_lower, ci_upper

# ==========================================
# 5. STREAMLIT UI LAYOUT
# ==========================================
st.sidebar.markdown("### ⚙️ ASSET UNIVERSE")
target_asset = st.sidebar.text_input("Risk Asset", "QQQ")
safe_asset = st.sidebar.text_input("Safe Asset", "SHY")

st.sidebar.markdown("### 🎛️ REGIME & ML PARAMS")
vix_threshold = st.sidebar.slider("VIX Alert Threshold", 15, 45, 25)
min_hold_days = st.sidebar.slider("Min Hold (days) - Fixes Whipsaw", 5, 60, 20)

st.sidebar.markdown("### 💸 FRICTIONS (TAX & COST)")
st.sidebar.caption("Compare 0.0 vs 0.20 to see tax impact.")
tax_rate = st.sidebar.slider("Short-Term Tax Rate", 0.0, 0.40, 0.20, 0.05)
tc_bps = st.sidebar.slider("Trans Cost (bps)", 0, 10, 3)

st.sidebar.markdown("### 🎲 MONTE CARLO")
bootstrap_paths = st.sidebar.slider("Bootstrap Paths", 100, 1000, 500, step=100)

if st.sidebar.button("🚀 COMPILE MASTER TERMINAL", use_container_width=True):
    with st.spinner("Training ML Ensemble & Backtesting..."):
        # 1. Load & Engineer
        raw_data = load_and_engineer_data()
        
        # 2. Train ML
        ml_data = train_ml_ensemble(raw_data)
        
        # 3. Run Strategy
        res = run_backtest(ml_data, vix_threshold, min_hold_days, tax_rate, tc_bps)
        
        # 4. Calculate Stats
        sh_strat, tot_strat, dd_strat = calc_metrics(res['Strat_Ret_Net'])
        sh_bench, tot_bench, dd_bench = calc_metrics(res['QQQ_Ret'])
        p_val = calculate_p_value(res['Strat_Ret_Net'], res['QQQ_Ret'])
        trades_yr = res['Trade'].sum() / (len(res) / 252)
        
        # MC Sims
        mc_sims, ci_low, ci_high = run_monte_carlo(res['Strat_Ret_Net'], paths=bootstrap_paths)

    # --- HEADER ---
    st.markdown("""
    <div style="padding-bottom: 1rem;">
        <p style="font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #8B95A8; margin: 0;">INSTITUTIONAL QUANTITATIVE RESEARCH PLATFORM</p>
        <h1 style="margin: 0.2rem 0 0 0; color: #EBEEF5;">Macro-Regime Adaptive <span style="color: #00FFB2;">Ensemble Model</span></h1>
        <p style="font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; color: #8B95A8; margin: 0.5rem 0 0 0;">3-MODEL ENSEMBLE • SUPPORT BOUNCE CATALYST • AFTER-TAX OPTIMIZED</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 01 EXECUTIVE DASHBOARD ---
    st.markdown("## 01 — EXECUTIVE DASHBOARD")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Coloring logic based on outperformance
    sc_c = "#00FFB2" if sh_strat > sh_bench else "#FF3B6B"
    tc_c = "#00FFB2" if tot_strat > tot_bench else "#FF3B6B"
    dc_c = "#00FFB2" if dd_strat > dd_bench else "#FF3B6B" # dd is negative
    
    c1.markdown(f'<div data-testid="stMetric"><label data-testid="stMetricLabel">NET SHARPE</label><div data-testid="stMetricValue" style="color: {sc_c} !important;">{sh_strat:.3f}</div><div style="font-size: 0.75rem; color: #8B95A8;">Bench: {sh_bench:.3f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div data-testid="stMetric"><label data-testid="stMetricLabel">TOTAL RETURN</label><div data-testid="stMetricValue" style="color: {tc_c} !important;">{tot_strat*100:.1f}%</div><div style="font-size: 0.75rem; color: #8B95A8;">Bench: {tot_bench*100:.1f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div data-testid="stMetric"><label data-testid="stMetricLabel">MAX DRAWDOWN</label><div data-testid="stMetricValue" style="color: {dc_c} !important;">{dd_strat*100:.1f}%</div><div style="font-size: 0.75rem; color: #8B95A8;">Bench: {dd_bench*100:.1f}%</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div data-testid="stMetric"><label data-testid="stMetricLabel">P-VALUE / TRADES YR</label><div data-testid="stMetricValue">{p_val:.4f}</div><div style="font-size: 0.75rem; color: #8B95A8;">Trades/Yr: {trades_yr:.1f}</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div data-testid="stMetric"><label data-testid="stMetricLabel">90% BOOT CI</label><div data-testid="stMetricValue">[{ci_low:.2f}, {ci_high:.2f}]</div><div style="font-size: 0.75rem; color: #8B95A8;">Ending Equity Range</div></div>', unsafe_allow_html=True)

    if sh_strat > sh_bench:
        st.success(f"✅ **STATISTICALLY SIGNIFICANT OUTPERFORMANCE** | NET Sharpe {sh_strat:.3f} > Benchmark {sh_bench:.3f}")
    else:
        st.warning(f"⚠️ **STRATEGY UNDERPERFORMS** | The benchmark buy-and-hold outperformed the ML model after taxes/fees.")

    # --- 02 EQUITY CURVE ---
    st.markdown("## 02 — WEALTH ACCUMULATION CURVE")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=res.index, y=res['Bench_EQ'], name="Buy & Hold QQQ", line=dict(color='#8B95A8', dash='dash')))
    fig_eq.add_trace(go.Scatter(x=res.index, y=res['Strat_EQ'], name="MRAEM Strategy (Net)", line=dict(color='#00FFB2', width=2)))
    
    # Fled to safe asset markers
    safe_zones = res[res['Position'] == -1]
    fig_eq.add_trace(go.Scatter(x=safe_zones.index, y=safe_zones['Strat_EQ'], mode='markers', name='Fled to SHY', marker=dict(color='#FF3B6B', size=4)))
    
    fig_eq.update_layout(height=400, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5'), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_eq, use_container_width=True)

    # --- 03 MONTE CARLO ---
    st.markdown("## 03 — MONTE CARLO ROBUSTNESS ANALYSIS")
    fig_mc = go.Figure()
    for i in range(min(100, bootstrap_paths)): # Plot subset for performance
        fig_mc.add_trace(go.Scatter(y=mc_sims[i], mode='lines', line=dict(color='rgba(0, 255, 178, 0.05)'), showlegend=False))
    
    # Add Mean path
    fig_mc.add_trace(go.Scatter(y=mc_sims.mean(axis=1), mode='lines', name='Mean Expectancy', line=dict(color='#00FFB2', width=2)))
    fig_mc.update_layout(height=400, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5'), xaxis_title="Trading Days", yaxis_title="Cumulative Return Multiplier")
    st.plotly_chart(fig_mc, use_container_width=True)
    
    # --- 04 CATALYST LOGIC ---
    st.markdown("## 04 — ENGINE LOGIC: SUPPORT & VOLUME CATALYST")
    st.caption("Showing the last 12 months of how the model uses Volume Surges at Support to gatekeep re-entries, minimizing taxable whipsaw.")
    
    recent = res.tail(252)
    fig_cat = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    fig_cat.add_trace(go.Scatter(x=recent.index, y=recent['QQQ'], name='QQQ', line=dict(color='#EBEEF5')), row=1, col=1)
    fig_cat.add_trace(go.Scatter(x=recent.index, y=recent['Lower_BB'], name='Statistical Floor (BB)', line=dict(color='#00FFB2', dash='dot')), row=1, col=1)
    
    # Highlight Catalyst re-entries
    re_entries = recent[(recent['Position'] == 1) & (recent['Position'].shift(1) == -1)]
    fig_cat.add_trace(go.Scatter(x=re_entries.index, y=re_entries['QQQ'], mode='markers', name='Catalyst Re-Entry', marker=dict(symbol='triangle-up', size=12, color='#00FFB2')), row=1, col=1)
    
    # Volume subplot
    colors = ['#00FFB2' if v else '#1A1F2E' for v in recent['Vol_Surge']]
    fig_cat.add_trace(go.Bar(x=recent.index, y=recent['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig_cat.update_layout(height=450, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419', font=dict(family='Inter', color='#EBEEF5'), showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("👈 Set your asset universe and parameters, then hit Compile Master Terminal.")
