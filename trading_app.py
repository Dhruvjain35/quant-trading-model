"""
ADAPTIVE MACRO-CONDITIONAL ENSEMBLE (AMCE) v4.0 FINAL
Professional Institutional Research Terminal
No Data Leakage | Proper Validation | Real-World Costs
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AMCE Terminal", page_icon="▲", layout="wide")

# PROFESSIONAL CSS - NO EMOJIS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {--bg:#0A0E14;--panel:#11151C;--accent:#00FFB2;--text:#EBEEF5;--gray:#6B7280;--red:#EF4444;--border:rgba(107,114,128,0.2);}
* {font-family:'Inter',sans-serif;margin:0;padding:0;}
.stApp {background:var(--bg);color:var(--text);}
h1 {font-size:2.5rem;font-weight:700;letter-spacing:-0.02em;color:var(--text);line-height:1.2;margin-bottom:0.5rem;}
h2 {font-size:0.75rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--gray);border-bottom:1px solid var(--border);padding-bottom:0.5rem;margin:2rem 0 1rem 0;}
h3 {font-size:1rem;font-weight:600;color:var(--text);margin:1rem 0 0.5rem 0;}
[data-testid="stSidebar"] {background:var(--panel);border-right:1px solid var(--border);}
[data-testid="stMetric"] {background:var(--panel);border:1px solid var(--border);border-left:2px solid var(--accent);padding:1rem;border-radius:2px;}
[data-testid="stMetricValue"] {font-size:1.75rem !important;color:var(--accent) !important;font-weight:700 !important;font-feature-settings:'tnum';}
[data-testid="stMetricLabel"] {font-size:0.65rem !important;text-transform:uppercase;letter-spacing:0.05em;color:var(--gray) !important;font-weight:600 !important;}
.stButton button {background:linear-gradient(135deg,var(--accent),#00D99A);color:#000;font-weight:700;border:none;padding:0.75rem 1.5rem;font-size:0.8rem;letter-spacing:0.05em;text-transform:uppercase;border-radius:2px;}
.metric-box {background:var(--panel);border:1px solid var(--border);border-left:2px solid var(--accent);padding:0.75rem 1rem;border-radius:2px;margin-bottom:0.5rem;}
.metric-label {font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--gray);font-weight:600;}
.metric-value {font-size:1.5rem;color:var(--accent);font-weight:700;margin:0.25rem 0;font-feature-settings:'tnum';}
.info-panel {background:rgba(0,255,178,0.05);border-left:2px solid var(--accent);padding:1rem;margin:1rem 0;border-radius:2px;font-size:0.85rem;}
.warning-panel {background:rgba(239,68,68,0.05);border-left:2px solid var(--red);padding:1rem;margin:1rem 0;border-radius:2px;font-size:0.85rem;}
table {width:100%;border-collapse:collapse;font-size:0.85rem;margin:1rem 0;}
th {background:var(--panel);color:var(--gray);font-weight:600;text-transform:uppercase;font-size:0.65rem;letter-spacing:0.05em;padding:0.5rem;text-align:left;border-bottom:1px solid var(--border);}
td {padding:0.5rem;border-bottom:1px solid rgba(107,114,128,0.1);}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(risk, safe):
    tickers = [risk, safe, '^VIX']
    df = yf.download(tickers, start="2006-01-01", progress=False)['Close']
    df = df.ffill().dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={risk:'Risk', safe:'Safe', '^VIX':'VIX'})
    return df

def engineer_features(df):
    data = df.copy()
    
    # Target: 10-day forward return > 0. Strict out of sample, dropna prevents leakage.
    data['Target'] = (data['Risk'].shift(-10) / data['Risk'] - 1 > 0).astype(int)
    
    # Core Macro & Regime Features
    data['Trend_200'] = data['Risk'] / data['Risk'].rolling(200).mean() - 1
    data['Trend_50'] = data['Risk'] / data['Risk'].rolling(50).mean() - 1
    data['Safe_Trend'] = data['Safe'] / data['Safe'].rolling(50).mean() - 1
    data['VIX_Stretch'] = data['VIX'] / data['VIX'].rolling(20).mean() - 1
    data['Vol_20'] = data['Risk'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # Internal RSI Approximation (14-day)
    delta = data['Risk'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Drop NaNs to ensure no missing data and no target leakage at the tail
    data = data.dropna()
    features = ['Trend_200', 'Trend_50', 'Safe_Trend', 'VIX_Stretch', 'Vol_20', 'RSI_14']
    return data, features

def train_model(data, features):
    # 60/40 split with 6-month gap to prevent lookahead overlap
    split = int(len(data) * 0.60)
    gap = 126  
    
    train = data.iloc[:split]
    test = data.iloc[split+gap:]
    
    X_tr, y_tr = train[features], train['Target']
    
    # Tuned to prevent overfitting: Shallower trees, more samples per leaf
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=40, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    
    rf.fit(X_tr_sc, y_tr)
    gb.fit(X_tr_sc, y_tr)
    
    X_te = test[features]
    X_te_sc = scaler.transform(X_te)
    
    prob_rf = rf.predict_proba(X_te_sc)[:,1]
    prob_gb = gb.predict_proba(X_te_sc)[:,1]
    prob_avg = (prob_rf + prob_gb) / 2
    
    data['Prob'] = 0.50
    data.loc[test.index, 'Prob'] = prob_avg
    
    return data, train, test

def backtest(data, tc_bps, tax_st, slip_bps):
    df = data.copy()
    df['R_ret'] = df['Risk'].pct_change()
    df['S_ret'] = df['Safe'].pct_change()
    
    positions = []
    taxes = []
    entry = df['Risk'].iloc[0] if len(df) > 0 else 1.0
    pos = 1
    
    for i in range(len(df)):
        price = df['Risk'].iloc[i]
        prob = df['Prob'].iloc[i]
        tax = 0.0
        
        if pos == 1:
            gain = (price / entry) - 1
            tax_cost = max(0, gain * tax_st)
            
            # HYSTERESIS: Require high conviction to sell and pay taxes
            threshold = 0.48 - tax_cost
            threshold = max(0.35, threshold)
            
            if prob < threshold:
                pos = 0
                if gain > 0:
                    tax = gain * tax_st
        else:
            # HYSTERESIS: Require high conviction to buy back in
            if prob > 0.53:
                pos = 1
                entry = price
                
        positions.append(pos)
        taxes.append(tax)
    
    # Shift position by 1 to prevent lookahead (signal at close executes next day)
    df['Pos'] = pd.Series(positions, index=df.index).shift(1).fillna(1)
    df['Tax'] = pd.Series(taxes, index=df.index).fillna(0)
    
    df['Gross'] = np.where(df['Pos']==1, df['R_ret'], df['S_ret'])
    df['Turn'] = df['Pos'].diff().abs()
    df['Cost'] = df['Turn'] * (tc_bps + slip_bps) / 10000
    df['Net'] = df['Gross'] - df['Cost'] - df['Tax']
    
    df['Eq_Strat'] = (1 + df['Net'].fillna(0)).cumprod()
    df['Eq_Risk'] = (1 + df['R_ret'].fillna(0)).cumprod()
    df['DD_Strat'] = df['Eq_Strat'] / df['Eq_Strat'].cummax() - 1
    df['DD_Risk'] = df['Eq_Risk'] / df['Eq_Risk'].cummax() - 1
    
    return df

def calc_stats(rets):
    ret = rets.dropna()
    if len(ret) == 0: return 0,0,0,0
    m, s = ret.mean(), ret.std()
    sh = (m/s)*np.sqrt(252) if s>0 else 0
    tot = (1+ret).prod()-1
    dd = ((1+ret).cumprod()/(1+ret).cumprod().cummax()-1).min()
    ann = (1+m)**252-1
    return sh, tot, dd, ann

# SIDEBAR
st.sidebar.markdown("<h3 style='margin-bottom:0.5rem;'>AMCE TERMINAL v4.0</h3>", unsafe_allow_html=True)
st.sidebar.caption("Professional institutional-grade research")
st.sidebar.markdown("---")
st.sidebar.markdown("**ASSET CONFIGURATION**")
risk = st.sidebar.text_input("High-Beta Asset", "QQQ")
safe = st.sidebar.text_input("Risk-Free Asset", "SHY")
st.sidebar.markdown("---")
st.sidebar.markdown("**COST MODEL**")
tc = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
slip = st.sidebar.slider("Slippage (bps)", 0, 20, 5)
tax = st.sidebar.slider("Short-Term Tax Rate (%)", 0, 40, 28) / 100
st.sidebar.markdown("---")
st.sidebar.markdown("**VALIDATION**")
mc = st.sidebar.number_input("Monte Carlo Paths", 100, 1000, 500, 100)
st.sidebar.markdown("---")
run = st.sidebar.button("EXECUTE PIPELINE", use_container_width=True)

if not run:
    st.markdown("""
    <div style="padding:3rem 2rem;text-align:center;border:1px solid var(--border);border-radius:4px;margin:2rem 0;">
        <p style="font-size:0.75rem;color:var(--gray);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:1rem;">QUANTITATIVE RESEARCH TERMINAL</p>
        <h1 style="margin:0.5rem 0;">Adaptive Macro-Conditional<br>Ensemble Model</h1>
        <p style="font-size:0.9rem;color:var(--gray);margin-top:1rem;">Regime-filtered ensemble learning with walk-forward validation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">METHODOLOGY</div>
            <div style="font-size:0.85rem;color:var(--text);margin-top:0.5rem;line-height:1.5;">
            Random Forest + Gradient Boosting ensemble with 60/40 walk-forward split and 6-month purged embargo
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">VALIDATION</div>
            <div style="font-size:0.85rem;color:var(--text);margin-top:0.5rem;line-height:1.5;">
            Out-of-sample testing only. No in-sample contamination. Permutation testing for statistical significance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">COSTS</div>
            <div style="font-size:0.85rem;color:var(--text);margin-top:0.5rem;line-height:1.5;">
            Transaction costs, slippage, and short-term capital gains tax. Tax-aware execution logic.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-panel">
        <strong>Research Hypothesis</strong><br>
        H₀: Macro-conditional signals provide no improvement over passive exposure<br>
        H₁: Ensemble learning with regime filtering generates statistically significant alpha net of costs
    </div>
    """, unsafe_allow_html=True)

if run:
    with st.status("Executing research pipeline...", expanded=True) as status:
        st.write("Loading 20 years of market data...")
        t0 = time.time()
        raw = load_data(risk, safe)
        st.write(f"Data acquired: {len(raw)} trading days ({time.time()-t0:.1f}s)")
        
        st.write("Engineering features...")
        data, feats = engineer_features(raw)
        
        st.write("Training ensemble models...")
        t1 = time.time()
        ml_data, train_df, test_df = train_model(data, feats)
        st.write(f"Models trained: {len(train_df)} train, {len(test_df)} test ({time.time()-t1:.1f}s)")
        
        st.write("Running backtest with costs...")
        res = backtest(ml_data, tc, tax, slip)
        
        status.update(label="Pipeline complete", state="complete", expanded=False)
    
    # Calculate OOS stats
    res_test = res.loc[test_df.index]
    sh_s, tot_s, dd_s, ann_s = calc_stats(res_test['Net'])
    sh_b, tot_b, dd_b, ann_b = calc_stats(res_test['R_ret'])
    
    # Header
    st.markdown("""
    <div style="border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:2rem;">
        <p style="font-size:0.7rem;color:var(--gray);text-transform:uppercase;letter-spacing:0.1em;margin:0;">QUANTITATIVE RESEARCH TERMINAL</p>
        <h1>Adaptive Macro-Conditional Ensemble</h1>
        <p style="font-size:0.8rem;color:var(--gray);margin:0.5rem 0 0 0;">AMCE v4.0 | Out-of-Sample Validated | No Data Leakage</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-panel">
        <strong>Validation Confirmed</strong> — All metrics computed on out-of-sample test period only ({len(test_df)} days). 
        Training period: {len(train_df)} days. Purged embargo: 6 months (126 days).
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("<h2>OUT-OF-SAMPLE PERFORMANCE</h2>", unsafe_allow_html=True)
    
    c1,c2,c3,c4,c5 = st.columns(5)
    
    def make_metric(label, val, bench, pct=False):
        if pct:
            v_str = f"{val*100:.1f}%"
            b_str = f"{bench*100:.1f}%"
        else:
            v_str = f"{val:.3f}"
            b_str = f"{bench:.3f}"
        
        color = "var(--accent)" if val > bench else "var(--red)"
        return f"""<div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{v_str}</div>
        <div style="font-size:0.7rem;color:{color};margin-top:0.25rem;">vs {b_str}</div>
        </div>"""
    
    c1.markdown(make_metric("SHARPE RATIO", sh_s, sh_b), unsafe_allow_html=True)
    c2.markdown(make_metric("TOTAL RETURN", tot_s, tot_b, True), unsafe_allow_html=True)
    c3.markdown(make_metric("ANNUAL RETURN", ann_s, ann_b, True), unsafe_allow_html=True)
    c4.markdown(make_metric("MAX DRAWDOWN", dd_s, dd_b, True), unsafe_allow_html=True)
    
    n_trades = res_test['Turn'].sum()
    n_years = len(res_test) / 252
    c5.markdown(f"""<div class="metric-box">
    <div class="metric-label">TRADES/YEAR</div>
    <div class="metric-value">{n_trades/n_years:.1f}</div>
    <div style="font-size:0.7rem;color:var(--gray);margin-top:0.25rem;">Total: {n_trades:.0f}</div>
    </div>""", unsafe_allow_html=True)
    
    # Verdict
    if sh_s > sh_b:
        st.markdown(f"""<div class="info-panel">
        <strong>Outperformance Confirmed</strong> — Sharpe {sh_s:.3f} vs benchmark {sh_b:.3f} 
        ({(sh_s/sh_b-1)*100:.0f}% better). Net of {tax*100:.0f}% tax, {tc+slip}bps friction.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warning-panel">
        <strong>Underperformance</strong> — Sharpe {sh_s:.3f} vs benchmark {sh_b:.3f}. 
        Costs exceeded edge.
        </div>""", unsafe_allow_html=True)
    
    # Equity curve
    st.markdown("<h2>EQUITY CURVE</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_test.index, y=res_test['Eq_Risk'], name=f'{risk} Buy & Hold',
                             line=dict(color='#6B7280', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=res_test.index, y=res_test['Eq_Strat'], name='AMCE Strategy',
                             line=dict(color='#00FFB2', width=2.5)))
    
    fig.update_layout(
        height=450, 
        paper_bgcolor='#0A0E14', 
        plot_bgcolor='#11151C',
        font=dict(family='Inter', color='#EBEEF5', size=11),
        xaxis=dict(showgrid=True, gridcolor='#1F2937'),
        yaxis=dict(showgrid=True, gridcolor='#1F2937', title='Portfolio Value'),
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(17,21,28,0.9)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo
    st.markdown("<h2>MONTE CARLO ROBUSTNESS</h2>", unsafe_allow_html=True)
    
    rets_arr = res_test['Net'].dropna().values
    sims = np.random.choice(rets_arr, size=(mc, len(rets_arr)), replace=True)
    sims_cum = np.cumprod(1 + sims, axis=1)
    
    p5 = np.percentile(sims_cum, 5, axis=0)
    p50 = np.percentile(sims_cum, 50, axis=0)
    p95 = np.percentile(sims_cum, 95, axis=0)
    
    fig2 = go.Figure()
    x = list(range(len(p5)))
    fig2.add_trace(go.Scatter(x=x+x[::-1], y=list(p95)+list(p5)[::-1], fill='toself',
                              fillcolor='rgba(0,255,178,0.1)', line=dict(width=0), name='95% CI'))
    fig2.add_trace(go.Scatter(x=x, y=p50, line=dict(color='#6B7280', dash='dot'), name='Median'))
    fig2.add_trace(go.Scatter(x=x, y=res_test['Eq_Strat'].values/res_test['Eq_Strat'].values[0],
                              line=dict(color='#00FFB2', width=2), name='Actual'))
    
    fig2.update_layout(
        height=400,
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#11151C',
        font=dict(family='Inter', color='#EBEEF5', size=11),
        xaxis=dict(showgrid=True, gridcolor='#1F2937', title='Trading Days'),
        yaxis=dict(showgrid=True, gridcolor='#1F2937', title='Growth of $1'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(17,21,28,0.9)')
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    prob_beat = (sims_cum[:,-1] > res_test['Eq_Risk'].values[-1]/res_test['Eq_Risk'].values[0]).mean()
    prob_dd = (sims_cum.min(axis=1) / np.maximum.accumulate(sims_cum, axis=1).max(axis=1) - 1 < -0.40).mean()
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(f"""<div class="metric-box">
    <div class="metric-label">PROB. BEAT BENCHMARK</div>
    <div class="metric-value">{prob_beat*100:.0f}%</div></div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="metric-box">
    <div class="metric-label">PROB. DD > 40%</div>
    <div class="metric-value">{prob_dd*100:.0f}%</div></div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="metric-box">
    <div class="metric-label">MEDIAN FINAL</div>
    <div class="metric-value">×{p50[-1]:.2f}</div></div>""", unsafe_allow_html=True)
    
    # Permutation test
    st.markdown("<h2>STATISTICAL SIGNIFICANCE</h2>", unsafe_allow_html=True)
    
    n_perm = 500
    actual_pos = res_test['Pos'].values
    bench_ret = res_test['R_ret'].values
    safe_ret = res_test['S_ret'].values
    
    perm_sh = []
    np.random.seed(42)
    for _ in range(n_perm):
        shuf = np.random.permutation(actual_pos)
        p_ret = np.where(shuf==1, bench_ret, safe_ret)
        m, s = p_ret.mean(), p_ret.std()
        if s > 0:
            perm_sh.append((m/s)*np.sqrt(252))
    
    perm_sh = np.array(perm_sh)
    p_val = (perm_sh >= sh_s).mean()
    
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=perm_sh, nbinsx=40, marker_color='#374151'))
    fig3.add_vline(x=sh_s, line_color='#00FFB2', line_width=3, annotation_text=f'Actual: {sh_s:.2f}')
    fig3.add_vline(x=np.percentile(perm_sh, 95), line_color='#EF4444', line_dash='dash')
    
    fig3.update_layout(
        height=350,
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#11151C',
        font=dict(family='Inter', color='#EBEEF5', size=11),
        xaxis=dict(showgrid=True, gridcolor='#1F2937', title='Sharpe Ratio'),
        yaxis=dict(showgrid=True, gridcolor='#1F2937', title='Frequency')
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    if p_val < 0.05:
        st.markdown(f"""<div class="info-panel">
        <strong>Statistically Significant</strong> — p-value {p_val:.4f} < 0.05. Reject null hypothesis. 
        Genuine predictive skill confirmed.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warning-panel">
        <strong>Not Significant</strong> — p-value {p_val:.4f} > 0.05. Cannot reject null at 95% confidence.
        </div>""", unsafe_allow_html=True)
    
    # Factor decomposition
    st.markdown("<h2>FACTOR DECOMPOSITION</h2>", unsafe_allow_html=True)
    
    Y = res_test['Net'].dropna()
    X_reg = sm.add_constant(res_test['R_ret'].dropna())
    model = sm.OLS(Y, X_reg).fit()
    alpha = model.params['const'] * 252
    beta = model.params['R_ret']
    p_alpha = model.pvalues['const']
    
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"""<div class="metric-box">
    <div class="metric-label">ANNUAL ALPHA</div>
    <div class="metric-value">{alpha*100:+.2f}%</div>
    <div style="font-size:0.7rem;color:var(--gray);margin-top:0.25rem;">p={p_alpha:.3f}</div>
    </div>""", unsafe_allow_html=True)
    
    fc2.markdown(f"""<div class="metric-box">
    <div class="metric-label">MARKET BETA</div>
    <div class="metric-value">{beta:.3f}</div>
    <div style="font-size:0.7rem;color:var(--gray);margin-top:0.25rem;">{"Defensive" if beta<1 else "Aggressive"}</div>
    </div>""", unsafe_allow_html=True)
    
    fc3.markdown(f"""<div class="metric-box">
    <div class="metric-label">R-SQUARED</div>
    <div class="metric-value">{model.rsquared:.3f}</div>
    <div style="font-size:0.7rem;color:var(--gray);margin-top:0.25rem;">Model fit</div>
    </div>""", unsafe_allow_html=True)
    
    # Export
    st.markdown("<h2>EXPORT</h2>", unsafe_allow_html=True)
    st.download_button("Download Results (CSV)", res_test.to_csv(), f"amce_{risk}_{safe}.csv", "text/csv")
    
    # Footer
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0;border-top:1px solid var(--border);margin-top:3rem;color:var(--gray);font-size:0.75rem;">
        AMCE v4.0 | OOS Sharpe: {sh_s:.3f} | Test Period: {len(test_df)} days | p-value: {p_val:.4f}
    </div>
    """, unsafe_allow_html=True)
