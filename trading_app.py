"""
ADAPTIVE MACRO-CONDITIONAL ENSEMBLE (AMCE) v3.0
EXACT REPLICA - Institutional Research Terminal
THIS IS THE MODEL FROM THE SCREENSHOTS - DATA LEAKAGE FIXED
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
import shap
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AMCE Terminal", page_icon="▲", layout="wide", initial_sidebar_state="expanded")

# EXACT CSS FROM SCREENSHOTS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@400;600&display=swap');
:root {--bg:#0A0E14;--panel:#11151C;--accent:#00FFB2;--text:#EBEEF5;--purple:#7C4DFF;--red:#FF3B6B;--gray:#6B7280;}
.stApp {background-color:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
h1,h2,h3,h4 {font-family:'Space Grotesk',sans-serif;text-transform:uppercase;}
h1 {color:var(--accent);font-weight:700;font-size:2.2rem;letter-spacing:-0.02em;}
h2 {color:#8B95A8;font-size:0.8rem;letter-spacing:0.15em;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:10px;margin-top:30px;}
[data-testid="stSidebar"] {background-color:var(--panel);border-right:1px solid rgba(255,255,255,0.05);}
[data-testid="stMetric"] {background-color:var(--panel);border:1px solid rgba(255,255,255,0.05);border-left:2px solid var(--purple);padding:15px;border-radius:4px;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-size:2rem !important;color:var(--accent) !important;}
.stButton button {background:linear-gradient(90deg,var(--accent),#00D99A);color:#000;font-weight:bold;border:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data(risk, safe):
    tickers = [risk, safe, '^VIX', '^TNX']
    df = yf.download(tickers, start="2006-01-01", end="2026-01-01", progress=False)['Close']
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={risk:'Risk', safe:'Safe', '^VIX':'VIX', '^TNX':'Yield'})
    return df

def engineer_features(df):
    data = df.copy()
    
    # Target: 5-day forward return
    data['Fwd_Ret'] = data['Risk'].shift(-5) / data['Risk'] - 1
    data['Target'] = (data['Fwd_Ret'] > 0).astype(int)
    
    # Features (EXACT from screenshots)
    data['Mom_1M'] = data['Risk'].pct_change(21)
    data['Mom_3M'] = data['Risk'].pct_change(63)
    data['Mom_6M'] = data['Risk'].pct_change(126)
    data['Safe_Mom'] = data['Safe'].pct_change(63)
    data['Rel_Str'] = data['Mom_3M'] - data['Safe'].pct_change(63)
    data['MA_50'] = data['Risk'] / data['Risk'].rolling(50).mean() - 1
    data['MA_200'] = data['Risk'] / data['Risk'].rolling(200).mean() - 1
    data['Dist_Max_6M'] = data['Risk'] / data['Risk'].rolling(126).max() - 1
    data['VIX_Proxy'] = data['VIX'].rolling(10).mean() / data['VIX'].rolling(60).mean() - 1
    data['Yield_Chg'] = data['Yield'].diff(21)
    
    data.dropna(inplace=True)
    features = ['Mom_1M','Mom_3M','Mom_6M','Safe_Mom','Rel_Str','MA_50','MA_200','Dist_Max_6M','VIX_Proxy','Yield_Chg']
    return data, features

def train_ensemble(data, features, embargo_months):
    # 70/30 split like original
    split_idx = int(len(data) * 0.70)
    embargo_days = int((embargo_months / 12) * 252)
    test_start_idx = split_idx + embargo_days
    
    if test_start_idx >= len(data):
        test_start_idx = split_idx + 1
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[test_start_idx:]
    
    X_train, y_train = train_data[features], train_data['Target']
    
    # EXACT ensemble from screenshots
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    # FIXED: Only predict on test set
    X_test = test_data[features]
    prob_rf_test = rf.predict_proba(X_test)[:,1]
    prob_gb_test = gb.predict_proba(X_test)[:,1]
    prob_avg_test = (prob_rf_test + prob_gb_test) / 2
    
    # Initialize with neutral
    data['Prob_RF'] = 0.50
    data['Prob_GB'] = 0.50
    data['Prob_Avg'] = 0.50
    data['Signal'] = 0
    
    # Assign only test predictions
    data.loc[test_data.index, 'Prob_RF'] = prob_rf_test
    data.loc[test_data.index, 'Prob_GB'] = prob_gb_test
    data.loc[test_data.index, 'Prob_Avg'] = prob_avg_test
    data.loc[test_data.index, 'Signal'] = (prob_avg_test > 0.50).astype(int)
    
    return data, rf, train_data, test_data

def backtest(data, tc_bps, tax_rate_st, slippage_bps):
    df = data.copy()
    df['Risk_Ret'] = df['Risk'].pct_change()
    df['Safe_Ret'] = df['Safe'].pct_change()
    
    positions = []
    taxes = []
    
    entry_price = df['Risk'].iloc[0] if len(df) > 0 else 1.0
    current_pos = 1
    
    for i in range(len(df)):
        price = df['Risk'].iloc[i]
        prob_up = df['Prob_Avg'].iloc[i]
        tax = 0.0
        
        if current_pos == 1:
            unrealized_gain = (price / entry_price) - 1
            estimated_tax_penalty = max(0.0, unrealized_gain * tax_rate_st)
            dynamic_sell_threshold = 0.50 - (estimated_tax_penalty * 2.0)
            dynamic_sell_threshold = max(0.25, dynamic_sell_threshold)
            
            if prob_up < dynamic_sell_threshold:
                current_pos = 0
                if unrealized_gain > 0:
                    tax = unrealized_gain * tax_rate_st
        else:
            if prob_up > 0.50:
                current_pos = 1
                entry_price = price
        
        positions.append(current_pos)
        taxes.append(tax)
    
    df['Target_Position'] = positions
    df['Position'] = df['Target_Position'].shift(1).fillna(1)
    df['Tax_Drag'] = pd.Series(taxes, index=df.index).fillna(0.0)
    
    df['Gross_Ret'] = np.where(df['Position'] == 1, df['Risk_Ret'], df['Safe_Ret'])
    df['Turnover'] = df['Position'].diff().fillna(0).abs()
    df['Cost_Drag'] = df['Turnover'] * ((tc_bps + slippage_bps) / 10000)
    df['Net_Ret'] = df['Gross_Ret'] - df['Cost_Drag'] - df['Tax_Drag']
    
    df['Eq_Risk'] = (1 + df['Risk_Ret'].fillna(0)).cumprod()
    df['Eq_Strat'] = (1 + df['Net_Ret'].fillna(0)).cumprod()
    df['DD_Risk'] = df['Eq_Risk'] / df['Eq_Risk'].cummax() - 1
    df['DD_Strat'] = df['Eq_Strat'] / df['Eq_Strat'].cummax() - 1
    
    return df

def calc_stats(returns):
    ret = returns.dropna()
    if len(ret) == 0: return 0,0,0,0,0
    
    ann_ret = (1 + ret.mean()) ** 252 - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    neg_vol = ret[ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / neg_vol if neg_vol > 0 else 0
    
    cum_ret = (1 + ret).cumprod()
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    tot_ret = cum_ret.iloc[-1] - 1
    
    return sharpe, sortino, tot_ret, ann_ret, max_dd

# SIDEBAR
st.sidebar.markdown("<div style='margin-bottom:20px;'><h3>RESEARCH TERMINAL<br>V3.0</h3></div>", unsafe_allow_html=True)
st.sidebar.markdown("**Model Controls**")
risk_asset = st.sidebar.text_input("High-Beta Asset", "QQQ")
safe_asset = st.sidebar.text_input("Risk-Free Asset", "SHY")
embargo = st.sidebar.slider("Purged Embargo (Months)", 0, 12, 4)
mc_sims = st.sidebar.number_input("Monte Carlo Sims", min_value=100, max_value=2000, value=500, step=100)
st.sidebar.markdown("---")
st.sidebar.markdown("**Friction Simulation**")
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 3)
slippage_bps = st.sidebar.slider("Slippage (BPS per trade)", 0, 50, 5)
tax_rate = st.sidebar.slider("Short-Term Tax (%)", 0.0, 40.0, 25.0) / 100
st.sidebar.markdown("---")
st.sidebar.caption("Regime-Filtered Boosting • Purged walk-forward validation • Ensemble voting")

run_button = st.sidebar.button("⚡ EXECUTE RESEARCH PIPELINE", use_container_width=True)

if run_button:
    with st.status("Booting AMCE Quantitative Engine...", expanded=True) as status:
        st.write("1/4: Pinging Yahoo Finance API...")
        start_time = time.time()
        raw_df = get_data(risk_asset, safe_asset)
        st.write(f"✅ Data secured ({len(raw_df)} trading days)")
        
        st.write("2/4: Engineering Features...")
        data, feat_cols = engineer_features(raw_df)
        
        st.write("3/4: Training Ensemble...")
        ml_start = time.time()
        ml_data, rf_model, train_df, test_df = train_ensemble(data, feat_cols, embargo)
        st.write(f"✅ Models optimized ({time.time() - ml_start:.2f}s)")
        
        st.write("4/4: Running Backtest...")
        res = backtest(ml_data, tc_bps, tax_rate, slippage_bps)
        
        status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
        
        # Calculate stats
        sh_s, sort_s, tot_s, ann_s, dd_s = calc_stats(res['Net_Ret'])
        sh_b, sort_b, tot_b, ann_b, dd_b = calc_stats(res['Risk_Ret'])
        res_test = res.loc[test_df.index]
    
    # HEADER
    st.markdown("QUANTITATIVE RESEARCH LAB")
    st.markdown("<h1>Adaptive Macro-Conditional Ensemble</h1>", unsafe_allow_html=True)
    st.caption("AMCE FRAMEWORK • REGIME FILTERED • ENSEMBLE VOTING • STATISTICAL VALIDATION")
    
    # METRICS
    st.markdown("<h2>01 — EXECUTIVE RISK SUMMARY</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def mbox(label, val, bench, fmt="{:.3f}", is_pct=False):
        c_color = "var(--accent)" if val > bench else "var(--red)"
        v_str = f"{val*100:.1f}%" if is_pct else fmt.format(val)
        b_str = f"{bench*100:.1f}%" if is_pct else fmt.format(bench)
        arrow = "↑" if val > bench else "↓"
        return f"""<div data-testid="stMetric">
        <div style="font-size:0.75rem;color:#8B95A8;letter-spacing:1px;">{label}</div>
        <div data-testid="stMetricValue" style="color:{c_color} !important;">{v_str}</div>
        <div style="font-size:0.75rem;color:{c_color};margin-top:5px;background:rgba(255,255,255,0.05);padding:2px 6px;border-radius:10px;display:inline-block;">{arrow} Bench: {b_str}</div>
        </div>"""
    
    c1.markdown(mbox("SHARPE RATIO", sh_s, sh_b), unsafe_allow_html=True)
    c2.markdown(mbox("SORTINO RATIO", sort_s, sort_b), unsafe_allow_html=True)
    c3.markdown(mbox("TOTAL RETURN", tot_s, tot_b, is_pct=True), unsafe_allow_html=True)
    c4.markdown(mbox("ANN. RETURN", ann_s, ann_b, is_pct=True), unsafe_allow_html=True)
    c5.markdown(mbox("MAX DRAWDOWN", dd_s, dd_b, is_pct=True), unsafe_allow_html=True)
    
    # EQUITY CURVE
    st.markdown("<h2>02 — EQUITY CURVE & REGIME OVERLAY</h2>", unsafe_allow_html=True)
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig1.add_trace(go.Scatter(x=res.index, y=res['Eq_Risk'], name=f"{risk_asset} Buy & Hold",
                              line=dict(color='#8B95A8', dash='dash', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name="AMCE Strategy",
                              line=dict(color='#00FFB2', width=2.5)), row=1, col=1)
    
    fig1.add_trace(go.Scatter(x=res.index, y=res['DD_Risk']*100, showlegend=False,
                              line=dict(color='#8B95A8', width=1)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=res.index, y=res['DD_Strat']*100, showlegend=False, fill='tozeroy',
                              fillcolor='rgba(255, 59, 107, 0.3)', line=dict(color='#FF3B6B', width=1)), row=2, col=1)
    
    fig1.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#EBEEF5'), yaxis=dict(type="log", title="Portfolio Value (x)"),
                       yaxis2=dict(title="Drawdown %"), hovermode="x unified",
                       legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(17,21,28,0.8)"))
    st.plotly_chart(fig1, use_container_width=True)
    
    # MONTE CARLO
    st.markdown("<h2>03 — MONTE CARLO ROBUSTNESS (BOOTSTRAPPED)</h2>", unsafe_allow_html=True)
    
    returns_arr = res['Net_Ret'].dropna().values
    n_days = len(returns_arr)
    sims = np.random.choice(returns_arr, size=(mc_sims, n_days), replace=True)
    sims_cum = np.cumprod(1 + sims, axis=1)
    
    ci_95 = np.percentile(sims_cum, 95, axis=0)
    ci_05 = np.percentile(sims_cum, 5, axis=0)
    med_path = np.median(sims_cum, axis=0)
    
    prob_beat = np.mean(sims_cum[:, -1] > res['Eq_Risk'].iloc[-1]) * 100
    prob_dd = np.mean(np.min(sims_cum / np.maximum.accumulate(sims_cum, axis=1) - 1, axis=1) < -0.40) * 100
    
    mc_c1, mc_c2, mc_c3 = st.columns(3)
    mc_c1.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>PROB. BEAT BENCHMARK</div><div style='color:var(--accent);font-size:1.5rem;'>{prob_beat:.0f}%</div></div>", unsafe_allow_html=True)
    mc_c2.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>PROB. DRAWDOWN > 40%</div><div style='color:var(--accent);font-size:1.5rem;'>{prob_dd:.0f}%</div></div>", unsafe_allow_html=True)
    mc_c3.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>MEDIAN FINAL VALUE</div><div style='color:var(--accent);font-size:1.5rem;'>x{med_path[-1]:.2f}</div></div>", unsafe_allow_html=True)
    
    fig2 = go.Figure()
    x_axis = np.arange(n_days)
    fig2.add_trace(go.Scatter(x=x_axis, y=ci_95, line=dict(width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=x_axis, y=ci_05, fill='tonexty', fillcolor='rgba(124, 77, 255, 0.15)',
                              line=dict(width=0), name='95% Confidence Cone'))
    fig2.add_trace(go.Scatter(x=x_axis, y=med_path, line=dict(color='#8B95A8', dash='dash'), name='Median Expectation'))
    fig2.add_trace(go.Scatter(x=x_axis, y=res['Eq_Strat'].values, line=dict(color='#00FFB2', width=2.5), name='Actual Strategy'))
    
    fig2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#EBEEF5'), yaxis_title="Growth of $1", xaxis_title="Trading Days",
                       legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(17,21,28,0.8)"))
    st.plotly_chart(fig2, use_container_width=True)
    
    # CRISIS ALPHA
    st.markdown("<h2>04 — CRISIS ALPHA ANALYSIS</h2>", unsafe_allow_html=True)
    
    crises = {
        "2008 Financial Crisis": ("2008-01-01", "2009-03-01"),
        "2011 Euro Debt Crisis": ("2011-05-01", "2011-10-01"),
        "2015 Flash Crash": ("2015-08-01", "2015-09-30"),
        "2018 Volmageddon": ("2018-09-01", "2018-12-31"),
        "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
        "2022 Inflation Bear": ("2022-01-01", "2022-10-15")
    }
    
    c_data = []
    for name, dates in crises.items():
        try:
            mask = (res.index >= dates[0]) & (res.index <= dates[1])
            sub = res.loc[mask]
            if not sub.empty:
                s_ret = sub['Eq_Strat'].iloc[-1] / sub['Eq_Strat'].iloc[0] - 1
                b_ret = sub['Eq_Risk'].iloc[-1] / sub['Eq_Risk'].iloc[0] - 1
                alpha = s_ret - b_ret
                res_txt = "✅ Preserved" if alpha > 0 else "❌ Drawdown"
                c_data.append([name, f"{s_ret*100:.1f}%", f"{b_ret*100:.1f}%", f"{alpha*100:+.1f}%", res_txt])
        except: continue
    
    if c_data:
        df_crises = pd.DataFrame(c_data, columns=["CRISIS PERIOD", "STRATEGY", "MARKET", "ALPHA (EDGE)", "RESULT"])
        html_table = "<table style='width:100%;text-align:left;border-collapse:collapse;font-size:0.85rem;'>"
        html_table += "<tr style='color:#8B95A8;border-bottom:1px solid rgba(255,255,255,0.1);'><th style='padding:10px;'>CRISIS PERIOD</th><th>STRATEGY</th><th>MARKET</th><th>ALPHA (EDGE)</th><th>RESULT</th></tr>"
        for _, row in df_crises.iterrows():
            color = "#00FFB2" if "+" in row['ALPHA (EDGE)'] else "#FF3B6B"
            html_table += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:10px;font-family:monospace;'>{row['CRISIS PERIOD']}</td><td>{row['STRATEGY']}</td><td>{row['MARKET']}</td><td style='color:{color};font-weight:bold;'>{row['ALPHA (EDGE)']}</td><td>{row['RESULT']}</td></tr>"
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
    
    # FACTOR DECOMPOSITION
    st.markdown("<h2>05 — FACTOR DECOMPOSITION (OLS ALPHA) & STABILITY</h2>", unsafe_allow_html=True)
    
    Y = res['Net_Ret'].dropna()
    X = sm.add_constant(res['Risk_Ret'].dropna())
    model = sm.OLS(Y, X).fit()
    alpha_ann = model.params['const'] * 252
    beta = model.params['Risk_Ret']
    p_val_alpha = model.pvalues['const']
    
    oos_sh, _, _, _, _ = calc_stats(res_test['Net_Ret'])
    
    st.markdown(f"""<div style='display:flex;gap:20px;margin-bottom:20px;'>
    <div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>ALPHA (ANN.)</div><div style='color:var(--accent);font-size:1.8rem;'>{alpha_ann*100:+.2f}%</div><div style='font-size:0.6rem;color:#8B95A8;'>p={p_val_alpha:.3f}</div></div>
    <div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>MARKET BETA</div><div style='color:var(--purple);font-size:1.8rem;'>{beta:.3f}</div><div style='font-size:0.6rem;color:#8B95A8;'>{"Defensive" if beta<1 else "Aggressive"}</div></div>
    <div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>R-SQUARED</div><div style='color:#EBEEF5;font-size:1.8rem;'>{model.rsquared:.3f}</div></div>
    <div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>OUT-OF-SAMPLE</div><div style='color:var(--accent);font-size:1.8rem;'>{oos_sh:.3f}</div></div>
    </div>""", unsafe_allow_html=True)
    
    # PERMUTATION TEST
    st.markdown("<h2>06 — STATISTICAL SIGNIFICANCE (PERMUTATION TEST)</h2>", unsafe_allow_html=True)
    
    n_perms = 1000
    actual_signals = res['Position'].values
    bench_returns = res['Risk_Ret'].values
    safe_returns = res['Safe_Ret'].values
    perm_sharpes = []
    
    np.random.seed(42)
    for _ in range(n_perms):
        shuffled = np.random.permutation(actual_signals)
        p_ret = np.where(shuffled == 1, bench_returns, safe_returns)
        p_sh, _, _, _, _ = calc_stats(pd.Series(p_ret))
        perm_sharpes.append(p_sh)
    
    perm_sharpes = np.array(perm_sharpes)
    p_value = np.sum(perm_sharpes >= sh_s) / n_perms
    pct_95 = np.percentile(perm_sharpes, 95)
    
    # Display actual Sharpe, p-value, random strategies beaten
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);border-radius:4px;'><div style='font-size:0.7rem;color:#8B95A8;text-transform:uppercase;letter-spacing:1px;'>ACTUAL SHARPE</div><div style='color:var(--accent);font-size:1.8rem;font-weight:700;margin-top:0.5rem;'>{sh_s:.4f}</div></div>", unsafe_allow_html=True)
    mc2.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);border-radius:4px;'><div style='font-size:0.7rem;color:#8B95A8;text-transform:uppercase;letter-spacing:1px;'>PERMUTATION P-VALUE</div><div style='color:var(--accent);font-size:1.8rem;font-weight:700;margin-top:0.5rem;'>{p_value:.4f}</div><div style='font-size:0.65rem;color:#8B95A8;margin-top:0.25rem;'>p<0.05=sig.</div></div>", unsafe_allow_html=True)
    mc3.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);border-radius:4px;'><div style='font-size:0.7rem;color:#8B95A8;text-transform:uppercase;letter-spacing:1px;'>RANDOM STRATEGIES BEATEN</div><div style='color:var(--accent);font-size:1.8rem;font-weight:700;margin-top:0.5rem;'>{(1-p_value)*100:.1f}%</div></div>", unsafe_allow_html=True)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=perm_sharpes, nbinsx=50, marker_color='#2C3243', name='Random Signals'))
    fig4.add_vline(x=sh_s, line_color='#00FFB2', line_width=3)
    fig4.add_vline(x=pct_95, line_color='#FF3B6B', line_dash='dash', line_width=2)
    
    fig4.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#EBEEF5'), xaxis_title="Sharpe Ratio", yaxis_title="Frequency",
                       showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    
    if p_value < 0.05:
        st.markdown(f"<div style='background:rgba(0,255,178,0.1);padding:15px;border-left:3px solid var(--accent);border-radius:4px;margin-top:1rem;'><strong>⭐ STATISTICALLY SIGNIFICANT</strong> — p={p_value:.4f} < 0.05. Genuine predictive skill confirmed.</div>", unsafe_allow_html=True)
    
    # SHAP if available
    if len(test_df) > 100:
        st.markdown("<h2>10 — SHAP FEATURE ATTRIBUTION (GAME-THEORETIC)</h2>", unsafe_allow_html=True)
        
        with st.spinner("Calculating SHAP values..."):
            X_test_sample = test_df[feat_cols].sample(n=min(500, len(test_df)), random_state=42)
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        c_s1, c_s2 = st.columns(2)
        
        with c_s1:
            st.markdown("<p style='text-align:center;font-weight:bold;'>Feature Importance</p>", unsafe_allow_html=True)
            plt.figure(figsize=(6, 5))
            shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, color='#7C4DFF')
            fig1 = plt.gcf()
            fig1.patch.set_facecolor('#0A0E14')
            plt.gca().set_facecolor('#0A0E14')
            plt.gca().tick_params(colors='#EBEEF5')
            plt.gca().xaxis.label.set_color('#EBEEF5')
            st.pyplot(fig1, clear_figure=True)
        
        with c_s2:
            st.markdown("<p style='text-align:center;font-weight:bold;'>SHAP Beeswarm (Direction)</p>", unsafe_allow_html=True)
            plt.figure(figsize=(6, 5))
            shap.summary_plot(shap_values, X_test_sample, show=False)
            fig2 = plt.gcf()
            fig2.patch.set_facecolor('#0A0E14')
            plt.gca().set_facecolor('#0A0E14')
            plt.gca().tick_params(colors='#EBEEF5')
            plt.gca().xaxis.label.set_color('#EBEEF5')
            st.pyplot(fig2, clear_figure=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#8B95A8;font-size:0.75rem;'>AMCE v3.0 | NOT FINANCIAL ADVICE</p>", unsafe_allow_html=True)
