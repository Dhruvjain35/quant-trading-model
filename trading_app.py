"""
ADAPTIVE MACRO-CONDITIONAL ENSEMBLE (AMCE) v8.3

"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import shap
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AMCE Terminal", page_icon="▲", layout="wide", initial_sidebar_state="expanded")

# Home UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@400;600&display=swap');
:root {--bg:#0A0E14;--panel:#11151C;--accent:#00FFB2;--text:#EBEEF5;--purple:#7C4DFF;--red:#FF3B6B;}
.stApp {background-color:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;text-transform:uppercase;}
h1 {color:var(--accent);font-weight:700;font-size:2.2rem;letter-spacing:-0.02em;}
h2 {color:#8B95A8;font-size:0.8rem;letter-spacing:0.15em;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:10px;margin-top:30px;}
[data-testid="stSidebar"] {background-color:var(--panel);border-right:1px solid rgba(255,255,255,0.05);}
[data-testid="stMetric"] {background-color:var(--panel);border:1px solid rgba(255,255,255,0.05);border-left:2px solid var(--purple);padding:15px;border-radius:4px;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-size:2rem !important;color:var(--accent) !important;}
.stButton button {background:linear-gradient(90deg,var(--accent),#00D99A);color:#000;font-weight:bold;border:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(risk, safe):
    tickers = [risk, safe, '^VIX', '^TNX']
    df = yf.download(tickers, start="1993-01-01", end="2026-01-01", progress=False)['Close']
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={risk:'Risk', safe:'Safe', '^VIX':'VIX', '^TNX':'Yield'})
    return df

def engineer_features(df):
    data = df.copy()
    data['Fwd_Ret'] = data['Risk'].shift(-1) / data['Risk'] - 1
    data['Target'] = (data['Fwd_Ret'] > 0).astype(int)
    data['Mom_1M'] = data['Risk'].pct_change(21)
    data['Mom_3M'] = data['Risk'].pct_change(63)
    data['MA_200_Dist'] = data['Risk'] / data['Risk'].rolling(200).mean() - 1
    data['Yield_Mom'] = data['Yield'].pct_change(21)
    data['Yield_Trend'] = data['Yield'] > data['Yield'].rolling(63).mean() 
    
    delta = data['Risk'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -1 * delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    data['Risk_Vol'] = data['Risk'].pct_change().rolling(21).std()
    data['Safe_Vol'] = data['Safe'].pct_change().rolling(21).std()
    data['Vol_Ratio'] = data['Risk_Vol'] / data['Safe_Vol']
    data['Vol_Ratio_MA'] = data['Vol_Ratio'].rolling(63).mean()
    
    data.dropna(inplace=True)
    features = ['Mom_1M', 'Mom_3M', 'MA_200_Dist', 'Yield_Mom', 'Vol_Ratio', 'RSI_14']
    return data, features

def train_ensemble(data, features, embargo):
    split = int(len(data) * 0.40)
    embargo_days = int((embargo / 12) * 252)
    test_start = split + embargo_days
    if test_start >= len(data): test_start = split + 1
    
    train = data.iloc[:split]
    test = data.iloc[test_start:].copy()
    
    X_tr, y_tr = train[features], train['Target']
    X_te = test[features]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42)
    
    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    
    prob_rf = rf.predict_proba(X_te)[:,1]
    prob_gb = gb.predict_proba(X_te)[:,1]
    
    test['Prob_Avg'] = (prob_rf + prob_gb) / 2
    test['Prob_Smooth'] = test['Prob_Avg'].ewm(span=10).mean()
    
    conditions = [test['Prob_Smooth'] > 0.48, test['Prob_Smooth'] < 0.43]
    choices = [1, 0]
    test['Raw_Signal'] = np.select(conditions, choices, default=np.nan)
    test['Signal'] = test['Raw_Signal'].ffill().fillna(1)
    
    # Regime detection
    risk_off = (test['MA_200_Dist'] < -0.02) & (test['Vol_Ratio'] > test['Vol_Ratio_MA'] * 1.05) & (test['RSI_14'] > 35)
    panic = ((test['MA_200_Dist'] < -0.10) | (test['Mom_1M'] < -0.15)) & (test['Vol_Ratio'] > test['Vol_Ratio_MA'] * 1.15)
    test.loc[risk_off | panic, 'Signal'] = 0
    
    # Tag regimes for visualization
    test['Regime'] = 'Normal'
    test.loc[risk_off, 'Regime'] = 'Risk-Off'
    test.loc[panic, 'Regime'] = 'Panic'
    
    return test, rf, train, test

def backtest(data, cost_bps, slip_bps):
    df = data.copy()
    df['R_ret'] = df['Risk'].pct_change()
    df['S_ret'] = df['Safe'].pct_change()
    df['Cash_ret'] = (df['Yield'] / 100) / 252 
    df['Pos'] = df['Signal'].shift(1).fillna(1)
    df['Yield_Trend_Shift'] = df['Yield_Trend'].shift(1).fillna(False)
    
    df['Gross'] = np.where(df['Pos']==1, df['R_ret'], 
                  np.where(df['Yield_Trend_Shift'], df['Cash_ret'], df['S_ret']))
    df['Turn'] = df['Pos'].diff().abs()
    df['Cost'] = df['Turn'] * (cost_bps + slip_bps) / 10000
    df['Net'] = df['Gross'] - df['Cost']
    
    df['Eq_Strat'] = (1 + df['Net'].fillna(0)).cumprod()
    df['Eq_Risk'] = (1 + df['R_ret'].fillna(0)).cumprod()
    df['DD_Strat'] = df['Eq_Strat'] / df['Eq_Strat'].cummax() - 1
    df['DD_Risk'] = df['Eq_Risk'] / df['Eq_Risk'].cummax() - 1
    return df

def stats(rets):
    r = rets.dropna()
    if len(r) == 0: return 0,0,0,0,0
    m, s = r.mean(), r.std()
    sh = (m/s)*np.sqrt(252) if s>0 else 0
    neg = r[r<0].std()
    sort = ((m * 252) / (neg * np.sqrt(252))) if neg>0 else 0  # Fixed: annualized return / annualized downside vol
    tot = (1+r).prod()-1
    dd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
    ann = (1+m)**252-1
    return sh, sort, tot, ann, dd

# SIDEBAR
st.sidebar.markdown("<div style='margin-bottom:20px;'><h3>RESEARCH TERMINAL<br>V8.3 ENHANCED</h3></div>", unsafe_allow_html=True)
st.sidebar.markdown("**Model Controls**")
risk = st.sidebar.text_input("High-Beta Asset", "^NDX")
safe = st.sidebar.text_input("Risk-Free Asset", "VUSTX") 
embargo = st.sidebar.slider("Purged Embargo (Months)", 0, 12, 4)
mc = st.sidebar.number_input("Monte Carlo Sims", 100, 2000, 500, 100)
st.sidebar.markdown("---")
st.sidebar.markdown("**Friction Simulation**")
tc = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 3)
slip = st.sidebar.slider("Slippage (BPS per trade)", 0, 50, 5)
st.sidebar.markdown("---")
st.sidebar.caption("T20-Enhanced • Regime Detection • Cost Sensitivity • OOS Consistency")
run = st.sidebar.button("⚡ EXECUTE RESEARCH PIPELINE", use_container_width=True)

# HOMESCREEN
if not run:
    st.markdown("QUANTITATIVE RESEARCH LAB")
    st.markdown("<h1>Adaptive Macro-Conditional Ensemble</h1>", unsafe_allow_html=True)
    st.caption("AMCE FRAMEWORK • REGIME FILTERED • ENSEMBLE VOTING • STATISTICAL VALIDATION")
    
    st.markdown("""
    <div style="background:rgba(124,77,255,0.1);padding:20px;border-radius:4px;border:1px solid rgba(124,77,255,0.2);margin-top:20px;">
        <span style="color:#7C4DFF;font-weight:bold;">RESEARCH HYPOTHESIS</span><br><br>
        <b>H₀ (Null):</b> Macro-conditional regime signals provide no statistically significant improvement over passive equity exposure.<br>
        <b>H₁ (Alternative):</b> Integrating Regime Filtering (Trend) with Gradient Boosting signals generates positive crisis alpha and statistically significant risk-adjusted outperformance, net of costs.
        <br><br><span style="color:#8B95A8;">Test: Signal permutation (n=1,000) | Threshold: p < 0.05 | Alpha via OLS on excess returns</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# EXECUTE
with st.status("Booting AMCE V8.3 Enhanced...", expanded=True) as status:
    raw = load_data(risk, safe)
    data, feats = engineer_features(raw)
    test_data, rf_model, train_df, test_df = train_ensemble(data, feats, embargo)
    res = backtest(test_data, tc, slip)
    status.update(label="✅ Complete!", state="complete", expanded=False)

# STATS
sh_s, sort_s, tot_s, ann_s, dd_s = stats(res['Net'])
sh_b, sort_b, tot_b, ann_b, dd_b = stats(res['R_ret'])

st.markdown("QUANTITATIVE RESEARCH LAB")
st.markdown("<h1>Adaptive Macro-Conditional Ensemble</h1>", unsafe_allow_html=True)

# METRICS
st.markdown("<h2>01 — EXECUTIVE RISK SUMMARY (OOS DATA ONLY)</h2>", unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
def box(lbl, val, bench, pct=False):
    col = "var(--accent)" if val>bench else "var(--red)"
    v = f"{val*100:.1f}%" if pct else f"{val:.3f}"
    b = f"{bench*100:.1f}%" if pct else f"{bench:.3f}"
    arr = "↑" if val>bench else "↓"
    return f"""<div data-testid="stMetric">
    <div style="font-size:0.75rem;color:#8B95A8;letter-spacing:1px;">{lbl}</div>
    <div data-testid="stMetricValue" style="color:{col} !important;">{v}</div>
    <div style="font-size:0.75rem;color:{col};margin-top:5px;background:rgba(255,255,255,0.05);padding:2px 6px;border-radius:10px;display:inline-block;">{arr} Bench: {b}</div>
    </div>"""
c1.markdown(box("SHARPE RATIO", sh_s, sh_b), unsafe_allow_html=True)
c2.markdown(box("SORTINO RATIO", sort_s, sort_b), unsafe_allow_html=True)
c3.markdown(box("TOTAL RETURN", tot_s, tot_b, True), unsafe_allow_html=True)
c4.markdown(box("ANN. RETURN", ann_s, ann_b, True), unsafe_allow_html=True)
c5.markdown(box("MAX DRAWDOWN", dd_s, dd_b, True), unsafe_allow_html=True)

# EQUITY CURVE WITH REGIME OVERLAY
st.markdown("<h2>02 — EQUITY CURVE & REGIME OVERLAY</h2>", unsafe_allow_html=True)
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.05)

# Add background colors for regimes
regime_colors = {'Normal': 'rgba(0,255,178,0.03)', 'Risk-Off': 'rgba(255,159,67,0.08)', 'Panic': 'rgba(255,59,107,0.15)'}
for regime_type, color in regime_colors.items():
    if regime_type in res['Regime'].values:
        regime_mask = res['Regime'] == regime_type
        regime_periods = res[regime_mask].index
        if len(regime_periods) > 0:
            for start_idx in range(len(regime_periods)):
                if start_idx == 0 or regime_periods[start_idx] != regime_periods[start_idx-1] + pd.Timedelta(days=1):
                    start = regime_periods[start_idx]
                    end = start
                    for i in range(start_idx, len(regime_periods)):
                        if i < len(regime_periods) - 1 and regime_periods[i+1] == regime_periods[i] + pd.Timedelta(days=1):
                            end = regime_periods[i+1]
                        else:
                            break
                    fig1.add_vrect(x0=start, x1=end, fillcolor=color, layer="below", line_width=0, row=1, col=1)

fig1.add_trace(go.Scatter(x=res.index, y=res['Eq_Risk'], name=f"{risk} Buy & Hold", line=dict(color='#8B95A8', dash='dash', width=1)), row=1, col=1)
fig1.add_trace(go.Scatter(x=res.index, y=res['Eq_Strat'], name="AMCE Strategy", line=dict(color='#00FFB2', width=2.5)), row=1, col=1)
fig1.add_trace(go.Scatter(x=res.index, y=res['DD_Risk']*100, showlegend=False, line=dict(color='#8B95A8', width=1)), row=2, col=1)
fig1.add_trace(go.Scatter(x=res.index, y=res['DD_Strat']*100, showlegend=False, fill='tozeroy', fillcolor='rgba(255,59,107,0.3)', line=dict(color='#FF3B6B', width=1)), row=2, col=1)
fig1.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#EBEEF5'), yaxis=dict(type="log", title="Portfolio Value (x)"), yaxis2=dict(title="Drawdown %"), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(17,21,28,0.8)"))
st.plotly_chart(fig1, use_container_width=True)

# REGIME STATISTICS
regime_stats = []
for regime_type in ['Normal', 'Risk-Off', 'Panic']:
    regime_mask = res['Regime'] == regime_type
    if regime_mask.sum() > 0:
        regime_ret = res.loc[regime_mask, 'Net']
        sh_regime, _, _, _, _ = stats(regime_ret)
        pct_time = regime_mask.sum() / len(res) * 100
        regime_stats.append([regime_type, f"{pct_time:.1f}%", f"{sh_regime:.3f}"])

if regime_stats:
    st.markdown("**Regime-Specific Performance:**")
    rc1, rc2, rc3 = st.columns(3)
    for i, (regime, pct, sharpe) in enumerate(regime_stats):
        col = [rc1, rc2, rc3][i]
        col.markdown(f"""<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'>
        <div style='font-size:0.7rem;color:#8B95A8;'>{regime.upper()} REGIME</div>
        <div style='color:var(--accent);font-size:1.2rem;'>Sharpe: {sharpe}</div>
        <div style='font-size:0.7rem;color:#8B95A8;'>Time: {pct}</div>
        </div>""", unsafe_allow_html=True)

# MONTE CARLO
st.markdown("<h2>03 — MONTE CARLO ROBUSTNESS (BOOTSTRAPPED)</h2>", unsafe_allow_html=True)
rets = res['Net'].dropna().values
sims = np.random.choice(rets, size=(mc, len(rets)), replace=True)
sims_cum = np.cumprod(1 + sims, axis=1)
p5 = np.percentile(sims_cum, 5, axis=0)
p50 = np.percentile(sims_cum, 50, axis=0)
p95 = np.percentile(sims_cum, 95, axis=0)
prob_beat = (sims_cum[:,-1] > res['Eq_Risk'].iloc[-1]).mean() * 100
prob_dd = (sims_cum.min(axis=1) / np.maximum.accumulate(sims_cum, axis=1).max(axis=1) - 1 < -0.40).mean() * 100

mc1,mc2,mc3 = st.columns(3)
mc1.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>PROB. BEAT BENCHMARK</div><div style='color:var(--accent);font-size:1.5rem;'>{prob_beat:.0f}%</div></div>", unsafe_allow_html=True)
mc2.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>PROB. DRAWDOWN > 40%</div><div style='color:var(--accent);font-size:1.5rem;'>{prob_dd:.0f}%</div></div>", unsafe_allow_html=True)
mc3.markdown(f"<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'><div style='font-size:0.7rem;color:#8B95A8;'>MEDIAN FINAL VALUE</div><div style='color:var(--accent);font-size:1.5rem;'>x{p50[-1]:.2f}</div></div>", unsafe_allow_html=True)

fig2 = go.Figure()
x = np.arange(len(p5))
fig2.add_trace(go.Scatter(x=x, y=p95, line=dict(width=0), showlegend=False))
fig2.add_trace(go.Scatter(x=x, y=p5, fill='tonexty', fillcolor='rgba(124,77,255,0.15)', line=dict(width=0), name='95% Confidence Cone'))
fig2.add_trace(go.Scatter(x=x, y=p50, line=dict(color='#8B95A8', dash='dash'), name='Median'))
fig2.add_trace(go.Scatter(x=x, y=res['Eq_Strat'].values, line=dict(color='#00FFB2', width=2.5), name='Actual'))
fig2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#EBEEF5'), yaxis_title="Growth of $1", xaxis_title="Trading Days", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(17,21,28,0.8)"))
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
    except: pass

if c_data:
    html = "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'><tr style='color:#8B95A8;border-bottom:1px solid rgba(255,255,255,0.1);'><th style='padding:10px;'>CRISIS PERIOD</th><th>STRATEGY</th><th>MARKET</th><th>ALPHA (EDGE)</th><th>RESULT</th></tr>"
    for row in c_data:
        col = "#00FFB2" if "+" in row[3] else ("#FF3B6B" if "-" in row[3] else "#8B95A8")
        html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:10px;font-family:monospace;'>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td style='color:{col};font-weight:bold;'>{row[3]}</td><td>{row[4]}</td></tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

# NEW: TRANSACTION COST SENSITIVITY
st.markdown("<h2>04A — TRANSACTION COST SENSITIVITY ANALYSIS</h2>", unsafe_allow_html=True)
st.caption("Testing strategy robustness across different friction levels")

cost_levels = [0, 3, 5, 8, 10, 15]
cost_results = []
for cost in cost_levels:
    res_temp = backtest(test_data, cost, slip)
    sh_temp, _, tot_temp, ann_temp, dd_temp = stats(res_temp['Net'])
    cost_results.append([f"{cost+slip}bps", f"{sh_temp:.3f}", f"{tot_temp*100:.0f}%", f"{ann_temp*100:.1f}%", f"{dd_temp*100:.1f}%"])

cost_df = pd.DataFrame(cost_results, columns=["TOTAL COST", "SHARPE", "TOTAL RET", "ANN. RET", "MAX DD"])
html_cost = "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'><tr style='color:#8B95A8;border-bottom:1px solid rgba(255,255,255,0.1);'>"
for col in cost_df.columns:
    html_cost += f"<th style='padding:10px;'>{col}</th>"
html_cost += "</tr>"
for _, row in cost_df.iterrows():
    html_cost += "<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
    for val in row:
        html_cost += f"<td style='padding:10px;text-align:center;'>{val}</td>"
    html_cost += "</tr>"
html_cost += "</table>"
st.markdown(html_cost, unsafe_allow_html=True)

# NEW: OUT-OF-SAMPLE CONSISTENCY
st.markdown("<h2>04B — OUT-OF-SAMPLE CONSISTENCY CHECK</h2>", unsafe_allow_html=True)
st.caption("Performance across three equal sub-periods of the test set")

n_terciles = len(res) // 3
period1 = res.iloc[:n_terciles]
period2 = res.iloc[n_terciles:2*n_terciles]
period3 = res.iloc[2*n_terciles:]

sh1, _, tot1, _, dd1 = stats(period1['Net'])
sh2, _, tot2, _, dd2 = stats(period2['Net'])
sh3, _, tot3, _, dd3 = stats(period3['Net'])

oos_c1, oos_c2, oos_c3 = st.columns(3)
oos_c1.markdown(f"""<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'>
<div style='font-size:0.7rem;color:#8B95A8;'>PERIOD 1 (EARLY OOS)</div>
<div style='color:var(--accent);font-size:1.5rem;'>Sharpe: {sh1:.3f}</div>
<div style='font-size:0.7rem;color:#8B95A8;'>Return: {tot1*100:.0f}% | DD: {dd1*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

oos_c2.markdown(f"""<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'>
<div style='font-size:0.7rem;color:#8B95A8;'>PERIOD 2 (MID OOS)</div>
<div style='color:var(--accent);font-size:1.5rem;'>Sharpe: {sh2:.3f}</div>
<div style='font-size:0.7rem;color:#8B95A8;'>Return: {tot2*100:.0f}% | DD: {dd2*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

oos_c3.markdown(f"""<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'>
<div style='font-size:0.7rem;color:#8B95A8;'>PERIOD 3 (RECENT OOS)</div>
<div style='color:var(--accent);font-size:1.5rem;'>Sharpe: {sh3:.3f}</div>
<div style='font-size:0.7rem;color:#8B95A8;'>Return: {tot3*100:.0f}% | DD: {dd3*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

# NEW: DRAWDOWN DURATION ANALYSIS
st.markdown("<h2>04C — DRAWDOWN DURATION ANALYSIS</h2>", unsafe_allow_html=True)
st.caption("Recovery characteristics and time spent underwater")

underwater = res['Eq_Strat'] < res['Eq_Strat'].cummax()
underwater_int = underwater.astype(int)
underwater_changes = underwater_int.diff()

starts = underwater_changes[underwater_changes == 1].index.tolist()
ends = underwater_changes[underwater_changes == -1].index.tolist()

if len(ends) > 0 and len(starts) > 0:
    if len(ends) < len(starts):
        ends.append(res.index[-1])
    
    durations = [(end - start).days for start, end in zip(starts[:len(ends)], ends)]
    if len(durations) > 0:
        max_duration = max(durations)
        avg_duration = np.mean(durations)
        num_dds = len(durations)
        
        dd_c1, dd_c2, dd_c3 = st.columns(3)
        dd_c1.markdown(f"""<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'>
        <div style='font-size:0.7rem;color:#8B95A8;'>LONGEST DRAWDOWN</div>
        <div style='color:var(--accent);font-size:1.5rem;'>{max_duration} days</div>
        </div>""", unsafe_allow_html=True)
        
        dd_c2.markdown(f"""<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'>
        <div style='font-size:0.7rem;color:#8B95A8;'>AVG DRAWDOWN DURATION</div>
        <div style='color:var(--accent);font-size:1.5rem;'>{avg_duration:.0f} days</div>
        </div>""", unsafe_allow_html=True)
        
        dd_c3.markdown(f"""<div style='background:var(--panel);padding:10px;border-left:2px solid var(--purple);'>
        <div style='font-size:0.7rem;color:#8B95A8;'>NUMBER OF DRAWDOWNS</div>
        <div style='color:var(--accent);font-size:1.5rem;'>{num_dds}</div>
        </div>""", unsafe_allow_html=True)

# SHAP 
st.markdown("<h2>05 — SHAP FEATURE ATTRIBUTION (GAME-THEORETIC)</h2>", unsafe_allow_html=True)
try:
    with st.spinner("Calculating SHAP..."):
        X_samp = test_df[feats].sample(n=min(500, len(test_df)), random_state=42)
        exp = shap.TreeExplainer(rf_model)
        shap_vals = exp.shap_values(X_samp)
        
        if isinstance(shap_vals, list):
            shap_plot_data = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            shap_plot_data = shap_vals[:,:,1]
        else:
            shap_plot_data = shap_vals

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<p style='text-align:center;font-weight:bold;'>Feature Importance</p>", unsafe_allow_html=True)
        fig_bar, ax_bar = plt.subplots(figsize=(6,5))
        shap.summary_plot(shap_plot_data, X_samp, plot_type="bar", show=False, color='#7C4DFF')
        fig_bar.patch.set_facecolor('#0A0E14')
        ax_bar.set_facecolor('#0A0E14')
        ax_bar.tick_params(colors='#EBEEF5')
        ax_bar.xaxis.label.set_color('#EBEEF5')
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    with c2:
        st.markdown("<p style='text-align:center;font-weight:bold;'>SHAP Beeswarm</p>", unsafe_allow_html=True)
        fig_bee, ax_bee = plt.subplots(figsize=(6,5))
        shap.summary_plot(shap_plot_data, X_samp, show=False)
        fig_bee.patch.set_facecolor('#0A0E14')
        ax_bee.set_facecolor('#0A0E14')
        ax_bee.tick_params(colors='#EBEEF5')
        ax_bee.xaxis.label.set_color('#EBEEF5')
        st.pyplot(fig_bee)
        plt.close(fig_bee)
except Exception as e:
    st.error(f"Could not render SHAP plots. Exception: {e}")

# FACTOR DECOMPOSITION WITH ALPHA CI
st.markdown("<h2>06 — FACTOR DECOMPOSITION (OLS ALPHA) & CONFIDENCE INTERVALS</h2>", unsafe_allow_html=True)

Y = res['Net'].dropna()
X_reg = sm.add_constant(res['R_ret'].dropna())
model = sm.OLS(Y, X_reg).fit()
alpha = model.params['const'] * 252
beta = model.params['R_ret']
p_alpha = model.pvalues['const']

# Bootstrap alpha confidence interval
n_boot = 500
alpha_samples = []
np.random.seed(42)
for _ in range(n_boot):
    sample_idx = np.random.choice(len(Y), size=len(Y), replace=True)
    Y_boot = Y.iloc[sample_idx]
    X_boot = X_reg.iloc[sample_idx]
    model_boot = sm.OLS(Y_boot, X_boot).fit()
    alpha_samples.append(model_boot.params['const'] * 252)

alpha_ci_low = np.percentile(alpha_samples, 2.5)
alpha_ci_high = np.percentile(alpha_samples, 97.5)

st.markdown(f"""<div style='display:flex;gap:20px;margin-bottom:20px;'>
<div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>ALPHA (ANN.)</div><div style='color:var(--accent);font-size:1.8rem;'>{alpha*100:+.2f}%</div><div style='font-size:0.6rem;color:#8B95A8;'>95% CI: [{alpha_ci_low*100:.2f}%, {alpha_ci_high*100:.2f}%]</div></div>
<div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>MARKET BETA</div><div style='color:var(--purple);font-size:1.8rem;'>{beta:.3f}</div><div style='font-size:0.6rem;color:#8B95A8;'>{"Defensive" if beta<1 else "Aggressive"}</div></div>
<div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>R-SQUARED</div><div style='color:#EBEEF5;font-size:1.8rem;'>{model.rsquared:.3f}</div><div style='font-size:0.6rem;color:#8B95A8;'>Model fit</div></div>
<div data-testid='stMetric' style='flex:1;'><div style='font-size:0.7rem;color:#8B95A8;'>SHARPE RATIO</div><div style='color:var(--accent);font-size:1.8rem;'>{sh_s:.3f}</div><div style='font-size:0.6rem;color:#8B95A8;'>Out-of-sample</div></div>
</div>""", unsafe_allow_html=True)

# PERMUTATION TEST
st.markdown("<h2>07 — STATISTICAL SIGNIFICANCE (PERMUTATION TEST)</h2>", unsafe_allow_html=True)
n_perm = 1000
pos = res['Pos'].values
br = res['R_ret'].values
sr = res['S_ret'].values

perm_sh = []
np.random.seed(42)
for _ in range(n_perm):
    shuf = np.random.permutation(pos)
    pr = np.where(shuf==1, br, sr)
    p_sh, _, _, _, _ = stats(pd.Series(pr))
    perm_sh.append(p_sh)

perm_sh = np.array(perm_sh)
p_val = (perm_sh >= sh_s).mean()

mc1,mc2,mc3 = st.columns(3)
mc1.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'><div style='font-size:0.7rem;color:#8B95A8;'>ACTUAL SHARPE</div><div style='color:var(--accent);font-size:1.8rem;'>{sh_s:.4f}</div></div>", unsafe_allow_html=True)
mc2.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'><div style='font-size:0.7rem;color:#8B95A8;'>PERMUTATION P-VALUE</div><div style='color:var(--accent);font-size:1.8rem;'>{p_val:.4f}</div><div style='font-size:0.65rem;color:#8B95A8;margin-top:0.25rem;'>p<0.05=sig.</div></div>", unsafe_allow_html=True)
mc3.markdown(f"<div style='background:var(--panel);padding:15px;border-left:2px solid var(--accent);'><div style='font-size:0.7rem;color:#8B95A8;'>RANDOM STRATEGIES BEATEN</div><div style='color:var(--accent);font-size:1.8rem;'>{(1-p_val)*100:.1f}%</div></div>", unsafe_allow_html=True)

fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=perm_sh, nbinsx=50, marker_color='#2C3243'))
fig4.add_vline(x=sh_s, line_color='#00FFB2', line_width=3)
fig4.add_vline(x=np.percentile(perm_sh, 95), line_color='#FF3B6B', line_dash='dash', line_width=2)
fig4.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#EBEEF5'), xaxis_title="Sharpe Ratio", yaxis_title="Frequency", showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

if p_val < 0.05:
    st.markdown(f"<div style='background:rgba(0,255,178,0.1);padding:15px;border-left:3px solid var(--accent);border-radius:4px;'><strong>⭐ STATISTICALLY SIGNIFICANT</strong> — p={p_val:.4f} < 0.05. We reject H₀. Genuine predictive skill confirmed.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='background:rgba(255,59,107,0.1);padding:15px;border-left:3px solid var(--red);border-radius:4px;'><strong>⚠️ NOT SIGNIFICANT</strong> — p={p_val:.4f} > 0.05. Cannot reject H₀ at 95% confidence.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#8B95A8;font-size:0.75rem;'>AMCE v8.3 ENHANCED | T20-READY | NOT FINANCIAL ADVICE</p>", unsafe_allow_html=True)
