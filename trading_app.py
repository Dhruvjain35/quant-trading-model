"""
MRAEM - ULTIMATE VERSION
Institutional-Grade Quantitative Research Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MRAEM Terminal", page_icon="📊", layout="wide")

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
h1 {font-size: 2.5rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important;}
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
# DATA LOADING (BULLETPROOF YFINANCE)
# ==========================================
@st.cache_data(show_spinner=False)
def load_data(risky, safe, start="2005-01-01"):
    try:
        tickers = f"{risky} {safe} ^VIX"
        raw = yf.download(tickers, start=start, progress=False)
        
        # Handle modern yfinance MultiIndex output
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw['Close']
        else:
            closes = raw
            
        prices = closes[[risky, safe]].dropna()
        returns = prices.pct_change().dropna()
        vix = closes['^VIX'].reindex(prices.index).ffill()
        
        return prices, returns, vix
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None, None, None

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features(prices, returns, risky, vix):
    df = pd.DataFrame(index=prices.index)
    
    # Momentum & Trend
    df['mom_10'] = prices[risky].pct_change(10)
    df['mom_20'] = prices[risky].pct_change(20)
    df['mom_60'] = prices[risky].pct_change(60)
    
    # Moving Averages
    ma50 = prices[risky].rolling(50).mean()
    ma200 = prices[risky].rolling(200).mean()
    df['dist_ma50'] = (prices[risky] / ma50) - 1
    df['dist_ma200'] = (prices[risky] / ma200) - 1
    
    # Volatility
    vol_20 = returns[risky].rolling(20).std() * np.sqrt(252)
    df['vol'] = vol_20
    df['vix'] = vix
    df['dd'] = (prices[risky] / prices[risky].rolling(252).max()) - 1
    
    df = df.dropna()
    
    # Target: Predicting the intermediate trend (10-day forward return > 0)
    # ML is much better at this than predicting 1-day binary outcomes.
    target = (prices[risky].shift(-10) > prices[risky]).astype(int)
    common = df.index.intersection(target.index)
    
    return df.loc[common], target.loc[common]

# ==========================================
# ENSEMBLE BACKTEST ENGINE
# ==========================================
def run_backtest(X, y, vix_thresh=20, min_hold=10):
    results = []
    models = {
        'lr': LogisticRegression(C=0.1, class_weight='balanced', max_iter=500, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }
    scaler = StandardScaler()
    train_win = 1000
    last_trade_date = None
    curr_sig = 1  # DEFAULT TO LONG QQQ
    
    for i in range(train_win, len(X), 10):  # Evaluate every 2 weeks
        if i >= len(X): break
        
        X_tr, y_tr = X.iloc[i-train_win:i], y.iloc[i-train_win:i]
        X_te = X.iloc[i:i+1]
        date = X_te.index[0]
        
        # Hysteresis: Don't flip-flop too often
        if last_trade_date and (date - last_trade_date).days < min_hold:
            results.append({'date': date, 'signal': curr_sig, 'crisis': False})
            continue
            
        vix_now = X_te['vix'].values[0]
        dd_now = X_te['dd'].values[0]
        
        # DEFINE CRISIS REGIME: High VIX or in a noticeable drawdown
        crisis = (vix_now > vix_thresh) or (dd_now < -0.05)
        
        if not crisis:
            # Bull market? Just hold QQQ. Let it ride.
            sig = 1
        else:
            # Crisis? Ask the ML ensemble if it's a dip to buy, or a crash to avoid.
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            preds = []
            models['lr'].fit(X_tr_sc, y_tr)
            preds.append(models['lr'].predict_proba(X_te_sc)[0, 1])
            
            models['rf'].fit(X_tr, y_tr)
            preds.append(models['rf'].predict_proba(X_te)[0, 1])
            
            models['gb'].fit(X_tr, y_tr)
            preds.append(models['gb'].predict_proba(X_te)[0, 1])
            
            avg = np.mean(preds)
            # If models are bearish (prob < 0.5), flee to safe asset (-1). Else stay long (1).
            sig = 1 if avg >= 0.50 else -1
            
        if sig != curr_sig:
            last_trade_date = date
            curr_sig = sig
            
        results.append({'date': date, 'signal': sig, 'crisis': crisis})
        
    return pd.DataFrame(results).set_index('date')

# ==========================================
# REALISTIC ACCOUNTING
# ==========================================
def calc_returns(sigs, rets, risky, safe, tax_st=0.25, tc_bps=3):
    df = sigs.join(rets[[risky, safe]]).dropna()
    df = df.rename(columns={risky: 'r', safe: 's'})
    
    # 1 = QQQ, -1 = SHY
    df['signal'] = df['signal'].ffill()
    df['gross'] = np.where(df['signal'] == 1, df['r'], df['s'])
    
    # Costs applied ONLY when state changes
    df['trade'] = df['signal'] != df['signal'].shift(1)
    df['cost'] = df['trade'] * (tc_bps / 10000)
    df['after_tc'] = df['gross'] - df['cost']
    
    # Simplified Tax Drag on positive trades
    df['tax'] = df['after_tc'].clip(lower=0) * (tax_st * 0.1) # Proxy for blended tax impact
    df['net'] = df['after_tc'] - df['tax']
    df['bench'] = df['r']
    
    return df

# ==========================================
# STATISTICAL ROBUSTNESS
# ==========================================
def boot_ci(rets, n=500):
    sharpes = []
    for _ in range(n):
        samp = np.random.choice(rets, size=len(rets), replace=True)
        m, s = samp.mean(), samp.std()
        if s > 0: sharpes.append((m/s) * np.sqrt(252))
    return np.percentile(sharpes, 5), np.percentile(sharpes, 95), sharpes

def metrics(r):
    m, s = r.mean(), r.std()
    sh = (m/s)*np.sqrt(252) if s>0 else 0
    tot = (1+r).prod() - 1
    cum = (1+r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return sh, tot, dd

# ==========================================
# APP UI & EXECUTION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ MASTER ENGINE")
    risky = st.text_input("High-Beta Asset", "QQQ")
    safe = st.text_input("Risk-Free Asset", "SHY")
    
    st.markdown("### 🎛️ REGIME PARAMS")
    vix_th = st.slider("VIX Alert Threshold", 15, 35, 20)
    min_h = st.slider("Min Hold (days)", 5, 30, 10)
    
    st.markdown("### 💸 FRICTIONS")
    tax_s = st.slider("Short-Term Tax", 0.0, 0.40, 0.25, 0.01)
    tc = st.slider("Trans Cost (bps)", 0, 10, 3)
    
    st.markdown("### 🎲 MONTE CARLO")
    n_boot = st.number_input("Bootstrap Paths", 100, 1000, 500, 100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀 COMPILE MASTER TERMINAL", use_container_width=True)

# HEADER
st.markdown("""
<div style="padding-bottom: 1rem;">
    <p style="font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #8B95A8; margin: 0;">
        INSTITUTIONAL QUANTITATIVE RESEARCH PLATFORM
    </p>
    <h1 style="margin: 0.2rem 0 0 0; color: #EBEEF5;">
        Macro-Regime Adaptive <span style="color: #00FFB2;">Ensemble Model</span>
    </h1>
    <p style="font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; color: #8B95A8; margin: 0.5rem 0 0 0;">
        3-MODEL ENSEMBLE • LONG-BIASED W/ CRISIS EVASION • WALK-FORWARD VALIDATION
    </p>
</div>
""", unsafe_allow_html=True)

if not run:
    st.info("👈 Configure your universe parameters and click Compile.")
else:
    with st.spinner("Initializing Quant Pipeline..."):
        prices, returns, vix = load_data(risky, safe)
        
        if prices is None:
            st.stop()
            
        X, y = engineer_features(prices, returns, risky, vix)
        sigs = run_backtest(X, y, vix_th, min_h)
        res = calc_returns(sigs, returns, risky, safe, tax_s, tc)
        
        sh_lo, sh_hi, boot_sh = boot_ci(res['net'].values, n_boot)
        
        sh_net, tot_net, dd_net = metrics(res['net'])
        sh_bench, tot_bench, dd_bench = metrics(res['bench'])
        n_trades = res['trade'].sum()
        n_years = len(res)/252

    # ==========================================
    # DASHBOARD
    # ==========================================
    st.markdown("## 01 — EXECUTIVE DASHBOARD")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NET SHARPE", f"{sh_net:.3f}", f"Bench: {sh_bench:.3f}")
    c2.metric("TOTAL RETURN", f"{tot_net*100:.0f}%", f"Bench: {tot_bench*100:.0f}%")
    c3.metric("MAX DRAWDOWN", f"{dd_net*100:.1f}%", f"Bench: {dd_bench*100:.1f}%", delta_color="inverse")
    c4.metric("TRADES / YR", f"{n_trades/n_years:.1f}", f"Total Trades: {n_trades}")
    c5.metric("90% BOOT CI", f"[{sh_lo:.2f}, {sh_hi:.2f}]", "Sharpe Range")
    
    if sh_net > sh_bench:
        st.success(f"✅ **STRATEGY OUTPERFORMS** | The ML evasion logic successfully improved risk-adjusted returns (Sharpe {sh_net:.2f} vs {sh_bench:.2f}).")
    else:
        st.warning(f"⚠️ **STRATEGY UNDERPERFORMS** | The benchmark buy-and-hold outperformed the ML model in this timeframe.")

    # ==========================================
    # EQUITY CURVE
    # ==========================================
    st.markdown("## 02 — WEALTH ACCUMULATION CURVE")
    
    fig = go.Figure()
    cum_net = (1+res['net']).cumprod()
    cum_bench = (1+res['bench']).cumprod()
    
    fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=f'Buy & Hold {risky}',
                             line=dict(color='#8B95A8', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=cum_net.index, y=cum_net, name='MRAEM Strategy',
                             line=dict(color='#00FFB2', width=2.5), fill='tonexty', fillcolor='rgba(0,255,178,0.05)'))
    
    # Highlight crisis avoidance zones
    crisis_zones = res[res['signal'] == -1].index
    if len(crisis_zones) > 0:
        fig.add_trace(go.Scatter(x=crisis_zones, y=cum_net.loc[crisis_zones], mode='markers', 
                                 marker=dict(color='#FF3B6B', size=4), name='Fled to Safe Asset'))

    fig.update_layout(
        height=500, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
        font=dict(family='Inter', color='#EBEEF5'),
        xaxis=dict(showgrid=True, gridcolor='#1A1F2E', title="Date"),
        yaxis=dict(showgrid=True, gridcolor='#1A1F2E', title="Portfolio Multiple"),
        hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # MONTE CARLO STRESS TEST
    # ==========================================
    st.markdown("## 03 — MONTE CARLO ROBUSTNESS")
    
    mc = []
    # Safe numpy choice using the values array directly
    net_vals = res['net'].values
    for _ in range(n_boot):
        mc.append(np.cumprod(1 + np.random.choice(net_vals, size=len(net_vals), replace=True)))
    mc = np.array(mc)
    
    p5, p50, p95 = np.percentile(mc, 5, axis=0), np.percentile(mc, 50, axis=0), np.percentile(mc, 95, axis=0)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(p95)))+list(range(len(p5)))[::-1],
                              y=list(p95)+list(p5)[::-1], fill='toself',
                              fillcolor='rgba(0,255,178,0.1)', line=dict(color='rgba(0,255,178,0)'),
                              name='90% Confidence Interval'))
    fig2.add_trace(go.Scatter(x=list(range(len(p50))), y=p50,
                              line=dict(color='#8B95A8', dash='dot'), name='Median Expectation'))
    fig2.add_trace(go.Scatter(x=list(range(len(cum_net))), y=cum_net.values,
                              line=dict(color='#00FFB2', width=2), name='Actual Strategy Path'))
    
    fig2.update_layout(
        height=450, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
        font=dict(family='Inter', color='#EBEEF5'),
        xaxis=dict(showgrid=True, gridcolor='#1A1F2E', title="Trading Days"),
        yaxis=dict(showgrid=True, gridcolor='#1A1F2E', title="Simulated Portfolio Multiple")
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Prob. Beating Benchmark", f"{(mc[:,-1] > cum_bench.values[-1]).mean():.1%}")
    mc2.metric("Prob. Drawdown > 30%", f"{(mc.min(axis=1) < 0.7).mean():.1%}")
    mc3.metric("Median Final Wealth", f"{np.median(mc[:,-1]):.2f}x")
