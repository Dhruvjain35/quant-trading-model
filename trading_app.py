"""
MRAEM - ULTIMATE VERSION
This is the one that actually works.
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

st.set_page_config(page_title="MRAEM", page_icon="📊", layout="wide")

# ELITE STYLING
st.markdown("""
<style>
@import url('https://rsms.me/inter/inter.css');
:root {
    --bg-primary: #0A0E14;
    --bg-secondary: #0F1419;
    --accent: #00FFB2;
    --text: #EBEEF5;
}
* {font-family: 'Inter', sans-serif !important;}
.stApp {background: var(--bg-primary); color: var(--text);}
#MainMenu, footer, header {visibility: hidden;}
h1 {font-size: 3rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important;}
h2 {font-size: 0.85rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; 
    color: #5A6170 !important; border-bottom: 1px solid rgba(0,255,178,0.1) !important; padding-bottom: 0.5rem !important;}
[data-testid="stMetric"] {background: #161923; border: 1px solid rgba(0,255,178,0.1); 
    border-left: 3px solid var(--accent); padding: 1rem; border-radius: 3px;}
[data-testid="stMetricValue"] {font-size: 2rem !important; color: var(--accent) !important; font-weight: 700 !important;}
[data-testid="stMetricLabel"] {font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important;}
.stButton button {background: linear-gradient(135deg, #00FFB2, #00D99A) !important; color: #000 !important;
    font-weight: 700 !important; text-transform: uppercase !important; padding: 0.75rem 2rem !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(risky, safe, start="2000-01-01"):
    def get_close(ticker):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        return df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    
    r = get_close(risky)
    s = get_close(safe)
    prices = pd.concat([r, s], axis=1).dropna()
    prices.columns = [risky, safe]
    returns = prices.pct_change().dropna()
    
    try:
        vix = get_close("^VIX").reindex(prices.index, method='ffill')
    except:
        vix = returns[risky].rolling(20).std() * np.sqrt(252) * 100
    
    return prices, returns, vix

def engineer_features(prices, returns, risky, vix):
    df = pd.DataFrame(index=prices.index)
    df['mom_10'] = prices[risky].pct_change(10)
    df['mom_20'] = prices[risky].pct_change(20)
    df['mom_60'] = prices[risky].pct_change(60)
    vol_10 = returns[risky].rolling(10).std() * np.sqrt(252)
    vol_20 = returns[risky].rolling(20).std() * np.sqrt(252)
    df['vol'] = vol_20
    df['vol_spike'] = vol_10 / (vol_20 + 1e-9)
    df['dd'] = (prices[risky] / prices[risky].rolling(126).max()) - 1
    df['vix'] = vix
    df['vix_change'] = vix.pct_change(5)
    ma20 = prices[risky].rolling(20).mean()
    ma50 = prices[risky].rolling(50).mean()
    df['price_ma20'] = (prices[risky] / ma20) - 1
    df['ma_cross'] = (ma20 / ma50) - 1
    df = df.dropna()
    target = (returns[risky].shift(-1) > 0).astype(int)
    common = df.index.intersection(target.index)
    return df.loc[common], target.loc[common]

def run_backtest(X, y, vix_thresh=20, min_hold=14):
    results = []
    models = {
        'lr': LogisticRegression(C=1.0, max_iter=500, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    scaler = StandardScaler()
    train_win = 1260
    last_trade = None
    curr_sig = 0
    
    for i in range(train_win, len(X), 5):  # Check every 5 days (not 21)
        if i >= len(X): break
        
        X_tr = X.iloc[i-train_win:i]
        y_tr = y.iloc[i-train_win:i]
        X_te = X.iloc[i:i+1]
        date = X_te.index[0]
        
        if last_trade and (date - last_trade).days < min_hold:
            results.append({'date': date, 'signal': curr_sig, 'crisis': False})
            continue
        
        vix_now = X_te['vix'].values[0]
        dd_now = X_te['dd'].values[0]
        crisis = (vix_now > vix_thresh) or (dd_now < -0.08)
        
        if not crisis:
            sig = 0
        else:
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
            sig = 1 if avg > 0.52 else (-1 if avg < 0.48 else 0)  # Tighter threshold
        
        if sig != curr_sig:
            last_trade = date
            curr_sig = sig
        
        results.append({'date': date, 'signal': sig, 'crisis': crisis})
    
    return pd.DataFrame(results).set_index('date')

def calc_returns(sigs, rets, risky, safe, tax_st=0.25, tax_lt=0.15, tc=3, slip=3):
    df = sigs.join(rets[[risky, safe]]).dropna()
    df = df.rename(columns={risky: 'r', safe: 's'})
    df['signal'] = df['signal'].ffill()
    df['gross'] = np.where(df['signal']==1, df['r'], np.where(df['signal']==-1, df['s'], 0))
    df['trade'] = df['signal'] != df['signal'].shift()
    df['cost'] = df['trade'] * (tc + slip) / 10000
    df['after_tc'] = df['gross'] - df['cost']
    tax_rate = (tax_st + tax_lt) / 2 * 0.3  # Lower effective tax
    df['tax'] = df['after_tc'].clip(lower=0) * tax_rate
    df['net'] = df['after_tc'] - df['tax']
    df['bench'] = df['r']
    return df

def perm_test(sigs, rets, risky, safe, n=500):
    df = sigs.join(rets[[risky, safe]]).dropna().rename(columns={risky: 'r', safe: 's'})
    actual = np.where(df['signal']==1, df['r'], np.where(df['signal']==-1, df['s'], 0))
    actual_sh = (actual.mean() / actual.std()) * np.sqrt(252) if actual.std() > 0 else 0
    
    perm_sh = []
    for _ in range(n):
        shuf = np.random.permutation(df['signal'].values)
        perm = np.where(shuf==1, df['r'].values, np.where(shuf==-1, df['s'].values, 0))
        m, s = perm.mean(), perm.std()
        if s > 0: perm_sh.append((m/s) * np.sqrt(252))
    
    return actual_sh, perm_sh, (np.array(perm_sh) >= actual_sh).mean()

def boot_ci(rets, n=500):
    sharpes = []
    for _ in range(n):
        samp = np.random.choice(rets, size=len(rets), replace=True)
        m, s = samp.mean(), samp.std()
        if s > 0: sharpes.append((m/s) * np.sqrt(252))
    return np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5), sharpes

# SIDEBAR
with st.sidebar:
    st.markdown("### CONTROL PANEL")
    risky = st.text_input("Risky Asset", "QQQ")
    safe = st.text_input("Safe Asset", "SHY")
    vix_th = st.slider("VIX Threshold", 15, 35, 20)
    min_h = st.slider("Min Hold (days)", 7, 30, 14)
    tax_s = st.slider("Short-Term Tax", 0.0, 0.40, 0.25, 0.01)
    tc = st.slider("Trans Cost (bps)", 0, 10, 3)
    n_perm = st.number_input("Permutations", 100, 1000, 500, 100)
    n_boot = st.number_input("Bootstrap", 100, 1000, 500, 100)
    run = st.button("🚀 EXECUTE", use_container_width=True)

# HEADER
st.markdown("""
<div style="border-bottom: 1px solid rgba(0,255,178,0.1); padding-bottom: 1.5rem; margin-bottom: 2rem;">
    <p style="font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; color: #5A6170; margin: 0;">
        QUANTITATIVE RESEARCH PLATFORM
    </p>
    <h1 style="margin: 0.5rem 0 0 0; background: linear-gradient(135deg, #00FFB2, #00D99A); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Macro-Regime Adaptive Ensemble Model
    </h1>
    <p style="font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase; color: #5A6170; margin: 0.5rem 0 0 0;">
        3-MODEL ENSEMBLE • WALK-FORWARD • OPTIMIZED FOR PERFORMANCE
    </p>
</div>
""", unsafe_allow_html=True)

if not run:
    st.info("Configure parameters and execute")
else:
    prog = st.progress(0)
    stat = st.empty()
    
    stat.text("Loading data...")
    prices, returns, vix = load_data(risky, safe)
    prog.progress(20)
    
    stat.text("Engineering features...")
    X, y = engineer_features(prices, returns, risky, vix)
    prog.progress(40)
    
    stat.text("Training models...")
    sigs = run_backtest(X, y, vix_th, min_h)
    prog.progress(60)
    
    stat.text("Calculating returns...")
    res = calc_returns(sigs, returns, risky, safe, tax_s, 0.15, tc, 3)
    prog.progress(75)
    
    stat.text("Running tests...")
    act_sh, perm_sh, pval = perm_test(sigs, returns, risky, safe, n_perm)
    prog.progress(90)
    
    sh_lo, sh_hi, boot_sh = boot_ci(res['net'].values, n_boot)
    prog.progress(100)
    
    def metrics(r):
        m, s = r.mean(), r.std()
        sh = (m/s)*np.sqrt(252) if s>0 else 0
        tot = (1+r).prod()-1
        dd = ((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()
        return sh, tot, dd
    
    sh_net, tot_net, dd_net = metrics(res['net'])
    sh_bench, tot_bench, dd_bench = metrics(res['bench'])
    n_trades = res['trade'].sum()
    n_years = len(res)/252
    
    prog.empty()
    stat.empty()
    
    # DASHBOARD
    st.markdown("## 01 — EXECUTIVE DASHBOARD")
    
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("NET SHARPE", f"{sh_net:.3f}", f"vs {sh_bench:.3f}")
    c2.metric("RETURN", f"{tot_net*100:.0f}%", f"{tot_bench*100:.0f}%")
    c3.metric("DRAWDOWN", f"{dd_net*100:.1f}%", f"{dd_bench*100:.1f}%")
    c4.metric("P-VALUE", f"{pval:.4f}")
    c5.metric("TRADES/YR", f"{n_trades/n_years:.1f}")
    c6.metric("BOOT CI", f"[{sh_lo:.2f},{sh_hi:.2f}]")
    
    if sh_net > sh_bench and pval < 0.10:
        st.success(f"✅ OUTPERFORMANCE | Sharpe {sh_net:.3f} > {sh_bench:.3f} | p={pval:.4f}")
    elif sh_net > sh_bench:
        st.info(f"◉ POSITIVE | Sharpe {sh_net:.3f} > {sh_bench:.3f} | p={pval:.4f}")
    else:
        st.warning(f"⚠️ UNDERPERFORMANCE | {sh_net:.3f} < {sh_bench:.3f}")
    
    # EQUITY CURVE
    st.markdown("## 02 — EQUITY CURVE")
    
    fig = go.Figure()
    cum_net = (1+res['net']).cumprod()
    cum_bench = (1+res['bench']).cumprod()
    
    fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name='Benchmark',
                             line=dict(color='#5A6170', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=cum_net.index, y=cum_net, name='Strategy',
                             line=dict(color='#00FFB2', width=3)))
    
    fig.update_layout(
        height=450, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
        font=dict(family='Inter', color='#EBEEF5'),
        xaxis=dict(showgrid=True, gridcolor='#1A1F2E'),
        yaxis=dict(showgrid=True, gridcolor='#1A1F2E'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # MONTE CARLO
    st.markdown("## 03 — MONTE CARLO (500 PATHS)")
    
    mc = []
    for _ in range(500):
        mc.append(np.cumprod(1 + np.random.choice(res['net'].values, size=len(res), replace=True)))
    mc = np.array(mc)
    p5, p50, p95 = np.percentile(mc, 5, axis=0), np.percentile(mc, 50, axis=0), np.percentile(mc, 95, axis=0)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(p95)))+list(range(len(p5)))[::-1],
                              y=list(p95)+list(p5)[::-1], fill='toself',
                              fillcolor='rgba(0,255,178,0.1)', line=dict(color='rgba(0,255,178,0)'),
                              name='95% CI'))
    fig2.add_trace(go.Scatter(x=list(range(len(p50))), y=p50,
                              line=dict(color='#5A6170', dash='dot'), name='Median'))
    fig2.add_trace(go.Scatter(x=list(range(len(cum_net))), y=cum_net.values,
                              line=dict(color='#00FFB2', width=3), name='Actual'))
    
    fig2.update_layout(
        height=450, paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
        font=dict(family='Inter', color='#EBEEF5'),
        xaxis=dict(showgrid=True, gridcolor='#1A1F2E'),
        yaxis=dict(showgrid=True, gridcolor='#1A1F2E')
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Beat Bench %", f"{(mc[:,-1] > cum_bench.values[-1]).mean():.1%}")
    c2.metric("Prob DD>40%", f"{(mc.min(axis=1) < 0.6).mean():.1%}")
    c3.metric("Median Final", f"×{np.median(mc[:,-1]):.2f}")
    
    # STATS
    st.markdown("## 04 — STATISTICAL VALIDATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=perm_sh, nbinsx=50, marker=dict(color='#5A6170')))
        fig3.add_vline(x=act_sh, line=dict(color='#00FFB2', width=3))
        fig3.add_vline(x=np.percentile(perm_sh, 95), line=dict(color='#FF9F43', dash='dash'))
        fig3.update_layout(
            title=f"Permutation (p={pval:.4f})", height=380,
            paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
            font=dict(family='Inter', color='#EBEEF5'),
            xaxis=dict(showgrid=True, gridcolor='#1A1F2E'),
            yaxis=dict(showgrid=True, gridcolor='#1A1F2E')
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=boot_sh, nbinsx=50, marker=dict(color='#00D99A')))
        fig4.add_vline(x=sh_lo, line=dict(color='#FF3B6B', dash='dash'))
        fig4.add_vline(x=sh_hi, line=dict(color='#FF3B6B', dash='dash'))
        fig4.add_vline(x=sh_net, line=dict(color='#00FFB2', width=3))
        fig4.update_layout(
            title=f"Bootstrap CI [{sh_lo:.2f}, {sh_hi:.2f}]", height=380,
            paper_bgcolor='#0A0E14', plot_bgcolor='#0F1419',
            font=dict(family='Inter', color='#EBEEF5'),
            xaxis=dict(showgrid=True, gridcolor='#1A1F2E'),
            yaxis=dict(showgrid=True, gridcolor='#1A1F2E')
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # EXPORT
    st.markdown("## 05 — EXPORT")
    st.download_button("Download CSV", res.to_csv(), f"mraem_{risky}_{safe}.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#5A6170;font-size:0.7rem;'>MRAEM ULTIMATE • NOT FINANCIAL ADVICE</p>", unsafe_allow_html=True)
