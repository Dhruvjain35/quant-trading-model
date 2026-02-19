import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & ELITE STYLING
# ============================================================
st.set_page_config(
    page_title="AMCE | Adaptive Macro-Conditional Ensemble",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COLOR PALETTE FOR ALTAIR
c_bg = '#0C0F1A'
c_accent = '#00FFB2'
c_accent2 = '#FF3B6B'
c_accent3 = '#7B61FF'
c_text = '#E8EAF6'
c_muted = '#5C6480'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:#060810; --bg2:#0C0F1A; --bg3:#111527;
    --accent:#00FFB2; --accent2:#FF3B6B; --accent3:#7B61FF;
    --text:#E8EAF6; --muted:#5C6480; --border:rgba(0,255,178,0.15);
}
.stApp { background-color: var(--bg); font-family: 'Syne', sans-serif; color: var(--text); }
.stApp > header { display: none; }
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
.block-container { padding: 2rem 2.5rem; max-width: 100%; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg3) 0%, rgba(17,21,39,0.8) 100%);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 1.2rem 1.5rem; position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family: 'Syne', sans-serif !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: var(--muted) !important; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; }
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1.2rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0.5rem !important; margin-top: 2.5rem !important; }
.stButton button {
    background: linear-gradient(135deg, var(--accent) 0%, #00CC8E 100%) !important;
    color: #000 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 2px !important; width: 100%; font-size: 0.8rem !important;
}
.stTextInput input {
    background: var(--bg3) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-family: 'Space Mono', monospace !important;
    border-radius: 2px !important;
}
table { font-family: 'Space Mono', monospace; font-size: 0.78rem; border-collapse: collapse; width: 100%; }
th { color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.62rem; border-bottom: 1px solid var(--border); padding: 0.7rem 1rem; text-align: left; }
td { padding: 0.55rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.03); }
.section-label { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: var(--accent); letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.25rem; }
.hero-stat { font-family: 'Syne', sans-serif; font-size: 0.82rem; color: var(--muted); line-height: 1.6; margin-bottom: 1rem; }
.pill-green { display:inline-block; background:rgba(0,255,178,0.08); border:1px solid var(--accent); color:var(--accent); font-family:'Space Mono',monospace; font-size:0.6rem; padding:2px 10px; border-radius:100px; letter-spacing:0.08em; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Helper for Altair Theme ─────────────────────────────────
def neon_theme():
    return {
        'config': {
            'view': {'stroke': 'transparent'},
            'axis': {
                'domainColor': '#1E2540', 'gridColor': '#1A1F35',
                'labelColor': c_muted, 'titleColor': c_muted,
                'tickColor': '#1E2540', 'gridOpacity': 0.6
            },
            'legend': {'labelColor': c_text, 'titleColor': c_muted},
            'background': c_bg
        }
    }
alt.themes.register('neon', neon_theme)
alt.themes.enable('neon')

# ============================================================
# DATA & FEATURES
# ============================================================
@st.cache_data(show_spinner=False)
def get_price(ticker, start="2000-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    elif 'Close' in df.columns:
        df = df['Close']
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df

def make_features(prices, rets, r):
    df = pd.DataFrame(index=prices.index)
    df['Mom_1M']      = prices[r].pct_change(21)
    df['Mom_3M']      = prices[r].pct_change(63)
    df['Mom_6M']      = prices[r].pct_change(126)
    df['Mom_12M']     = prices[r].pct_change(252)
    vol1              = rets[r].rolling(21).std() * np.sqrt(252)
    vol3              = rets[r].rolling(63).std() * np.sqrt(252)
    df['Vol_1M']      = vol1
    df['Vol_Regime']  = vol1 / (vol3 + 1e-9)
    delta = prices[r].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI']         = 100 - 100/(1 + gain/(loss+1e-9))
    df['VIX_Proxy']   = vol1 * 100
    df['Rate_Stress'] = prices.iloc[:,1].pct_change(21) * -1
    df['Yield_Trend'] = prices.iloc[:,1].pct_change(63) * -1
    df['Rel_Str']     = prices[r].pct_change(63) - prices.iloc[:,1].pct_change(63)
    ma50              = prices[r].rolling(50).mean()
    df['Price_MA']    = (prices[r] / ma50) - 1
    df = df.dropna()
    tgt = (rets[r].shift(-1) > 0).astype(int)
    idx = df.index.intersection(tgt.index)
    return df.loc[idx], tgt.loc[idx]

# ============================================================
# ENSEMBLE
# ============================================================
def run_ensemble(X, y, gap):
    results, last_rf, last_Xtr = [], None, None
    lr = LogisticRegression(C=0.5, solver='liblinear', max_iter=500)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=42)
    sc = StandardScaler()
    for i in range(1260, len(X), 63):
        te = i - gap
        if te < 252: continue
        Xtr, ytr = X.iloc[:te], y.iloc[:te]
        end = min(i+63, len(X))
        Xte = X.iloc[i:end]
        if Xte.empty: break
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        lr.fit(Xtr_s, ytr); rf.fit(Xtr, ytr)
        p_lr = lr.predict_proba(Xte_s)[:,1]
        p_rf = rf.predict_proba(Xte)[:,1]
        avg  = (p_lr + p_rf)/2
        results.append(pd.DataFrame({'Signal':(avg>0.52).astype(int),'Prob_LR':p_lr,'Prob_RF':p_rf}, index=Xte.index))
        last_rf, last_Xtr = rf, Xtr
    return (pd.concat(results) if results else pd.DataFrame()), last_rf, last_Xtr

# ── Metric helpers ──────────────────────────────────────────
def sharpe(r, f=252): m,s=r.mean(),r.std(); return (m/s)*np.sqrt(f) if s>0 else 0
def sortino(r, f=252): m=r.mean(); d=r[r<0].std(); return (m/d)*np.sqrt(f) if d>0 else 0
def cvar(r, a=0.05): c=r.quantile(a); return r[r<=c].mean()
def max_dd(c): return (c/c.cummax()-1).min()
def calmar(a,d): return abs(a/d) if d<0 else 0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="section-label">Research Terminal v2.0</p>', unsafe_allow_html=True)
    st.markdown("### Model Controls")
    R  = st.text_input("High-Beta Asset", "QQQ")
    S  = st.text_input("Risk-Free Asset",  "SHY")
    emb= st.slider("Purged Embargo (Months)", 1, 12, 2)
    nmc= st.number_input("Monte Carlo Sims", 100, 1000, 500, step=100)
    st.markdown("---")
    st.markdown('<p class="hero-stat" style="font-size:0.72rem;">Purged walk-forward validation · Voting ensemble · SHAP attribution · Permutation testing · Factor decomposition · Bootstrap Monte Carlo</p>', unsafe_allow_html=True)
    run = st.button("⚡ Execute Research Pipeline")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="border-bottom:1px solid rgba(0,255,178,0.15);padding-bottom:1.5rem;margin-bottom:2rem;">
  <p class="section-label">Quantitative Research Lab</p>
  <h1 style="margin:0;background:linear-gradient(135deg,#00FFB2,#7B61FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.6rem;">
    Adaptive Macro-Conditional Ensemble
  </h1>
  <p style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#5C6480;margin-top:0.5rem;letter-spacing:0.15em;">
    AMCE FRAMEWORK &nbsp;·&nbsp; PURGED WALK-FORWARD &nbsp;·&nbsp; ENSEMBLE VOTING &nbsp;·&nbsp; STATISTICAL VALIDATION
  </p>
</div>
<div style="background:linear-gradient(135deg,rgba(123,97,255,0.07),rgba(0,255,178,0.04));border:1px solid rgba(123,97,255,0.25);border-radius:2px;padding:1.25rem 1.5rem;margin-bottom:2rem;">
  <p style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#7B61FF;letter-spacing:0.2em;margin:0 0 0.75rem 0;">RESEARCH HYPOTHESIS</p>
  <p style="font-size:0.85rem;margin:0 0 0.4rem 0;"><strong>H₀ (Null):</strong> Macro-conditional regime signals provide no statistically significant improvement over passive equity exposure.</p>
  <p style="font-size:0.85rem;margin:0 0 0.4rem 0;"><strong>H₁ (Alternative):</strong> Integrating VIX dynamics and yield curve signals with purged walk-forward validation generates positive crisis alpha and statistically significant risk-adjusted outperformance.</p>
  <p style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#5C6480;margin:0;">Test: Signal permutation (n=1,000) &nbsp;|&nbsp; Threshold: p &lt; 0.05 &nbsp;|&nbsp; Alpha via OLS on excess returns</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PIPELINE
# ============================================================
if run:
    prog = st.progress(0)
    stat = st.empty()

    stat.markdown('<p class="hero-stat">⟳ Ingesting market data…</p>', unsafe_allow_html=True)
    try:
        pr = get_price(R); ps = get_price(S)
        prices = pd.concat([pr, ps], axis=1).dropna(); prices.columns = [R, S]
        rets   = prices.pct_change().dropna()
    except Exception as e:
        st.error(f"Data error: {e}"); st.stop()
    prog.progress(10)

    stat.markdown('<p class="hero-stat">⟳ Engineering 10 factors…</p>', unsafe_allow_html=True)
    feats, tgt = make_features(prices, rets, R)
    prog.progress(20)

    stat.markdown('<p class="hero-stat">⟳ Running purged walk-forward ensemble…</p>', unsafe_allow_html=True)
    bt, rf_model, X_last = run_ensemble(feats, tgt, emb*21)
    prog.progress(50)

    # Join backtest signals with returns
    rets_bt = rets[[R, S]].copy()
    res = bt.join(rets_bt).dropna()
    
    # Rename
    res = res.rename(columns={R: 'RET_RISKY', S: 'RET_SAFE'})
    
    res['SR'] = np.where(res['Signal']==1, res['RET_RISKY'], res['RET_SAFE'])  # strategy return
    res['BR'] = res['RET_RISKY']                                               # benchmark return
    res['ER'] = res['SR'] - res['RET_SAFE']                                    # excess return

    cs = (1+res['SR']).cumprod()
    cb = (1+res['BR']).cumprod()

    tot   = float(cs.iloc[-1])-1; bch = float(cb.iloc[-1])-1
    ann   = (1+tot)**(252/len(res))-1; bann=(1+bch)**(252/len(res))-1
    sh    = sharpe(res['SR']); bsh=sharpe(res['BR'])
    so    = sortino(res['SR']); cv=cvar(res['SR'])
    md    = max_dd(cs);          ca=calmar(ann,md)
    wr    = (res['SR']>0).mean()
    ir    = sharpe(res['ER'])
    prog.progress(60); stat.empty()

    # ── 01 EXECUTIVE SUMMARY ────────────────────────────
    st.markdown("## 01 — Executive Risk Summary")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Sharpe Ratio",  f"{sh:.3f}",       f"Bench: {bsh:.3f}")
    c2.metric("Sortino Ratio", f"{so:.3f}",       "Downside-adj.")
    c3.metric("Total Return",  f"{tot*100:.0f}%", f"Bench: {bch*100:.0f}%")
    c4.metric("Ann. Return",   f"{ann*100:.1f}%", f"Bench: {bann*100:.1f}%")
    c5.metric("Max Drawdown",  f"{md*100:.1f}%",  f"Calmar: {ca:.2f}")
    c6.metric("CVaR (95%)",    f"{cv*100:.2f}%",  f"Win: {wr:.1%}")

    # ── 02 EQUITY CURVE + DRAWDOWN (ALTAIR) ─────────────
    st.markdown("## 02 — Equity Curve & Regime Overlay")
    
    # Prepare Data for Altair
    eq_data = pd.DataFrame({
        'Date': cs.index,
        'AMCE Strategy': cs.values,
        f'{R} Buy & Hold': cb.values
    }).melt('Date', var_name='Asset', value_name='Value')

    # Chart
    c_eq = alt.Chart(eq_data).mark_line().encode(
        x='Date:T',
        y=alt.Y('Value:Q', title='Portfolio Value (x)'),
        color=alt.Color('Asset', scale=alt.Scale(domain=['AMCE Strategy', f'{R} Buy & Hold'], range=[c_accent, c_muted])),
        strokeDash=alt.condition(alt.datum.Asset == f'{R} Buy & Hold', alt.value([5, 5]), alt.value([0]))
    ).properties(height=300)

    # Drawdown
    dd_data = pd.DataFrame({
        'Date': cs.index,
        'Drawdown': (cs/cs.cummax()-1)
    })
    
    c_dd = alt.Chart(dd_data).mark_area(opacity=0.3, color=c_accent2).encode(
        x='Date:T',
        y=alt.Y('Drawdown:Q', axis=alt.Axis(format='%')),
    ).properties(height=100)

    st.altair_chart(c_eq & c_dd, use_container_width=True)

    # ── 03 MONTE CARLO (ALTAIR) ─────────────────────────
    st.markdown("## 03 — Monte Carlo Robustness (Bootstrapped)")
    
    sims = []
    for _ in range(int(nmc)):
        r2 = np.random.choice(res['SR'].values, size=len(res), replace=True)
        sims.append(np.cumprod(1+r2))
    sa = np.array(sims)
    p97, p50, p03 = np.percentile(sa,97.5,axis=0), np.percentile(sa,50,axis=0), np.percentile(sa,2.5,axis=0)

    prob_beat = (sa[:,-1] > float(cb.iloc[-1])).mean()
    prob_dd40 = (sa.min(axis=1) < 0.60).mean()

    mc1,mc2,mc3 = st.columns(3)
    mc1.metric("Prob. Beat Benchmark", f"{prob_beat*100:.0f}%")
    mc2.metric("Prob. Drawdown > 40%", f"{prob_dd40*100:.0f}%")
    mc3.metric("Median Final Value",   f"×{p50[-1]:.2f}")

    # MC Data
    days = np.arange(len(p50))
    mc_df = pd.DataFrame({'Day': days, 'Median': p50, 'Lower': p03, 'Upper': p97})
    
    mc_cone = alt.Chart(mc_df).mark_area(opacity=0.2, color=c_accent3).encode(
        x='Day', y='Lower', y2='Upper'
    )
    mc_line = alt.Chart(mc_df).mark_line(color=c_text, strokeDash=[5,5]).encode(
        x='Day', y='Median'
    )
    
    st.altair_chart((mc_cone + mc_line).properties(height=300), use_container_width=True)
    prog.progress(65)

    # ── 04 CRISIS ALPHA ───────────────────────────────────
    st.markdown("## 04 — Crisis Alpha Analysis")
    crises = [
        ("Dot-com Crash",    "2000-03-01","2002-10-01"),
        ("2008 Fin. Crisis", "2007-10-01","2009-03-01"),
        ("2011 Euro Crisis", "2011-05-01","2011-10-01"),
        ("2015 Flash Crash", "2015-08-01","2016-02-01"),
        ("2018 Volmageddon", "2018-09-01","2018-12-31"),
        ("2020 COVID Crash", "2020-02-19","2020-03-23"),
        ("2022 Inflation",   "2022-01-01","2022-12-31"),
    ]
    rows=[]
    for nm,s,e in crises:
        sub = res.loc[(res.index>=s)&(res.index<=e)]
        if len(sub)<5: continue
        sp=(1+sub['SR']).cumprod().iloc[-1]-1; bp=(1+sub['BR']).cumprod().iloc[-1]-1; ap=sp-bp
        col = '#00FFB2' if ap>0 else '#FF3B6B'
        rows.append({"Crisis Period":nm,"Strategy":f"{sp*100:.1f}%","Market":f"{bp*100:.1f}%",
                     "Alpha (Edge)":f'<span style="color:{col};font-weight:700;">{ap*100:+.1f}%</span>',
                     "Result":"✅ Preserved" if ap>0 else "⚠️ Corr. Spike"})
    if rows:
        st.markdown(pd.DataFrame(rows).to_html(escape=False,index=False), unsafe_allow_html=True)

    # ── 05 FACTOR DECOMPOSITION ───────────────────────────
    st.markdown("## 05 — Factor Decomposition (OLS Alpha)")
    Xols = sm.add_constant(res['BR'].values)
    try:
        ols  = sm.OLS(res['ER'].values, Xols).fit()
        a_d  = ols.params[0]; beta_v=ols.params[1]
        a_ann= (1+a_d)**252-1; a_p=ols.pvalues[0]; a_t=ols.tvalues[0]; r2=ols.rsquared
    except: a_ann,beta_v,a_p,a_t,r2 = 0,1,1,0,0

    ac = c_accent if a_ann>0 else c_accent2
    sc_txt = f"p={a_p:.3f} ✓ SIGNIFICANT" if a_p<0.05 else f"p={a_p:.3f} — not significant"
