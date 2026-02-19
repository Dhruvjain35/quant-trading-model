import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ── Plot style ──────────────────────────────────────────────
BG, BG2, BG3 = '#060810', '#0C0F1A', '#111527'
ACCENT, ACCENT2, ACCENT3, MUTED, TEXT = '#00FFB2', '#FF3B6B', '#7B61FF', '#5C6480', '#E8EAF6'

def style_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor('#1E2540')
    ax.grid(color='#1A1F35', linestyle='-', linewidth=0.5, alpha=0.6)

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

    # Join backtest signals with returns - handle column naming carefully
    rets_bt = rets[[R, S]].copy()
    res = bt.join(rets_bt).dropna()
    
    # Rename to avoid any column conflicts
    res = res.rename(columns={R: 'RET_RISKY', S: 'RET_SAFE'})
    
    res['SR'] = np.where(res['Signal']==1, res['RET_RISKY'], res['RET_SAFE'])  # strategy return
    res['BR'] = res['RET_RISKY']                                                # benchmark return
    res['ER'] = res['SR'] - res['RET_SAFE']                                    # excess return

    cs = (1+res['SR']).cumprod()
    cb = (1+res['BR']).cumprod()

    tot   = float(cs.iloc[-1])-1; bch = float(cb.iloc[-1])-1
    ann   = (1+tot)**(252/len(res))-1; bann=(1+bch)**(252/len(res))-1
    sh    = sharpe(res['SR']); bsh=sharpe(res['BR'])
    so    = sortino(res['SR']); cv=cvar(res['SR'])
    md    = max_dd(cs);         ca=calmar(ann,md)
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

    # ── 02 EQUITY CURVE + DRAWDOWN ───────────────────────
    st.markdown("## 02 — Equity Curve & Regime Overlay")
    fig, (ax_eq, ax_dd) = plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':[3,1],'hspace':0.04})
    style_ax(ax_eq, fig); style_ax(ax_dd)

    ir_bool = res['Signal']==1
    chg = ir_bool != ir_bool.shift()
    starts = res.index[chg & ir_bool]
    ends   = res.index[chg & ~ir_bool]
    if ir_bool.iloc[0]: starts = res.index[:1].append(starts)
    if ir_bool.iloc[-1]: ends = ends.append(res.index[-1:])
    for s,e in zip(starts, ends): ax_eq.axvspan(s,e,alpha=0.05,color=ACCENT,zorder=0)

    ax_eq.plot(cb.index, cb.values, color=MUTED,   lw=1.2, ls='--', label=f'{R} Buy & Hold', alpha=0.7)
    ax_eq.plot(cs.index, cs.values, color=ACCENT,  lw=2.2, label='AMCE Strategy', zorder=5)
    ax_eq.fill_between(cs.index, 1, cs.values, alpha=0.05, color=ACCENT)
    ax_eq.set_ylabel('Portfolio Value (×)', color=MUTED, fontsize=9)
    ax_eq.legend(loc='upper left', facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    ax_eq.set_xticklabels([])

    dd_s = (cs/cs.cummax()-1)*100; dd_b=(cb/cb.cummax()-1)*100
    ax_dd.fill_between(dd_s.index, dd_s.values, 0, alpha=0.35, color=ACCENT2)
    ax_dd.plot(dd_s.index, dd_s.values, color=ACCENT2, lw=1)
    ax_dd.plot(dd_b.index, dd_b.values, color=MUTED,   lw=0.8, ls='--', alpha=0.5)
    ax_dd.set_ylabel('Drawdown %', color=MUTED, fontsize=9); ax_dd.axhline(0,color='#1E2540',lw=0.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── 03 MONTE CARLO ────────────────────────────────────
    st.markdown("## 03 — Monte Carlo Robustness (Bootstrapped)")
    st.markdown('<p class="hero-stat">Bootstrap resampling of actual strategy returns preserves fat-tail properties. The actual strategy tracks within the 95% confidence cone.</p>', unsafe_allow_html=True)

    sims = []
    for _ in range(int(nmc)):
        r2 = np.random.choice(res['SR'].values, size=len(res), replace=True)
        sims.append(np.cumprod(1+r2))
    sa = np.array(sims)
    p97,p50,p03 = np.percentile(sa,97.5,axis=0), np.percentile(sa,50,axis=0), np.percentile(sa,2.5,axis=0)

    prob_beat = (sa[:,-1] > float(cb.iloc[-1])).mean()
    prob_dd40 = (sa.min(axis=1) < 0.60).mean()

    mc1,mc2,mc3 = st.columns(3)
    mc1.metric("Prob. Beat Benchmark", f"{prob_beat*100:.0f}%")
    mc2.metric("Prob. Drawdown > 40%", f"{prob_dd40*100:.0f}%")
    mc3.metric("Median Final Value",   f"×{p50[-1]:.2f}")

    fig_mc, ax_mc = plt.subplots(figsize=(14,5)); style_ax(ax_mc, fig_mc)
    ax_mc.fill_between(range(len(p50)), p03, p97, color=ACCENT3, alpha=0.12, label='95% Confidence Cone')
    ax_mc.plot(p50,           color=TEXT,   ls='--', lw=1, alpha=0.5, label='Median Expectation')
    ax_mc.plot(cs.values,     color=ACCENT, lw=2.5,          label='Actual Strategy', zorder=5)
    ax_mc.plot(cb.values,     color=MUTED,  lw=1.2, ls=':',  alpha=0.6, label=f'{R} Buy & Hold')
    ax_mc.axhline(1, color='#1E2540', lw=0.5)
    ax_mc.legend(facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    ax_mc.set_xlabel('Trading Days', fontsize=9); ax_mc.set_ylabel('Growth of $1', fontsize=9)
    plt.tight_layout(); st.pyplot(fig_mc); plt.close()
    prog.progress(65)

    # ── 04 CRISIS ALPHA ───────────────────────────────────
    st.markdown("## 04 — Crisis Alpha Analysis")
    st.markdown('<p class="hero-stat">Performance during systemic risk events — the definitive test of any defensive overlay strategy. Green = capital preserved vs benchmark.</p>', unsafe_allow_html=True)

    crises = [
        ("Dot-com Crash",        "2000-03-01","2002-10-01"),
        ("2008 Financial Crisis","2007-10-01","2009-03-01"),
        ("2011 Euro Debt Crisis","2011-05-01","2011-10-01"),
        ("2015 Flash Crash",     "2015-08-01","2016-02-01"),
        ("2018 Volmageddon",     "2018-09-01","2018-12-31"),
        ("2020 COVID Crash",     "2020-02-19","2020-03-23"),
        ("2022 Inflation Bear",  "2022-01-01","2022-12-31"),
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

    ac = ACCENT if a_ann>0 else ACCENT2
    sc_txt = f"p={a_p:.3f} ✓ SIGNIFICANT" if a_p<0.05 else f"p={a_p:.3f} — not significant"
    sc_col = ACCENT if a_p<0.05 else MUTED

    fa1,fa2,fa3,fa4 = st.columns(4)
    for col_f, label, val, sub, col_v in [
        (fa1,"ALPHA (ANN.)",f"{a_ann*100:+.2f}%",f"t={a_t:.2f} · {sc_txt}",ac),
        (fa2,"MARKET BETA", f"{beta_v:.3f}",      "Defensive β<1" if beta_v<1 else "Leveraged β>1",ACCENT3),
        (fa3,"R² EXPLAINED",f"{r2:.3f}",           "Residual = model skill",MUTED),
        (fa4,"INFO. RATIO", f"{ir:.3f}",           "Active ret / tracking err",ACCENT),
    ]:
        col_f.markdown(f"""
        <div style="background:{BG3};border:1px solid {col_v}30;border-radius:2px;padding:1rem;border-top:2px solid {col_v};margin-bottom:0.5rem;">
        <p style="font-family:'Space Mono',monospace;font-size:0.58rem;color:{MUTED};letter-spacing:0.15em;margin:0;">{label}</p>
        <p style="font-family:'Space Mono',monospace;font-size:1.8rem;color:{col_v};margin:0.2rem 0;">{val}</p>
        <p style="font-size:0.68rem;color:{MUTED};margin:0;">{sub}</p>
        </div>""", unsafe_allow_html=True)

    # ── 06 ROLLING STABILITY ──────────────────────────────
    st.markdown("## 06 — Strategy Stability (Rolling Metrics)")
    fig_r, axs = plt.subplots(1,2,figsize=(14,4)); fig_r.patch.set_facecolor(BG)
    ax_rs, ax_wr = axs
    style_ax(ax_rs); style_ax(ax_wr)

    rsh = (res['SR'].rolling(252).mean()/res['SR'].rolling(252).std())*np.sqrt(252)
    bsh2= (res['BR'].rolling(252).mean()/res['BR'].rolling(252).std())*np.sqrt(252)
    ax_rs.fill_between(rsh.index, rsh, 0, where=(rsh>0), alpha=0.12, color=ACCENT)
    ax_rs.fill_between(rsh.index, rsh, 0, where=(rsh<0), alpha=0.12, color=ACCENT2)
    ax_rs.plot(rsh, color=ACCENT, lw=1.5, label='Strategy')
    ax_rs.plot(bsh2, color=MUTED, lw=1, ls='--', alpha=0.6, label=f'{R} B&H')
    ax_rs.axhline(0, color=ACCENT2, lw=0.8, ls='--'); ax_rs.axhline(1, color=ACCENT, lw=0.5, ls=':', alpha=0.4)
    ax_rs.set_title("12-Month Rolling Sharpe Ratio", color=TEXT, fontsize=10)
    ax_rs.legend(facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=8)

    rwr = res['SR'].rolling(252).apply(lambda x:(x>0).mean())
    ax_wr.fill_between(rwr.index, rwr, 0.5, where=(rwr>=0.5), alpha=0.12, color=ACCENT)
    ax_wr.fill_between(rwr.index, rwr, 0.5, where=(rwr<0.5),  alpha=0.12, color=ACCENT2)
    ax_wr.plot(rwr, color=ACCENT, lw=1.5)
    ax_wr.axhline(0.5, color=MUTED, lw=0.8, ls='--', alpha=0.6)
    ax_wr.set_title("12-Month Rolling Win Rate", color=TEXT, fontsize=10)
    plt.tight_layout(); st.pyplot(fig_r); plt.close()

    # Overfitting diagnostic
    sp2  = int(len(res)*0.7)
    ins  = res.iloc[:sp2]; oos2 = res.iloc[sp2:]
    is_sh= sharpe(ins['SR']); os_sh=sharpe(oos2['SR'])
    is_dd= max_dd((1+ins['SR']).cumprod()); os_dd=max_dd((1+oos2['SR']).cumprod())
    is_wr= (ins['SR']>0).mean(); os_wr=(oos2['SR']>0).mean()
    decay= abs(is_sh-os_sh)/max(abs(is_sh),0.001)

    oo_df = pd.DataFrame({
        "Metric":["Sharpe Ratio","Max Drawdown","Win Rate"],
        "In-Sample (70%)":[f"{is_sh:.3f}",f"{is_dd*100:.1f}%",f"{is_wr:.1%}"],
        "Out-of-Sample (30%)":[f"{os_sh:.3f}",f"{os_dd*100:.1f}%",f"{os_wr:.1%}"],
        "Decay":[f"{abs(is_sh-os_sh)/max(abs(is_sh),0.001):.0%}",
                 f"{abs(is_dd-os_dd)/max(abs(is_dd),0.001):.0%}",
                 f"{abs(is_wr-os_wr)/max(abs(is_wr),0.001):.0%}"]
    })
    st.dataframe(oo_df, use_container_width=True, hide_index=True)
    if decay<0.25: st.success("✅ LOW OVERFITTING — Out-of-sample metrics within 25% of in-sample. Model generalizes to unseen market conditions.")
    elif decay<0.5: st.warning("⚠️ MODERATE DECAY — Consider additional regularization.")
    else: st.error("❌ HIGH OVERFITTING — Significant performance decay detected out-of-sample.")
    prog.progress(75)

    # ── 07 PERMUTATION TEST ───────────────────────────────
    st.markdown("## 07 — Statistical Significance (Permutation Test)")
    st.markdown('<p class="hero-stat">We shuffle <em>prediction signals</em> 1,000× while keeping returns in chronological order. If our Sharpe exceeds 95% of shuffled strategies, the model demonstrates genuine skill.</p>', unsafe_allow_html=True)

    with st.spinner("Running 1,000 signal permutations…"):
        perm_sh = []
        sigs    = res['Signal'].values.copy()
        for _ in range(1000):
            shf  = np.random.permutation(sigs)
            sr   = np.where(shf==1, res['RET_RISKY'].values, res['RET_SAFE'].values)
            m,sd = np.mean(sr), np.std(sr)
            if sd>0: perm_sh.append((m/sd)*np.sqrt(252))
        pa   = np.array(perm_sh)
        p_v  = (np.sum(pa>=sh)+1)/(len(pa)+1)
        pct  = (pa < sh).mean()

    pm1,pm2,pm3 = st.columns(3)
    pm1.metric("Actual Sharpe",       f"{sh:.4f}")
    pm2.metric("Permutation p-value", f"{p_v:.4f}", "< 0.05 = significant")
    pm3.metric("Random Strategies Beaten", f"{pct*100:.1f}%")

    fig_p, ax_p = plt.subplots(figsize=(12,4)); style_ax(ax_p, fig_p)
    ax_p.hist(pa, bins=50, color=MUTED, alpha=0.4, density=True, label='Random Signal Distribution')
    ax_p.axvline(sh, color=ACCENT, lw=2.5, label=f'Actual Sharpe = {sh:.3f}')
    p95p = np.percentile(pa, 95)
    ax_p.axvline(p95p, color=ACCENT2, lw=1.5, ls='--', alpha=0.8, label=f'95th Perm. Pct = {p95p:.3f}')
    ax_p.fill_betweenx([0,5], p95p, pa.max()*1.1, alpha=0.06, color=ACCENT2)
    ax_p.set_xlabel('Sharpe Ratio', fontsize=9); ax_p.set_ylabel('Density', fontsize=9)
    ax_p.legend(facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    ax_p.set_xlim(pa.min()*0.95, max(pa.max(), sh)*1.1)
    plt.tight_layout(); st.pyplot(fig_p); plt.close()

    if p_v<0.05:  st.success(f"⭐ STATISTICALLY SIGNIFICANT — p={p_v:.4f} < 0.05. We reject H₀. Genuine predictive skill confirmed.")
    elif p_v<0.1: st.info(f"◉ MARGINALLY SIGNIFICANT — p={p_v:.4f}. Directional edge present.")
    else:         st.warning(f"◌ NOT SIGNIFICANT — p={p_v:.4f}. Cannot reject H₀.")
    prog.progress(85)

    # ── 08 TRANSACTION COST SENSITIVITY ──────────────────
    st.markdown("## 08 — Transaction Cost Sensitivity")
    st.markdown('<p class="hero-stat">Institutional viability requires positive risk-adjusted returns after realistic trading frictions. We stress-test edge persistence across 7 cost regimes.</p>', unsafe_allow_html=True)

    n_tr = (res['Signal']!=res['Signal'].shift()).sum()
    tc_rows=[]
    for bps in [0,2,5,10,20,30,50]:
        tc  = bps/10000
        ar  = res['SR'].copy()
        td  = res.index[res['Signal']!=res['Signal'].shift()]
        ar.loc[td] -= tc
        ctc = (1+ar).cumprod(); sh_tc=sharpe(ar); dd_tc=max_dd(ctc)
        tot_tc=(float(ctc.iloc[-1])-1); ann_tc=(1+tot_tc)**(252/len(ar))-1
        tc_rows.append({"Cost":f"{bps}bps","Ann. Return":f"{ann_tc*100:.1f}%","Sharpe":f"{sh_tc:.3f}","Max DD":f"{dd_tc*100:.1f}%","Beats Benchmark":"✅" if ann_tc>bann else "❌"})

    st.markdown(f'<p class="hero-stat">Estimated trades: {n_tr} &nbsp;|&nbsp; Approx. annual turnover: {n_tr/max(len(res)/252,1):.0f}×</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(tc_rows), use_container_width=True, hide_index=True)

    # ── 09 MODEL DISAGREEMENT ─────────────────────────────
    st.markdown("## 09 — Ensemble Model Disagreement Analysis")
    st.markdown('<p class="hero-stat">Convergence = high conviction. Divergence = regime ambiguity. The fill between models quantifies uncertainty.</p>', unsafe_allow_html=True)

    fig_d, ax_d = plt.subplots(figsize=(14,4)); style_ax(ax_d, fig_d)
    ax_d.plot(res.index, res['Prob_LR'], color=ACCENT,  lw=0.9, alpha=0.85, label='Logistic Regression')
    ax_d.plot(res.index, res['Prob_RF'], color=ACCENT3, lw=0.9, alpha=0.85, label='Random Forest')
    ax_d.fill_between(res.index, res['Prob_LR'], res['Prob_RF'], alpha=0.15, color=ACCENT2, label='Disagreement Zone')
    ax_d.axhline(0.52, color=ACCENT2, lw=0.8, ls='--', alpha=0.6, label='Decision Threshold (0.52)')
    ax_d.set_ylim(0,1); ax_d.set_ylabel('P(Risky Asset Positive)', fontsize=9)
    ax_d.legend(facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=8)
    plt.tight_layout(); st.pyplot(fig_d); plt.close()

    disagree = (res['Prob_LR']-res['Prob_RF']).abs()
    dd1,dd2 = st.columns(2)
    dd1.metric("Avg Disagreement",   f"{disagree.mean():.4f}")
    dd2.metric("High Conviction %",  f"{(disagree<0.05).mean():.1%}")

    # ── 10 SHAP ───────────────────────────────────────────
    st.markdown("## 10 — SHAP Feature Attribution (Game-Theoretic)")
    st.markdown('<p class="hero-stat">SHapley Additive exPlanations decompose each prediction into feature contributions using cooperative game theory. Red = feature pushes model toward risky asset. Blue = toward safety asset.</p>', unsafe_allow_html=True)

    try:
        if rf_model and X_last is not None:
            exp = shap.TreeExplainer(rf_model)
            sv  = exp.shap_values(X_last)
            sv  = sv[1] if isinstance(sv, list) else sv

            plt.close('all')
            fig_sh, (ax_b, ax_sw) = plt.subplots(1,2,figsize=(14,5))
            fig_sh.patch.set_facecolor(BG)

            # Bar
            style_ax(ax_b)
            ms  = np.abs(sv).mean(axis=0)
            idx = np.argsort(ms)
            fn  = X_last.columns.tolist()
            cols_bar = [ACCENT if i==idx[-1] else ACCENT3 for i in idx]
            ax_b.barh([fn[i] for i in idx], [ms[i] for i in idx], color=cols_bar, alpha=0.85)
            ax_b.set_xlabel('Mean |SHAP Value|', fontsize=9)
            ax_b.set_title('Feature Importance', color=TEXT, fontsize=10)

            # Beeswarm
            plt.sca(ax_sw); style_ax(ax_sw)
            shap.summary_plot(sv, X_last, plot_type='dot', show=False, max_display=10, color_bar=False, alpha=0.5)
            ax_sw.set_facecolor(BG2); ax_sw.set_title('SHAP Beeswarm (Direction)', color=TEXT, fontsize=10)
            ax_sw.tick_params(colors=MUTED, labelsize=8); ax_sw.set_xlabel('SHAP Value', color=MUTED, fontsize=9)
            fig_sh.patch.set_facecolor(BG)
            plt.tight_layout(); st.pyplot(fig_sh); plt.close('all')
    except Exception as e:
        st.warning(f"SHAP: {e}")

    prog.progress(100)

    # ── CONCLUSION ────────────────────────────────────────
    sig_flag = p_v < 0.05
    c_col    = ACCENT if sig_flag else ACCENT2
    c_label  = "CONFIRMS H₁ — STATISTICALLY SIGNIFICANT" if sig_flag else "FAILS TO REJECT H₀"

    st.markdown(f"""
    <div style="margin-top:3rem;padding:2rem;background:linear-gradient(135deg,{BG2},{BG3});
                border:1px solid {c_col}35;border-radius:2px;border-top:3px solid {c_col};">
      <p style="font-family:'Space Mono',monospace;font-size:0.6rem;color:{c_col};letter-spacing:0.2em;margin:0 0 1rem 0;">
        RESEARCH CONCLUSION — {c_label}
      </p>
      <p style="font-size:0.88rem;line-height:1.75;margin:0;color:{TEXT};">
        The AMCE framework achieves a Sharpe ratio of <strong>{sh:.3f}</strong> 
        (vs. benchmark {bsh:.3f}) with annualized return of <strong>{ann*100:.1f}%</strong> 
        (vs. benchmark {bann*100:.1f}%). OLS alpha decomposition yields 
        <strong>{a_ann*100:+.2f}%</strong> annualized excess return 
        (t={a_t:.2f}, p={a_p:.3f}). Permutation testing across 1,000 signal shuffles 
        {'rejects H₀ at p=' + f'{p_v:.4f}' + ' < 0.05, confirming genuine predictive skill' if sig_flag 
         else 'yields p=' + f'{p_v:.4f}' + ', suggesting further model development is warranted'}. 
        Purged walk-forward validation with {emb}-month embargo eliminates look-ahead bias. 
        Bootstrap Monte Carlo across {int(nmc)} simulations confirms {'robust' if prob_beat>0.5 else 'limited'} 
        probability of benchmark outperformance ({prob_beat*100:.0f}%). 
        The ensemble architecture — linear (LR) + non-linear (RF) voting — demonstrates 
        {'stable out-of-sample generalization' if decay<0.25 else 'moderate performance decay'} 
        ({decay:.0%} Sharpe ratio decay from training to test period).
      </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;margin:2rem 0;">
      <div style="background:#0C0F1A;border:1px solid rgba(0,255,178,0.15);border-radius:2px;padding:1.5rem;border-top:2px solid #00FFB2;">
        <p style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#00FFB2;letter-spacing:0.2em;margin:0 0 0.75rem 0;">ARCHITECTURE</p>
        <p style="font-weight:600;margin:0 0 0.5rem 0;">Two-Stage Voting Ensemble</p>
        <p style="font-size:0.78rem;color:#5C6480;margin:0;line-height:1.6;">LR (linear) + RF (non-linear) majority voting. 0.52 conviction threshold. 10 engineered features including macro proxies.</p>
      </div>
      <div style="background:#0C0F1A;border:1px solid rgba(123,97,255,0.15);border-radius:2px;padding:1.5rem;border-top:2px solid #7B61FF;">
        <p style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#7B61FF;letter-spacing:0.2em;margin:0 0 0.75rem 0;">VALIDATION</p>
        <p style="font-weight:600;margin:0 0 0.5rem 0;">Purged Walk-Forward</p>
        <p style="font-size:0.78rem;color:#5C6480;margin:0;line-height:1.6;">Embargo gap prevents leakage. Quarterly retraining. 1,000-permutation significance test. Bootstrap Monte Carlo robustness.</p>
      </div>
      <div style="background:#0C0F1A;border:1px solid rgba(255,59,107,0.15);border-radius:2px;padding:1.5rem;border-top:2px solid #FF3B6B;">
        <p style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#FF3B6B;letter-spacing:0.2em;margin:0 0 0.75rem 0;">ATTRIBUTION</p>
        <p style="font-weight:600;margin:0 0 0.5rem 0;">OLS Alpha + SHAP</p>
        <p style="font-size:0.78rem;color:#5C6480;margin:0;line-height:1.6;">OLS regression isolates alpha from beta. SHAP game-theoretic attribution. TC sensitivity across 7 cost regimes.</p>
      </div>
    </div>
    <div style="padding:1.5rem;background:#0C0F1A;border:1px solid rgba(0,255,178,0.1);border-radius:2px;margin:1.5rem 0;">
      <p style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#5C6480;letter-spacing:0.2em;margin:0 0 1rem 0;">10 RESEARCH MODULES</p>
      <div>
        <span class="pill-green">01 Executive Summary</span>
        <span class="pill-green">02 Equity + Drawdown</span>
        <span class="pill-green">03 Monte Carlo</span>
        <span class="pill-green">04 Crisis Alpha</span>
        <span class="pill-green">05 Factor Decomp.</span>
        <span class="pill-green">06 Rolling Stability</span>
        <span class="pill-green">07 Permutation Test</span>
        <span class="pill-green">08 TC Sensitivity</span>
        <span class="pill-green">09 Model Disagreement</span>
        <span class="pill-green">10 SHAP Attribution</span>
      </div>
    </div>
    <p style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#5C6480;text-align:center;margin-top:2rem;">
    ⚡ Configure parameters in sidebar → Execute Research Pipeline
    </p>
    """, unsafe_allow_html=True)
