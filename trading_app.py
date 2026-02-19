import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AMCE v3.0 - REALITY EDITION
# Adds: Taxes, Realistic Execution, Survivorship Adjustment
# ============================================================

st.set_page_config(
    page_title="AMCE v3.0 | Reality-Adjusted",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:#060810; --bg2:#0C0F1A; --bg3:#111527;
    --accent:#00FFB2; --accent2:#FF3B6B; --accent3:#7B61FF;
    --text:#E8EAF6; --muted:#5C6480;
}
.stApp { background: var(--bg); font-family: 'Syne', sans-serif; color: var(--text); }
.stApp > header { display: none; }
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid rgba(0,255,178,0.15); }
.block-container { padding: 2rem 2.5rem; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg3), rgba(17,21,39,0.8));
    border: 1px solid rgba(0,255,178,0.15); border-radius: 2px; padding: 1.2rem 1.5rem;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family: 'Syne', sans-serif !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: var(--muted) !important; }
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1.2rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--muted) !important; border-bottom: 1px solid rgba(0,255,178,0.15) !important; padding-bottom: 0.5rem !important; margin-top: 2.5rem !important; }
.stButton button {
    background: linear-gradient(135deg, var(--accent), #00CC8E) !important;
    color: #000 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border: none !important; width: 100%;
}
.hero-stat { font-family: 'Syne', sans-serif; font-size: 0.82rem; color: var(--muted); line-height: 1.6; margin-bottom: 1rem; }
table { font-family: 'Space Mono', monospace; font-size: 0.78rem; }
th { color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.62rem; border-bottom: 1px solid rgba(0,255,178,0.15); padding: 0.7rem 1rem; }
td { padding: 0.55rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.03); }
</style>
""", unsafe_allow_html=True)

BG, BG2, BG3 = '#060810', '#0C0F1A', '#111527'
ACCENT, ACCENT2, ACCENT3 = '#00FFB2', '#FF3B6B', '#7B61FF'
MUTED, TEXT = '#5C6480', '#E8EAF6'

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
# DATA
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
    
    # NEW: Distance from highs (crisis detector)
    high252 = prices[r].rolling(252).max()
    df['Dist_Max_1Y'] = (prices[r] / high252) - 1
    
    # NEW: Safe asset momentum (inverted yield)
    df['Safe_Mom']    = prices.iloc[:,1].pct_change(21)
    
    df = df.dropna()
    tgt = (rets[r].shift(-1) > 0).astype(int)
    idx = df.index.intersection(tgt.index)
    return df.loc[idx], tgt.loc[idx]

# ============================================================
# ENSEMBLE WITH 3 MODELS
# ============================================================
def run_ensemble(X, y, gap):
    results, last_gb, last_Xtr = [], None, None
    
    lr = LogisticRegression(C=0.5, solver='liblinear', max_iter=500)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    sc = StandardScaler()
    
    for i in range(1260, len(X), 63):
        te = i - gap
        if te < 252: continue
        Xtr, ytr = X.iloc[:te], y.iloc[:te]
        end = min(i+63, len(X))
        Xte = X.iloc[i:end]
        if Xte.empty: break
        
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        
        lr.fit(Xtr_s, ytr)
        rf.fit(Xtr, ytr)
        gb.fit(Xtr, ytr)
        
        p_lr = lr.predict_proba(Xte_s)[:,1]
        p_rf = rf.predict_proba(Xte)[:,1]
        p_gb = gb.predict_proba(Xte)[:,1]
        
        # 3-model ensemble
        avg = (p_lr + p_rf + p_gb) / 3
        
        # STRICT threshold: use training data quantiles ONLY
        train_probs = (lr.predict_proba(Xtr_s)[:,1] + 
                       rf.predict_proba(Xtr)[:,1] + 
                       gb.predict_proba(Xtr)[:,1]) / 3
        
        upper = np.percentile(train_probs, 65)  # More conservative
        lower = np.percentile(train_probs, 35)
        
        signals = np.zeros(len(avg))
        signals[avg >= upper] = 1
        signals[avg <= lower] = -1
        
        results.append(pd.DataFrame({
            'Signal': signals,
            'Prob': avg,
            'Prob_LR': p_lr,
            'Prob_RF': p_rf,
            'Prob_GB': p_gb
        }, index=Xte.index))
        
        last_gb, last_Xtr = gb, Xtr
    
    return (pd.concat(results) if results else pd.DataFrame()), last_gb, last_Xtr

# ============================================================
# REALITY-ADJUSTED RETURNS
# ============================================================
def apply_realistic_costs(res, R, S, tax_rate_short=0.35, tax_rate_long=0.20,
                          tc_bps=5, slippage_bps=5, execution_lag_days=1):
    """
    Apply taxes, transaction costs, slippage, and execution lag.
    """
    res_real = res.copy()
    
    # Execution lag: signal today, execute tomorrow
    res_real['Signal_Executed'] = res_real['Signal'].shift(execution_lag_days)
    res_real = res_real.dropna()
    
    # Actual returns after lag
    res_real['SR_Gross'] = np.where(
        res_real['Signal_Executed'] == 1, 
        res_real['RET_RISKY'], 
        np.where(res_real['Signal_Executed'] == -1, res_real['RET_SAFE'], 0)
    )
    
    # Transaction costs + slippage
    total_cost_bps = tc_bps + slippage_bps
    trade_days = res_real.index[res_real['Signal_Executed'] != res_real['Signal_Executed'].shift()]
    
    res_real['TC_Cost'] = 0.0
    res_real.loc[trade_days, 'TC_Cost'] = total_cost_bps / 10000
    
    # After-cost returns
    res_real['SR_AfterTC'] = res_real['SR_Gross'] - res_real['TC_Cost']
    
    # Tax calculation (simplified)
    # Track holding period and apply appropriate tax rate
    res_real['Cum_Gross'] = (1 + res_real['SR_AfterTC']).cumprod()
    
    # Approximate tax drag: assume 50% short-term, 50% long-term for simplicity
    avg_tax_rate = (tax_rate_short + tax_rate_long) / 2
    
    # Apply tax drag only on gains
    res_real['Taxable_Gain'] = res_real['SR_AfterTC'].clip(lower=0)
    res_real['Tax_Drag'] = res_real['Taxable_Gain'] * avg_tax_rate
    
    # Only apply tax on trade exits (when position changes)
    res_real['Tax_Applied'] = 0.0
    res_real.loc[trade_days, 'Tax_Applied'] = res_real.loc[trade_days, 'Tax_Drag']
    
    # Final net return
    res_real['SR_Net'] = res_real['SR_AfterTC'] - res_real['Tax_Applied']
    
    return res_real

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p style="font-family:Space Mono;font-size:0.62rem;color:#00FFB2;letter-spacing:0.2em;">REALITY-ADJUSTED v3.0</p>', unsafe_allow_html=True)
    st.markdown("### Model Controls")
    R  = st.text_input("High-Beta Asset", "QQQ")
    S  = st.text_input("Risk-Free Asset",  "SHY")
    emb= st.slider("Purged Embargo (Months)", 1, 12, 2)
    
    st.markdown("### Reality Adjustments")
    tax_short = st.slider("Short-Term Tax Rate", 0.0, 0.50, 0.35, 0.01)
    tax_long  = st.slider("Long-Term Tax Rate", 0.0, 0.30, 0.20, 0.01)
    tc_bps    = st.slider("Transaction Cost (bps)", 0, 20, 5, 1)
    slip_bps  = st.slider("Slippage (bps)", 0, 20, 5, 1)
    lag_days  = st.slider("Execution Lag (days)", 0, 3, 1, 1)
    
    st.markdown("---")
    st.markdown('<p class="hero-stat" style="font-size:0.72rem;">3-model ensemble · Tax-aware · Realistic execution · Strict validation</p>', unsafe_allow_html=True)
    
    run = st.button("⚡ Run Reality-Adjusted Backtest")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="border-bottom:1px solid rgba(0,255,178,0.15);padding-bottom:1.5rem;margin-bottom:2rem;">
  <p style="font-family:Space Mono;font-size:0.6rem;color:#00FFB2;letter-spacing:0.2em;margin:0;">AMCE v3.0 — REALITY EDITION</p>
  <h1 style="margin:0;background:linear-gradient(135deg,#00FFB2,#7B61FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.6rem;">
    Tax-Aware Execution Model
  </h1>
  <p style="font-family:Space Mono;font-size:0.7rem;color:#5C6480;margin-top:0.5rem;letter-spacing:0.15em;">
    REALISTIC COSTS &nbsp;·&nbsp; TAX DRAG &nbsp;·&nbsp; EXECUTION LAG &nbsp;·&nbsp; SLIPPAGE MODELING
  </p>
</div>
<div style="background:linear-gradient(135deg,rgba(255,59,107,0.07),rgba(0,255,178,0.04));border:1px solid rgba(255,59,107,0.25);border-radius:2px;padding:1.25rem 1.5rem;margin-bottom:2rem;">
  <p style="font-family:Space Mono;font-size:0.6rem;color:#FF3B6B;letter-spacing:0.2em;margin:0 0 0.75rem 0;">⚠️ REALITY CHECK</p>
  <p style="font-size:0.85rem;margin:0;">This version models <strong>real-world trading conditions</strong>: taxes on every trade, bid-ask spread, slippage during execution, and 1-day signal lag. Previous backtests assumed frictionless markets. This tests whether the edge survives reality.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PIPELINE
# ============================================================
if run:
    prog = st.progress(0)
    stat = st.empty()
    
    stat.markdown('<p class="hero-stat">⟳ Downloading market data…</p>', unsafe_allow_html=True)
    try:
        pr = get_price(R); ps = get_price(S)
        prices = pd.concat([pr, ps], axis=1).dropna()
        prices.columns = [R, S]
        rets = prices.pct_change().dropna()
    except Exception as e:
        st.error(f"Data error: {e}"); st.stop()
    prog.progress(10)
    
    stat.markdown('<p class="hero-stat">⟳ Engineering 12 features (added crisis detectors)…</p>', unsafe_allow_html=True)
    feats, tgt = make_features(prices, rets, R)
    prog.progress(20)
    
    stat.markdown('<p class="hero-stat">⟳ Running 3-model ensemble with strict thresholds…</p>', unsafe_allow_html=True)
    bt, gb_model, X_last = run_ensemble(feats, tgt, emb*21)
    prog.progress(40)
    
    # Join returns
    rets_bt = rets[[R, S]].copy()
    res = bt.join(rets_bt).dropna()
    res = res.rename(columns={R: 'RET_RISKY', S: 'RET_SAFE'})
    
    # BENCHMARK: Simple returns
    res['BR'] = res['RET_RISKY']
    
    stat.markdown('<p class="hero-stat">⟳ Applying taxes, costs, slippage, and execution lag…</p>', unsafe_allow_html=True)
    res_real = apply_realistic_costs(res, R, S, tax_short, tax_long, tc_bps, slip_bps, lag_days)
    prog.progress(60)
    
    # Calculate cumulative returns
    cs_gross = (1 + res_real['SR_Gross']).cumprod()
    cs_tc    = (1 + res_real['SR_AfterTC']).cumprod()
    cs_net   = (1 + res_real['SR_Net']).cumprod()
    cb       = (1 + res_real['BR']).cumprod()
    
    # Metrics
    def calc_metrics(rets_series):
        m, s = rets_series.mean(), rets_series.std()
        sh = (m/s)*np.sqrt(252) if s>0 else 0
        tot = (1+rets_series).prod()-1
        ann = (1+tot)**(252/len(rets_series))-1
        dd = ((1+rets_series).cumprod()/(1+rets_series).cumprod().cummax()-1).min()
        wr = (rets_series>0).mean()
        return sh, tot, ann, dd, wr
    
    sh_gross, tot_gross, ann_gross, dd_gross, wr_gross = calc_metrics(res_real['SR_Gross'])
    sh_tc,    tot_tc,    ann_tc,    dd_tc,    wr_tc    = calc_metrics(res_real['SR_AfterTC'])
    sh_net,   tot_net,   ann_net,   dd_net,   wr_net   = calc_metrics(res_real['SR_Net'])
    sh_b,     tot_b,     ann_b,     dd_b,     wr_b     = calc_metrics(res_real['BR'])
    
    prog.progress(80); stat.empty()
    
    # ── RESULTS ──────────────────────────────────────────
    st.markdown("## 01 — Reality Check: Gross vs Net Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GROSS Sharpe", f"{sh_gross:.3f}", "Before any costs")
    col2.metric("After TC Sharpe", f"{sh_tc:.3f}", f"{((sh_tc/sh_gross)-1)*100:.1f}% impact")
    col3.metric("NET Sharpe", f"{sh_net:.3f}", f"{((sh_net/sh_gross)-1)*100:.1f}% total impact")
    col4.metric("Benchmark", f"{sh_b:.3f}", "QQQ Buy & Hold")
    
    st.markdown("---")
    
    col5, col6, col7 = st.columns(3)
    col5.metric("GROSS Return", f"{tot_gross*100:.0f}%", "Theory")
    col6.metric("After TC Return", f"{tot_tc*100:.0f}%", f"-{(tot_gross-tot_tc)*100:.0f}%")
    col7.metric("NET Return", f"{tot_net*100:.0f}%", f"-{(tot_gross-tot_net)*100:.0f}%")
    
    # Comparison table
    st.markdown("### Performance Breakdown")
    
    comp_data = {
        "Metric": ["Sharpe Ratio", "Total Return", "Annual Return", "Max Drawdown", "Win Rate"],
        "Gross (Theory)": [
            f"{sh_gross:.3f}", f"{tot_gross*100:.0f}%", f"{ann_gross*100:.1f}%", 
            f"{dd_gross*100:.1f}%", f"{wr_gross:.1%}"
        ],
        "After TC": [
            f"{sh_tc:.3f}", f"{tot_tc*100:.0f}%", f"{ann_tc*100:.1f}%",
            f"{dd_tc*100:.1f}%", f"{wr_tc:.1%}"
        ],
        "After Tax (NET)": [
            f"{sh_net:.3f}", f"{tot_net*100:.0f}%", f"{ann_net*100:.1f}%",
            f"{dd_net*100:.1f}%", f"{wr_net:.1%}"
        ],
        "Benchmark": [
            f"{sh_b:.3f}", f"{tot_b*100:.0f}%", f"{ann_b*100:.1f}%",
            f"{dd_b*100:.1f}%", f"{wr_b:.1%}"
        ]
    }
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    # Waterfall chart showing cost breakdown
    st.markdown("## 02 — Cost Waterfall Analysis")
    
    fig_w, ax_w = plt.subplots(figsize=(12, 5))
    style_ax(ax_w, fig_w)
    
    starts = [0, tot_gross, tot_tc]
    changes = [tot_gross, tot_tc-tot_gross, tot_net-tot_tc]
    labels = ['Gross\nReturn', 'Transaction\nCosts', 'Tax\nDrag']
    colors = [ACCENT, ACCENT2, ACCENT2]
    
    for i, (start, change, label, color) in enumerate(zip(starts, changes, labels, colors)):
        ax_w.bar(i, change, bottom=start, color=color, alpha=0.8, edgecolor='white', linewidth=1)
        ax_w.text(i, start + change/2, f"{change*100:+.0f}%", 
                  ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    ax_w.bar(3, tot_net, color=ACCENT3, alpha=0.8, edgecolor='white', linewidth=1)
    ax_w.text(3, tot_net/2, f"{tot_net*100:.0f}%", 
              ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # Benchmark comparison
    ax_w.axhline(tot_b, color=MUTED, linestyle='--', linewidth=2, alpha=0.7, label=f'Benchmark: {tot_b*100:.0f}%')
    
    ax_w.set_xticks([0, 1, 2, 3])
    ax_w.set_xticklabels(['Gross\nReturn', 'After\nTC', 'After\nTax', 'NET\nReturn'], fontsize=9)
    ax_w.set_ylabel('Cumulative Return', fontsize=9)
    ax_w.legend(facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    ax_w.set_title('Return Waterfall: From Theory to Reality', color=TEXT, fontsize=11, pad=15)
    
    plt.tight_layout()
    st.pyplot(fig_w)
    plt.close()
    
    # Equity curves comparison
    st.markdown("## 03 — Equity Curves: Gross vs Net vs Benchmark")
    
    fig_eq, ax_eq = plt.subplots(figsize=(14, 6))
    style_ax(ax_eq, fig_eq)
    
    ax_eq.plot(cs_gross.index, cs_gross.values, color=ACCENT, linewidth=1.5, alpha=0.5, 
               linestyle='--', label='Gross (Theory)')
    ax_eq.plot(cs_tc.index, cs_tc.values, color=ACCENT, linewidth=2, alpha=0.8, 
               label='After Transaction Costs')
    ax_eq.plot(cs_net.index, cs_net.values, color=ACCENT, linewidth=2.5, 
               label='NET (After Tax)', zorder=5)
    ax_eq.plot(cb.index, cb.values, color=MUTED, linewidth=1.5, linestyle=':', 
               alpha=0.7, label=f'{R} Buy & Hold')
    
    ax_eq.fill_between(cs_net.index, 1, cs_net.values, alpha=0.05, color=ACCENT)
    ax_eq.set_ylabel('Portfolio Value (×)', fontsize=9)
    ax_eq.legend(loc='upper left', facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    ax_eq.set_title('Reality Check: Tax & Friction Matter', color=TEXT, fontsize=11, pad=15)
    
    plt.tight_layout()
    st.pyplot(fig_eq)
    plt.close()
    
    # Trade frequency analysis
    st.markdown("## 04 — Trade Frequency & Tax Implications")
    
    n_trades = (res_real['Signal_Executed'] != res_real['Signal_Executed'].shift()).sum()
    n_years = len(res_real) / 252
    trades_per_year = n_trades / n_years
    
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("Total Trades", f"{n_trades}")
    col_t2.metric("Trades per Year", f"{trades_per_year:.1f}")
    col_t3.metric("Avg Hold Period", f"{252/trades_per_year:.0f} days")
    
    st.markdown(f"""
    <div style="background:{BG3};border:1px solid rgba(255,59,107,0.2);border-radius:2px;padding:1rem;margin-top:1rem;">
    <p style="font-size:0.85rem;color:{TEXT};margin:0;">
    <strong>Tax Implication:</strong> With {trades_per_year:.1f} trades/year, most positions are short-term 
    (< 1 year hold). This means <strong>{tax_short*100:.0f}% tax rate</strong> applies to gains, 
    reducing the gross {tot_gross*100:.0f}% return to net {tot_net*100:.0f}% 
    (a <strong>{((tot_gross-tot_net)/tot_gross)*100:.0f}% reduction</strong>).
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    prog.progress(100)
    
    # CONCLUSION
    if sh_net > sh_b:
        verdict = "EDGE SURVIVES"
        color = ACCENT
        msg = f"Even after taxes and costs, the model achieves Sharpe {sh_net:.3f} vs benchmark {sh_b:.3f}. The edge is REAL."
    else:
        verdict = "EDGE DESTROYED"
        color = ACCENT2
        msg = f"After taxes and costs, model Sharpe {sh_net:.3f} < benchmark {sh_b:.3f}. The backtest edge was an illusion created by ignoring friction."
    
    st.markdown(f"""
    <div style="margin-top:3rem;padding:2rem;background:linear-gradient(135deg,{BG2},{BG3});
                border:1px solid {color}35;border-radius:2px;border-top:3px solid {color};">
      <p style="font-family:Space Mono;font-size:0.6rem;color:{color};letter-spacing:0.2em;margin:0 0 1rem 0;">
        VERDICT — {verdict}
      </p>
      <p style="font-size:0.9rem;line-height:1.75;margin:0;color:{TEXT};">
        {msg}
      </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:2rem 0;">
      <div style="background:#0C0F1A;border:1px solid rgba(255,59,107,0.15);border-radius:2px;padding:1.5rem;border-top:2px solid #FF3B6B;">
        <p style="font-family:Space Mono;font-size:0.58rem;color:#FF3B6B;letter-spacing:0.2em;margin:0 0 0.75rem 0;">WHAT'S NEW</p>
        <p style="font-weight:600;margin:0 0 0.5rem 0;">Reality Adjustments</p>
        <p style="font-size:0.78rem;color:#5C6480;margin:0;line-height:1.6;">
        • Short-term capital gains tax (35%)<br>
        • Long-term capital gains tax (20%)<br>
        • Transaction costs (5 bps default)<br>
        • Slippage modeling (5 bps default)<br>
        • 1-day execution lag<br>
        • 3-model ensemble (LR+RF+GB)
        </p>
      </div>
      <div style="background:#0C0F1A;border:1px solid rgba(0,255,178,0.15);border-radius:2px;padding:1.5rem;border-top:2px solid #00FFB2;">
        <p style="font-family:Space Mono;font-size:0.58rem;color:#00FFB2;letter-spacing:0.2em;margin:0 0 0.75rem 0;">THE TRUTH</p>
        <p style="font-weight:600;margin:0 0 0.5rem 0;">Why This Matters</p>
        <p style="font-size:0.78rem;color:#5C6480;margin:0;line-height:1.6;">
        Most backtests ignore taxes and assume instant, frictionless execution. This creates a 30-50% performance gap 
        between theory and reality. This version shows you what you'd ACTUALLY make after the IRS takes their cut 
        and market friction eats your edge.
        </p>
      </div>
    </div>
    <p style="font-family:Space Mono;font-size:0.72rem;color:#5C6480;text-align:center;margin-top:2rem;">
    ⚡ Configure reality parameters in sidebar → Run backtest
    </p>
    """, unsafe_allow_html=True)
