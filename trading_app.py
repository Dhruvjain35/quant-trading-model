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
# AMCE v4.0 - CRISIS-FOCUSED + MOMENTUM FILTER
# Goal: Beat benchmark after taxes by reducing trade frequency
# ============================================================

st.set_page_config(
    page_title="AMCE v4.0 | Crisis-Focused Strategy",
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
[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid rgba(0,255,178,0.15); }
.block-container { padding: 2rem 2.5rem; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg3), rgba(17,21,39,0.8));
    border: 1px solid rgba(0,255,178,0.15); border-radius: 2px; padding: 1.2rem 1.5rem;
}
[data-testid="stMetricValue] { font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family: 'Syne', sans-serif !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: var(--muted) !important; }
h2 { font-family: 'Syne', sans-serif !important; font-size: 1.2rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--muted) !important; border-bottom: 1px solid rgba(0,255,178,0.15) !important; padding-bottom: 0.5rem !important; margin-top: 2.5rem !important; }
.stButton button {
    background: linear-gradient(135deg, var(--accent), #00CC8E) !important;
    color: #000 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border: none !important; width: 100%;
}
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
# VIX DATA DOWNLOAD
# ============================================================
@st.cache_data(show_spinner=False)
def get_vix():
    """Download actual VIX data"""
    try:
        vix = yf.download("^VIX", start="2000-01-01", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close']
        elif 'Close' in vix.columns:
            vix = vix['Close']
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        return vix
    except:
        return None

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

def make_features(prices, rets, r, vix_data=None):
    df = pd.DataFrame(index=prices.index)
    
    # Momentum features
    df['Mom_20D']     = prices[r].pct_change(20)
    df['Mom_60D']     = prices[r].pct_change(60)
    df['Mom_120D']    = prices[r].pct_change(120)
    
    # Volatility
    vol20             = rets[r].rolling(20).std() * np.sqrt(252)
    vol60             = rets[r].rolling(60).std() * np.sqrt(252)
    df['Vol_20D']     = vol20
    df['Vol_Regime']  = vol20 / (vol60 + 1e-9)
    
    # Crisis indicators
    high126 = prices[r].rolling(126).max()
    df['Dist_6M_High'] = (prices[r] / high126) - 1
    
    high252 = prices[r].rolling(252).max()
    df['Dist_1Y_High'] = (prices[r] / high252) - 1
    
    # Real VIX if available
    if vix_data is not None:
        vix_aligned = vix_data.reindex(df.index, method='ffill')
        df['VIX'] = vix_aligned
        df['VIX_20D_MA'] = df['VIX'].rolling(20).mean()
    else:
        df['VIX_Proxy'] = vol20 * 100
        df['VIX_20D_MA'] = df['VIX_Proxy'].rolling(20).mean()
    
    # Safe asset signals
    df['Safe_Mom']    = prices.iloc[:,1].pct_change(20)
    df['Yield_Stress'] = prices.iloc[:,1].pct_change(60) * -1
    
    # Moving averages
    ma50  = prices[r].rolling(50).mean()
    ma200 = prices[r].rolling(200).mean()
    df['Price_MA50']  = (prices[r] / ma50) - 1
    df['Price_MA200'] = (prices[r] / ma200) - 1
    df['MA50_MA200']  = (ma50 / ma200) - 1
    
    # RSI
    delta = prices[r].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI'] = 100 - 100/(1 + gain/(loss+1e-9))
    
    df = df.dropna()
    tgt = (rets[r].shift(-1) > 0).astype(int)
    idx = df.index.intersection(tgt.index)
    return df.loc[idx], tgt.loc[idx]

# ============================================================
# CRISIS-FOCUSED ENSEMBLE
# ============================================================
def run_crisis_ensemble(X, y, prices, gap, vix_thresh=25, min_hold_days=30, 
                       crisis_only=True, momentum_confirm=True):
    """
    Crisis-focused strategy with minimum hold periods and momentum confirmation.
    """
    results = []
    
    lr = LogisticRegression(C=0.5, solver='liblinear', max_iter=500)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    sc = StandardScaler()
    
    last_trade_date = None
    current_position = 0  # 0=cash, 1=risky, -1=safe
    
    for i in range(1260, len(X), 21):  # Check monthly instead of every 63 days
        te = i - gap
        if te < 252: continue
        
        Xtr, ytr = X.iloc[:te], y.iloc[:te]
        
        # Current date
        current_date = X.index[i]
        
        # Check if we can trade (minimum hold period)
        if last_trade_date is not None:
            days_since_trade = (current_date - last_trade_date).days
            if days_since_trade < min_hold_days:
                # Hold current position
                if current_position != 0:
                    results.append({
                        'date': current_date,
                        'Signal': current_position,
                        'Prob': 0.5,
                        'Crisis': False,
                        'Held': True
                    })
                continue
        
        # Crisis detection
        if 'VIX' in X.columns:
            current_vix = X.loc[current_date, 'VIX']
            in_crisis = current_vix > vix_thresh
        else:
            current_vix = X.loc[current_date, 'VIX_Proxy']
            in_crisis = current_vix > vix_thresh
        
        # Distance from high (alternative crisis indicator)
        dist_high = X.loc[current_date, 'Dist_6M_High']
        drawdown_crisis = dist_high < -0.10  # 10% off highs
        
        in_crisis = in_crisis or drawdown_crisis
        
        # If crisis-only mode and not in crisis, hold cash
        if crisis_only and not in_crisis:
            results.append({
                'date': current_date,
                'Signal': 0,
                'Prob': 0.5,
                'Crisis': False,
                'Held': False
            })
            current_position = 0
            continue
        
        # Train models
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(X.iloc[i:i+1])
        
        lr.fit(Xtr_s, ytr)
        rf.fit(Xtr, ytr)
        gb.fit(Xtr, ytr)
        
        # Get predictions
        p_lr = lr.predict_proba(Xte_s)[0, 1]
        p_rf = rf.predict_proba(X.iloc[i:i+1])[0, 1]
        p_gb = gb.predict_proba(X.iloc[i:i+1])[0, 1]
        
        # Require 2/3 model agreement (voting)
        votes = [p_lr > 0.55, p_rf > 0.55, p_gb > 0.55]
        vote_count = sum(votes)
        
        if vote_count >= 2:
            signal = 1  # Buy risky
        elif sum([p < 0.45 for p in [p_lr, p_rf, p_gb]]) >= 2:
            signal = -1  # Buy safe
        else:
            signal = 0  # Cash
        
        # Momentum confirmation (optional)
        if momentum_confirm and signal == 1:
            mom_20d = X.loc[current_date, 'Mom_20D']
            if mom_20d < -0.05:  # Don't buy if strong downtrend
                signal = 0
        
        # Only trade if signal changed
        if signal != current_position:
            last_trade_date = current_date
            current_position = signal
        
        avg_prob = (p_lr + p_rf + p_gb) / 3
        
        results.append({
            'date': current_date,
            'Signal': signal,
            'Prob': avg_prob,
            'Crisis': in_crisis,
            'Held': False
        })
    
    df_results = pd.DataFrame(results).set_index('date')
    return df_results, gb, Xtr

# ============================================================
# REALITY-ADJUSTED RETURNS
# ============================================================
def apply_realistic_costs_v2(res, R, S, tax_rate_short=0.35, tax_rate_long=0.20,
                             tc_bps=5, slippage_bps=5, execution_lag_days=1):
    """
    Improved tax calculation with long-term vs short-term distinction
    """
    res_real = res.copy()
    
    # Forward-fill signals
    res_real['Signal'] = res_real['Signal'].ffill()
    
    # Execution lag
    res_real['Signal_Exec'] = res_real['Signal'].shift(execution_lag_days)
    res_real = res_real.dropna()
    
    # Gross returns
    res_real['SR_Gross'] = np.where(
        res_real['Signal_Exec'] == 1, 
        res_real['RET_RISKY'],
        np.where(res_real['Signal_Exec'] == -1, res_real['RET_SAFE'], 0)
    )
    
    # Transaction costs
    trade_mask = res_real['Signal_Exec'] != res_real['Signal_Exec'].shift()
    res_real['TC_Cost'] = 0.0
    res_real.loc[trade_mask, 'TC_Cost'] = (tc_bps + slippage_bps) / 10000
    
    res_real['SR_AfterTC'] = res_real['SR_Gross'] - res_real['TC_Cost']
    
    # Tax calculation with holding period tracking
    res_real['Position_Entry'] = res_real.index[trade_mask]
    res_real['Position_Entry'] = res_real['Position_Entry'].ffill()
    
    res_real['Hold_Days'] = (res_real.index - res_real['Position_Entry']).days
    res_real['Is_LongTerm'] = res_real['Hold_Days'] > 365
    
    # Apply appropriate tax rate
    res_real['Taxable_Gain'] = res_real['SR_AfterTC'].clip(lower=0)
    res_real['Tax_Rate'] = np.where(res_real['Is_LongTerm'], tax_rate_long, tax_rate_short)
    res_real['Tax_Drag'] = res_real['Taxable_Gain'] * res_real['Tax_Rate']
    
    # Only apply tax on exits
    res_real['Tax_Applied'] = 0.0
    res_real.loc[trade_mask, 'Tax_Applied'] = res_real.loc[trade_mask, 'Tax_Drag']
    
    res_real['SR_Net'] = res_real['SR_AfterTC'] - res_real['Tax_Applied']
    
    return res_real

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p style="font-family:Space Mono;font-size:0.6rem;color:#00FFB2;letter-spacing:0.2em;">CRISIS-FOCUSED v4.0</p>', unsafe_allow_html=True)
    st.markdown("### Strategy Controls")
    R  = st.text_input("High-Beta Asset", "QQQ")
    S  = st.text_input("Risk-Free Asset", "SHY")
    emb = st.slider("Purged Embargo (Months)", 1, 12, 2)
    
    st.markdown("### Crisis Filters")
    vix_thresh = st.slider("VIX Crisis Threshold", 15, 40, 25, 1)
    min_hold = st.slider("Minimum Hold Period (Days)", 7, 90, 30, 7)
    crisis_only = st.checkbox("Trade Only During Crises", value=True)
    momentum_conf = st.checkbox("Require Momentum Confirmation", value=True)
    
    st.markdown("### Cost Assumptions")
    tax_short = st.slider("Short-Term Tax", 0.0, 0.50, 0.35, 0.01)
    tax_long  = st.slider("Long-Term Tax", 0.0, 0.30, 0.20, 0.01)
    tc_bps    = st.slider("Transaction Cost (bps)", 0, 20, 5, 1)
    slip_bps  = st.slider("Slippage (bps)", 0, 20, 5, 1)
    
    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem;color:#5C6480;line-height:1.6;">Strategy: Only trade during market stress (VIX>25 or 10%+ drawdown). Hold minimum 30 days. Require 2/3 model agreement.</p>', unsafe_allow_html=True)
    
    run = st.button("🔥 Run Crisis-Focused Strategy")

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div style="border-bottom:1px solid rgba(0,255,178,0.15);padding-bottom:1.5rem;margin-bottom:2rem;">
  <p style="font-family:Space Mono;font-size:0.6rem;color:#00FFB2;letter-spacing:0.2em;margin:0;">AMCE v4.0</p>
  <h1 style="margin:0;background:linear-gradient(135deg,#00FFB2,#7B61FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.6rem;">
    Crisis-Focused Tactical Allocation
  </h1>
  <p style="font-family:Space Mono;font-size:0.7rem;color:#5C6480;margin-top:0.5rem;letter-spacing:0.15em;">
    LOW FREQUENCY &nbsp;·&nbsp; CRISIS-ONLY &nbsp;·&nbsp; 30-DAY MINIMUM HOLD &nbsp;·&nbsp; 2/3 MODEL VOTING
  </p>
</div>
""", unsafe_allow_html=True)

if run:
    prog = st.progress(0)
    stat = st.empty()
    
    stat.markdown('<p style="font-size:0.82rem;color:#5C6480;">⟳ Downloading market data + VIX…</p>', unsafe_allow_html=True)
    try:
        pr = get_price(R)
        ps = get_price(S)
        vix_data = get_vix()
        
        prices = pd.concat([pr, ps], axis=1).dropna()
        prices.columns = [R, S]
        rets = prices.pct_change().dropna()
        
        if vix_data is not None:
            st.success("✓ Real VIX data loaded")
        else:
            st.warning("⚠ Using VIX proxy (realized volatility)")
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()
    prog.progress(15)
    
    stat.markdown('<p style="font-size:0.82rem;color:#5C6480;">⟳ Engineering features with crisis indicators…</p>', unsafe_allow_html=True)
    feats, tgt = make_features(prices, rets, R, vix_data)
    prog.progress(30)
    
    stat.markdown('<p style="font-size:0.82rem;color:#5C6480;">⟳ Running crisis-focused ensemble…</p>', unsafe_allow_html=True)
    bt, gb_model, X_last = run_crisis_ensemble(
        feats, tgt, prices, emb*21, 
        vix_thresh=vix_thresh, 
        min_hold_days=min_hold,
        crisis_only=crisis_only,
        momentum_confirm=momentum_conf
    )
    prog.progress(50)
    
    # Join returns
    rets_bt = rets[[R, S]].copy()
    res = bt.join(rets_bt).dropna()
    res = res.rename(columns={R: 'RET_RISKY', S: 'RET_SAFE'})
    res['BR'] = res['RET_RISKY']
    
    stat.markdown('<p style="font-size:0.82rem;color:#5C6480;">⟳ Applying tax-aware return calculations…</p>', unsafe_allow_html=True)
    res_real = apply_realistic_costs_v2(res, R, S, tax_short, tax_long, tc_bps, slip_bps, 1)
    prog.progress(70)
    
    # Metrics
    cs_gross = (1 + res_real['SR_Gross']).cumprod()
    cs_tc    = (1 + res_real['SR_AfterTC']).cumprod()
    cs_net   = (1 + res_real['SR_Net']).cumprod()
    cb       = (1 + res_real['BR']).cumprod()
    
    def calc_metrics(r):
        m, s = r.mean(), r.std()
        sh = (m/s)*np.sqrt(252) if s>0 else 0
        tot = (1+r).prod()-1
        ann = (1+tot)**(252/len(r))-1
        dd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
        return sh, tot, ann, dd
    
    sh_g, tot_g, ann_g, dd_g = calc_metrics(res_real['SR_Gross'])
    sh_t, tot_t, ann_t, dd_t = calc_metrics(res_real['SR_AfterTC'])
    sh_n, tot_n, ann_n, dd_n = calc_metrics(res_real['SR_Net'])
    sh_b, tot_b, ann_b, dd_b = calc_metrics(res_real['BR'])
    
    prog.progress(90); stat.empty()
    
    # RESULTS
    st.markdown("## Performance: Crisis-Focused vs Buy & Hold")
    
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("NET Sharpe", f"{sh_n:.3f}", f"vs {sh_b:.3f} bench")
    c2.metric("NET Return", f"{tot_n*100:.0f}%", f"vs {tot_b*100:.0f}%")
    c3.metric("Max Drawdown", f"{dd_n*100:.1f}%", f"vs {dd_b*100:.1f}%")
    
    n_trades = (res_real['Signal_Exec'] != res_real['Signal_Exec'].shift()).sum()
    n_years = len(res_real) / 252
    c4.metric("Trades/Year", f"{n_trades/n_years:.1f}", f"Total: {n_trades}")
    
    # Comparison table
    comp_df = pd.DataFrame({
        "Metric": ["Sharpe", "Total Return", "Annual Return", "Max DD"],
        "Gross": [f"{sh_g:.3f}", f"{tot_g*100:.0f}%", f"{ann_g*100:.1f}%", f"{dd_g*100:.1f}%"],
        "After TC": [f"{sh_t:.3f}", f"{tot_t*100:.0f}%", f"{ann_t*100:.1f}%", f"{dd_t*100:.1f}%"],
        "NET (After Tax)": [f"{sh_n:.3f}", f"{tot_n*100:.0f}%", f"{ann_n*100:.1f}%", f"{dd_n*100:.1f}%"],
        "Benchmark": [f"{sh_b:.3f}", f"{tot_b*100:.0f}%", f"{ann_b*100:.1f}%", f"{dd_b*100:.1f}%"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Equity curve
    st.markdown("## Equity Curve Comparison")
    fig, ax = plt.subplots(figsize=(14,6))
    style_ax(ax, fig)
    
    ax.plot(cs_net.index, cs_net.values, color=ACCENT, lw=2.5, label='Crisis Strategy (NET)', zorder=5)
    ax.plot(cb.index, cb.values, color=MUTED, lw=1.5, ls=':', alpha=0.7, label=f'{R} Buy & Hold')
    ax.fill_between(cs_net.index, 1, cs_net.values, alpha=0.05, color=ACCENT)
    
    # Mark crisis periods
    crisis_dates = res[res['Crisis'] == True].index
    for cd in crisis_dates[::5]:  # Mark every 5th crisis day to avoid clutter
        ax.axvline(cd, color=ACCENT2, alpha=0.03, lw=0.5)
    
    ax.set_ylabel('Portfolio Value (×)', fontsize=9)
    ax.legend(loc='upper left', facecolor=BG3, labelcolor=TEXT, edgecolor='#1E2540', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Tax breakdown
    st.markdown("## Tax Analysis")
    
    n_short = (res_real['Is_LongTerm'] == False).sum()
    n_long  = (res_real['Is_LongTerm'] == True).sum()
    
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Short-Term Periods", f"{n_short}", f"{tax_short*100:.0f}% tax")
    tc2.metric("Long-Term Periods", f"{n_long}", f"{tax_long*100:.0f}% tax")
    tc3.metric("Avg Hold Days", f"{res_real['Hold_Days'].median():.0f}")
    
    prog.progress(100)
    
    # VERDICT
    if sh_n > sh_b:
        verdict = "✅ EDGE SURVIVES REALITY"
        color = ACCENT
        msg = f"The crisis-focused strategy achieves NET Sharpe {sh_n:.3f} vs benchmark {sh_b:.3f}, proving the edge survives real-world costs. By trading only {n_trades/n_years:.1f}× per year during high-VIX periods, transaction costs and taxes are minimized while capturing crisis alpha."
    else:
        verdict = "⚠️ STILL NEEDS OPTIMIZATION"
        color = ACCENT2
        msg = f"NET Sharpe {sh_n:.3f} vs benchmark {sh_b:.3f}. The strategy reduced trades to {n_trades/n_years:.1f}/year but costs still outweigh the edge. Consider: higher VIX threshold, longer minimum holds, or different asset pair."
    
    st.markdown(f"""
    <div style="margin-top:2rem;padding:2rem;background:linear-gradient(135deg,{BG2},{BG3});
                border:1px solid {color}35;border-radius:2px;border-top:3px solid {color};">
      <p style="font-family:Space Mono;font-size:0.6rem;color:{color};letter-spacing:0.2em;margin:0 0 1rem 0;">
        {verdict}
      </p>
      <p style="font-size:0.88rem;line-height:1.75;margin:0;color:{TEXT};">
        {msg}
      </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Configure crisis filters and click 'Run Crisis-Focused Strategy' to test")
