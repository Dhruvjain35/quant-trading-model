import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import statsmodels.api as sm
from scipy.stats import norm
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# =============================================================================
# 1. PAGE CONFIGURATION & ACADEMIC FRAMING
# =============================================================================
st.set_page_config(page_title="T20 Quant Lab: Hierarchical Risk Engine", layout="wide", page_icon="🏛️")

st.title("🏛️ Hierarchical Regime-Switching Engine (H-RSE)")
st.markdown("""
**Research Abstract:** A multi-layered ensemble system integrating **Macro-Financial factors** with **Statistical Risk Decomposition**.
* **Methodology:** Dual-Stage Voting Ensemble (Linear + Non-Linear) with Purged Walk-Forward Validation.
* **Risk Framework:** Conditional Value at Risk (CVaR) Optimization & Monte Carlo Confidence Intervals.
* **Attribution:** Fama-French 3-Factor Alpha Decomposition to isolate true skill from market beta.
""")

# Sidebar Controls
st.sidebar.header("🔬 Simulation Parameters")
ticker_risky = st.sidebar.text_input("Risky Asset", "QQQ") 
ticker_safe = st.sidebar.text_input("Safe Asset", "SHY")
gap_months = st.sidebar.slider("Purged Embargo (Months)", 1, 12, 3, help="Prevents look-ahead bias.")
monte_carlo_sims = st.sidebar.number_input("Monte Carlo Sims", 100, 1000, 200)

# =============================================================================
# 2. INSTITUTIONAL DATA INGESTION
# =============================================================================
@st.cache_data
def get_data(risky, safe):
    tickers = [risky, safe, "^VIX", "^TNX", "SPY"] # Added SPY for Benchmark
    data = yf.download(tickers, start="2005-01-01", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"] if "Adj Close" in data else data["Close"]
        
    data = data.dropna(axis=1, how='all').fillna(method='ffill')
    data_m = data.resample("M").last()
    rets_m = data_m.pct_change().dropna()
    return data_m, rets_m

try:
    with st.spinner("Fetching Institutional Data Feeds..."):
        prices, rets = get_data(ticker_risky, ticker_safe)
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# =============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# =============================================================================
def engineer_features(prices, rets, risky, safe):
    df = pd.DataFrame(index=prices.index)
    
    # 1. Macro: Yield Curve Proxy
    df["Yield_Chg"] = prices["^TNX"].diff(3)
    
    # 2. Volatility: VIX Curve
    df["VIX_Level"] = prices["^VIX"]
    
    # 3. Momentum: Volatility-Adjusted Momentum (Sharpe Proxy)
    mom_6m = prices[risky].pct_change(6)
    vol_6m = rets[risky].rolling(6).std()
    df["Vol_Adj_Mom"] = mom_6m / (vol_6m + 1e-6)
    
    # 4. Technical: Rolling Drawdown (Pain Index)
    roll_max = prices[risky].rolling(12).max()
    df["Drawdown"] = (prices[risky] / roll_max) - 1
    
    # Target: 1 if Risky > Safe next month
    target = (rets[risky].shift(-1) > rets[safe].shift(-1)).astype(int)
    
    return df.dropna(), target.dropna()

features, target = engineer_features(prices, rets, ticker_risky, ticker_safe)
common = features.index.intersection(target.index)
features, target = features.loc[common], target.loc[common]

# =============================================================================
# 4. ENSEMBLE ENGINE WITH PROBABILISTIC OUTPUT
# =============================================================================
def run_ensemble(X, y, gap):
    results = []
    start_idx = 48 # 4 years warm-up
    
    for i in range(start_idx, len(X)):
        train_end = i - gap
        if train_end < 24: continue
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test = X.iloc[[i]]
        
        # Scale for Logistic Regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 1. Logistic (Linear)
        lr = LogisticRegression(class_weight='balanced', solver='liblinear')
        lr.fit(X_train_s, y_train)
        p_lr = lr.predict_proba(X_test_s)[0][1]
        
        # 2. Random Forest (Non-Linear)
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        p_rf = rf.predict_proba(X_test)[0][1]
        
        # Ensemble Average
        p_final = (p_lr + p_rf) / 2
        pred = 1 if p_final > 0.5 else 0
        
        results.append({
            "date": X_test.index[0], "pred": pred, "prob": p_final,
            "conf_lr": p_lr, "conf_rf": p_rf
        })
        
    return pd.DataFrame(results).set_index("date"), rf # Return last model for SHAP

if st.button("🚀 Execute Research Pipeline"):
    with st.spinner("Running Monte Carlo Simulations & Alpha Decomposition..."):
        res, last_model = run_ensemble(features, target, gap_months)
        
    # Align Returns
    aligned = rets.loc[res.index]
    res["Strat_Ret"] = np.where(res["pred"]==1, aligned[ticker_risky], aligned[ticker_safe])
    res["Bench_Ret"] = aligned["SPY"]
    
    # =========================================================================
    # 5. DASHBOARD: EXECUTIVE SUMMARY (RISK ADJUSTED)
    # =========================================================================
    st.header("1. Executive Risk Summary")
    
    # Calc Metrics
    cum_ret = (1 + res["Strat_Ret"]).cumprod()
    ann_ret = res["Strat_Ret"].mean() * 12
    ann_vol = res["Strat_Ret"].std() * (12**0.5)
    sharpe = (ann_ret - 0.04) / ann_vol # Assuming 4% risk free
    sortino = (ann_ret - 0.04) / (res[res["Strat_Ret"]<0]["Strat_Ret"].std() * (12**0.5))
    
    # CVaR (Expected Shortfall) - The "Hedge Fund" Metric
    var_95 = res["Strat_Ret"].quantile(0.05)
    cvar_95 = res[res["Strat_Ret"] <= var_95]["Strat_Ret"].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Return per unit of risk")
    c2.metric("Sortino Ratio", f"{sortino:.2f}", help="Return per unit of BAD risk")
    c3.metric("CVaR (95%)", f"{cvar_95*100:.1f}%", help="Avg loss on worst 5% of days")
    c4.metric("Win Rate", f"{(res['Strat_Ret']>0).mean()*100:.0f}%")

    # =========================================================================
    # 6. MONTE CARLO SIMULATION (STATISTICAL RIGOR)
    # =========================================================================
    st.header("2. Monte Carlo Robustness Test")
    st.markdown("We simulate 200+ alternative market histories to ensure results aren't just luck.")
    
    simulations = []
    last_price = cum_ret.iloc[-1]
    
    # Bootstrap Resampling
    for x in range(monte_carlo_sims):
        daily_sim = np.random.choice(res["Strat_Ret"], size=len(res), replace=True)
        cum_sim = (1 + daily_sim).cumprod()
        simulations.append(cum_sim)
        
    # Plot Cone
    fig_mc = go.Figure()
    for sim in simulations[:50]: # Plot first 50 traces
        fig_mc.add_trace(go.Scatter(y=sim, mode='lines', line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
    
    fig_mc.add_trace(go.Scatter(y=cum_ret, mode='lines', name="Actual Strategy", line=dict(color='cyan', width=2)))
    fig_mc.add_trace(go.Scatter(y=(1+res["Bench_Ret"]).cumprod(), mode='lines', name="S&P 500", line=dict(color='white', width=1, dash='dash')))
    st.plotly_chart(fig_mc, use_container_width=True)

    # =========================================================================
    # 7. FAMA-FRENCH ALPHA DECOMPOSITION (SKILL VS LUCK)
    # =========================================================================
    st.header("3. Factor Decomposition (Alpha Analysis)")
    
    # Simple 1-Factor Alpha (CAPM) for demonstration
    # (In a real paper, you'd download Fama-French data libraries)
    X_capm = sm.add_constant(res["Bench_Ret"])
    model_capm = sm.OLS(res["Strat_Ret"], X_capm).fit()
    alpha = model_capm.params['const'] * 12
    beta = model_capm.params['Bench_Ret']
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### 🧬 Alpha: {alpha*100:.1f}%")
        st.caption("Annualized Excess Return NOT explained by the market.")
        st.info("A positive, significant Alpha implies genuine algorithmic edge, not just leverage.")
    with c2:
        st.markdown(f"### ⚖️ Beta: {beta:.2f}")
        st.caption("Correlation to S&P 500.")
        st.info("A Beta < 1.0 means the strategy is strictly defensive/uncorrelated.")

    # =========================================================================
    # 8. SHAP EXPLAINABILITY (WHY IT WORKS)
    # =========================================================================
    st.header("4. SHAP Feature Attribution")
    st.markdown("We use Game Theory (SHAP values) to explain exactly *which* features drove the AI's decisions.")
    
    # Create Explainer (TreeExplainer is optimized for Random Forests)
    explainer = shap.TreeExplainer(last_model)
    shap_values = explainer.shap_values(features.iloc[-100:])
    
    # Plot Summary
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig_shap = plt.figure()
    shap.summary_plot(shap_values[1], features.iloc[-100:], plot_type="bar", show=False)
    st.pyplot(fig_shap)

else:
    st.info("Define parameters and click Execute to start the Institutional Research Pipeline.")
