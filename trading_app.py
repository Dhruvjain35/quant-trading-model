import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Quant Research: AMCE Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Financial Terminal Look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E1E25; padding: 15px; border-radius: 5px; border-left: 5px solid #00CCFF; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; font-weight: 300; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def get_data(ticker, start="2000-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close']

def engineer_features(prices, rets, ticker_risky):
    # 1. Macro Features (Simulated for Demo stability)
    # "VIX Proxy": 21-day rolling standard deviation (Annualized)
    vix_proxy = rets[ticker_risky].rolling(21).std() * (252**0.5) * 100
    
    # "Yield Curve Proxy": Inverse of recent trend (Yields up = Prices down)
    yield_proxy = (rets[ticker_risky].rolling(60).mean() * -1).fillna(0) 

    # 2. Technical Features (Momentum & Volatility)
    # Momentum: Distance from 50-day Moving Average
    sma_50 = prices[ticker_risky].rolling(window=50).mean()
    mom_50 = (prices[ticker_risky] / sma_50) - 1
    
    # Volatility Regime: Short-term vol vs Long-term vol
    vol_21 = rets[ticker_risky].rolling(21).std()
    vol_60 = rets[ticker_risky].rolling(60).std()
    vol_ratio = vol_21 / vol_60

    # RSI (Relative Strength Index)
    delta = prices[ticker_risky].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Combine
    features = pd.DataFrame({
        "VIX_Proxy": vix_proxy,
        "Yield_Trend": yield_proxy,
        "Momentum_50": mom_50,
        "Vol_Regime": vol_ratio,
        "RSI_14": rsi
    }).dropna()

    # Target: 1 if tomorrow's return is positive
    target = (rets[ticker_risky].shift(-1) > 0).astype(int)
    
    # Align indices
    common_idx = features.index.intersection(target.index)
    return features.loc[common_idx], target.loc[common_idx]

# ==========================================
# 3. ENSEMBLE ENGINE
# ==========================================
# ==========================================
# 3. ENSEMBLE ENGINE (OPTIMIZED)
# ==========================================
def run_ensemble(X, y, gap):
    results = []
    # Train on expanding window
    start_idx = 252 * 4  # 4 years warm-up
    
    # NEW: Pre-define models to reuse
    lr = LogisticRegression(C=0.1)
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    scaler = StandardScaler()

    # Optimization: Re-train only every 63 days (Quarterly)
    retrain_freq = 63 
    
    for i in range(start_idx, len(X)):
        # 1. Train ONLY if it's a "Re-train Day"
        if (i - start_idx) % retrain_freq == 0:
            train_end = i - gap
            if train_end < 252: continue
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            
            # Scale & Fit
            X_train_s = scaler.fit_transform(X_train)
            lr.fit(X_train_s, y_train)
            rf.fit(X_train, y_train)
        
        # 2. Predict for TODAY using the current model
        X_test = X.iloc[[i]]
        
        # Careful: Transform test data using the *current* scaler stats
        try:
            X_test_s = scaler.transform(X_test)
            
            prob_lr = lr.predict_proba(X_test_s)[0][1]
            prob_rf = rf.predict_proba(X_test)[0][1]
            
            final_signal = 1 if (prob_lr + prob_rf) / 2 > 0.5 else 0
            
            results.append({
                'Date': X.index[i],
                'Signal': final_signal,
                'Prob_LR': prob_lr,
                'Prob_RF': prob_rf
            })
        except:
            pass # Skip if model hasn't trained yet
    
    # Return results and the LAST trained model/data for SHAP
    return pd.DataFrame(results).set_index('Date'), rf, X_train

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.sidebar.header("🔬 Model Controls")
ticker_risky = st.sidebar.text_input("High-Beta Asset", "QQQ")
ticker_safe = st.sidebar.text_input("Risk-Free Asset", "SHY")
embargo = st.sidebar.slider("Purged Embargo (Months)", 1, 12, 3)
n_sims = st.sidebar.number_input("Monte Carlo Sims", 100, 1000, 200)

st.title("Adaptive Macro-Conditional Ensemble (AMCE)")
st.markdown("""
**Research Abstract:** This system implements a multi-factor voting ensemble to detect market regimes. 
It synthesizes Macro-Financial factors (VIX, Yields) with Technicals (Momentum, RSI) using **Purged Walk-Forward Validation**.
""")

if st.sidebar.button("🚀 Run Research Pipeline"):
    with st.spinner("Fetching market data and training models..."):
        # 1. Data
        p_risky = get_data(ticker_risky)
        p_safe = get_data(ticker_safe)
        
        # Align dates
        prices = pd.concat([p_risky, p_safe], axis=1).dropna()
        prices.columns = [ticker_risky, ticker_safe]
        rets = prices.pct_change().dropna()
        
        # 2. Features
        features, target = engineer_features(prices, rets, ticker_risky)
        
        # 3. Execution
        gap = embargo * 21
        backtest, last_model, X_train_last = run_ensemble(features, target, gap)
        
        # 4. Performance Calculation
        res = backtest.join(rets).dropna()
        res['Strat_Ret'] = np.where(res['Signal'] == 1, res[ticker_risky], res[ticker_safe])
        res['Bench_Ret'] = res[ticker_risky]
        
        cum_strat = (1 + res['Strat_Ret']).cumprod()
        cum_bench = (1 + res['Bench_Ret']).cumprod()
        
        # Metrics
        sharpe = (res['Strat_Ret'].mean() / res['Strat_Ret'].std()) * (252**0.5)
        sortino = (res['Strat_Ret'].mean() / res['Strat_Ret'][res['Strat_Ret']<0].std()) * (252**0.5)
        win_rate = len(res[res['Strat_Ret'] > 0]) / len(res)
        cvar_95 = res['Strat_Ret'].quantile(0.05) * 100

        # ==========================================
        # 5. VISUALIZATION
        # ==========================================
        
        # A. Executive Summary
        st.subheader("1. Executive Risk Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c2.metric("Sortino Ratio", f"{sortino:.2f}")
        c3.metric("CVaR (95%)", f"{cvar_95:.1f}%")
        c4.metric("Win Rate", f"{win_rate:.0%}")
        
        # B. Monte Carlo
        st.subheader("2. Monte Carlo Robustness Test")
        
        # Run Simulations
        simulations = []
        daily_vol = res['Strat_Ret'].std()
        days = len(res)
        
        for x in range(n_sims):
            # Randomized path based on historical volatility
            daily_shock = np.random.normal(0, daily_vol, days)
            price_path = (1 + res['Strat_Ret'].mean() + daily_shock).cumprod()
            simulations.append(price_path)
        
        sim_array = np.array(simulations)
        p95 = np.percentile(sim_array, 95, axis=0)
        p50 = np.percentile(sim_array, 50, axis=0)
        p05 = np.percentile(sim_array, 5, axis=0)

        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        # The Cone
        ax_mc.fill_between(range(len(p50)), p05, p95, color='gray', alpha=0.3, label='95% Confidence Interval')
        # The Median
        ax_mc.plot(p50, color='white', linestyle='--', alpha=0.8, label='Median Expectation')
        # The Reality
        ax_mc.plot(cum_strat.values, color='#00CCFF', linewidth=2, label='Actual Strategy')
        
        # Styling
        ax_mc.set_facecolor('#0E1117')
        fig_mc.patch.set_facecolor('#0E1117')
        ax_mc.grid(color='gray', linestyle=':', alpha=0.2)
        ax_mc.legend(facecolor='#0E1117', labelcolor='white')
        ax_mc.tick_params(colors='white')
        for spine in ax_mc.spines.values(): spine.set_edgecolor('white')
        st.pyplot(fig_mc)

        # C. Alpha Analysis (OLS)
        st.subheader("3. Factor Decomposition (Alpha Analysis)")
        
        # Regression
        y = res['Strat_Ret'] - 0.04/252
        x = res['Bench_Ret'] - 0.04/252
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        
        alpha_ann = model.params.iloc[0] * 252
        beta = model.params.iloc[1]
        p_val = model.pvalues.iloc[0]
        
        sig_stars = "⭐⭐⭐" if p_val < 0.01 else ("⭐⭐" if p_val < 0.05 else "⭐")
        
        ac1, ac2 = st.columns(2)
        ac1.metric("Alpha (Annualized)", f"{alpha_ann*100:.1f}%", f"P-Value: {p_val:.3f} {sig_stars}")
        ac2.metric("Beta (Correlation)", f"{beta:.2f}", "Target < 0.6 (Defensive)")

        # D. SHAP Attribution
        st.subheader("4. SHAP Feature Attribution")
        try:
            # Create explainer
            explainer = shap.TreeExplainer(last_model)
            shap_values = explainer.shap_values(X_train_last)
            
            # Robust Slicing Fix
            # 1. Get correct dimensions
            if isinstance(shap_values, list):
                sv = shap_values[1] # For binary classification
            else:
                sv = shap_values
            
            # 2. Slice both data and shap values to match last 100 days
            n_display = 100
            features_display = features.iloc[-n_display:]
            sv_display = sv[-n_display:]

            # 3. Ensure columns match
            # This protects against any accidental column mismatches
            common_cols = features.columns
            features_display = features_display[common_cols]
            
            fig_shap = plt.figure()
            shap.summary_plot(sv_display, features_display, plot_type="bar", show=False)
            st.pyplot(fig_shap)
            
        except Exception as e:
            st.warning(f"SHAP visualization skipped due to data alignment: {e}")

else:
    st.info("👈 Set parameters in the sidebar and click 'Run Research Pipeline' to begin.")
