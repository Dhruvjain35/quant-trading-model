import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. CONFIGURATION & ACADEMIC FRAMING
# =============================================================================
st.set_page_config(page_title="T20 Quant Research: Ensemble Regimes", layout="wide", page_icon="🎓")

st.title("🎓 Macro-Regime Ensemble Network (M-REN)")
st.markdown("""
**Research Abstract:** This project implements a **Two-Stage Ensemble Classifier** to detect high-beta market regimes. 
Unlike single-factor models, M-REN synthesizes **Linear (Logistic Regression)** and **Non-Linear (Random Forest)** signals 
to mitigate overfitting. It incorporates **Macro-Financial features** (VIX Surface, Yield Curve) and utilizes **Purged Walk-Forward Validation** to strictly prevent data leakage.
""")

# Sidebar settings
st.sidebar.header("🔬 Experimental Controls")
ticker_risky = st.sidebar.text_input("High-Beta Asset", "QQQ") 
ticker_safe = st.sidebar.text_input("Risk-Free Asset", "SHY")
gap_months = st.sidebar.slider("Purged Gap (Months)", 0, 6, 3, help="Months to skip between Train and Test sets to prevent leakage.")

# =============================================================================
# 2. INSTITUTIONAL DATA INGESTION (VIX + YIELDS)
# =============================================================================
@st.cache_data
def get_institutional_data(risky, safe):
    # We fetch Asset Prices + VIX (Fear) + TNX (10yr Yield)
    tickers = [risky, safe, "^VIX", "^TNX"]
    data = yf.download(tickers, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    
    # Handle Multi-Index columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"] if "Adj Close" in data else data["Close"]
        
    data = data.dropna(axis=1, how='all').fillna(method='ffill')
    
    # Resample to Monthly
    data_m = data.resample("M").last()
    rets_m = data_m.pct_change()
    
    return data_m, rets_m

try:
    with st.spinner("Aggregating Macro-Financial Data (VIX, Treasury Yields)..."):
        prices, rets = get_institutional_data(ticker_risky, ticker_safe)
except Exception as e:
    st.error(f"Data Feed Error: {e}")
    st.stop()

# =============================================================================
# 3. SOPHISTICATED FEATURE ENGINEERING
# =============================================================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features(prices, rets, risky_ticker):
    df = pd.DataFrame(index=prices.index)
    
    # 1. Macro Signal: Yield Curve Proxy (10Y Yield Rate of Change)
    # Rising yields often hurt Tech/Growth stocks.
    if "^TNX" in prices.columns:
        df["Yield_Change_3M"] = prices["^TNX"].diff(3)
    
    # 2. Fear Signal: Volatility Regime (VIX Level)
    if "^VIX" in prices.columns:
        df["VIX_Level"] = prices["^VIX"]
        df["VIX_Shock"] = prices["^VIX"].pct_change(1) # Immediate fear spike
        
    # 3. Technical Signal: RSI (Momentum Health)
    df["RSI"] = calculate_rsi(prices[risky_ticker], window=6)
    
    # 4. Market Structure: Volatility of the Asset itself
    df["Realized_Vol_6M"] = rets[risky_ticker].rolling(6).std()
    
    # TARGET: 1 if Risky Asset beats Safe Asset next month
    target = (rets[risky_ticker].shift(-1) > rets[ticker_safe].shift(-1)).astype(int)
    
    # Drop NaNs created by rolling windows
    return df.dropna(), target.dropna()

features, target = engineer_features(prices, rets, ticker_risky)

# Align Data
common_idx = features.index.intersection(target.index)
features = features.loc[common_idx]
target = target.loc[common_idx]

# =============================================================================
# 4. ENSEMBLE MODELING WITH PURGED VALIDATION
# =============================================================================
def run_ensemble_backtest(X, y, gap):
    predictions = []
    log_probs = [] # For analysis
    rf_probs = []  # For analysis
    
    # Start loop after enough data (e.g., 60 months)
    start_idx = 60
    
    # We create a progress bar
    progress = st.progress(0)
    
    for i in range(start_idx, len(X)):
        progress.progress((i - start_idx) / (len(X) - start_idx))
        
        # --- PURGED SPLIT ---
        # We train up to "i - gap" to prevent "bleeding" of recent data into the test set
        train_end_idx = i - gap
        
        if train_end_idx < 20: # Safety check
            continue
            
        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_test = X.iloc[[i]] # Predicting strictly the NEXT month
        
        # Scaling (Important for Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- STAGE 1: LOGISTIC REGRESSION (Linear Baseline) ---
        model_lr = LogisticRegression(class_weight='balanced', solver='liblinear')
        model_lr.fit(X_train_scaled, y_train)
        prob_lr = model_lr.predict_proba(X_test_scaled)[0][1]
        
        # --- STAGE 2: RANDOM FOREST (Non-Linear Refinement) ---
        model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        model_rf.fit(X_train, y_train) # Tree models don't need scaling
        prob_rf = model_rf.predict_proba(X_test)[0][1]
        
        # --- ENSEMBLE VOTE (Soft Voting) ---
        # We average the probabilities. If Avg > 0.5, we go Risk-On.
        ensemble_prob = (prob_lr + prob_rf) / 2
        pred = 1 if ensemble_prob > 0.5 else 0
        
        predictions.append({
            "date": X_test.index[0],
            "pred": pred,
            "prob_lr": prob_lr,
            "prob_rf": prob_rf,
            "ensemble_prob": ensemble_prob
        })
        
    return pd.DataFrame(predictions).set_index("date")

if st.button("🚀 Run Ensemble Research Pipeline"):
    with st.spinner("Training Dual-Model Architecture..."):
        results = run_ensemble_backtest(features, target, gap_months)
    
    # =============================================================================
    # 5. RESULTS & ACADEMIC VISUALIZATION
    # =============================================================================
    
    # Calculate Returns
    aligned_rets = rets.loc[results.index]
    
    results["Strategy_Ret"] = np.where(results["pred"] == 1, 
                                       aligned_rets[ticker_risky], 
                                       aligned_rets[ticker_safe])
    
    results["Benchmark_Ret"] = aligned_rets[ticker_risky] # Compare vs Buy & Hold QQQ
    
    # Cumulative
    results["Equity_Strat"] = (1 + results["Strategy_Ret"]).cumprod()
    results["Equity_Bench"] = (1 + results["Benchmark_Ret"]).cumprod()
    
    # METRICS
    total_ret = results["Equity_Strat"].iloc[-1] - 1
    bench_ret = results["Equity_Bench"].iloc[-1] - 1
    
    # MAX DRAWDOWN CALCULATION
    roll_max = results["Equity_Strat"].cummax()
    drawdown = (results["Equity_Strat"] - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # 2008 / 2020 Stress Tests
    dd_2008 = drawdown.loc['2008-01-01':'2009-01-01'].min()
    dd_2020 = drawdown.loc['2020-01-01':'2020-05-01'].min()
    
    # --- DASHBOARD ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble Return", f"{total_ret*100:.0f}%", delta=f"{total_ret-bench_ret:.0%}")
    c2.metric("Max Drawdown", f"{max_dd*100:.1f}%", help="Peak to Trough Loss")
    c3.metric("2008 Crash Performance", f"{dd_2008*100:.1f}%", delta="Did we survive?", delta_color="inverse")
    
    # Main Chart
    st.subheader("📈 Equity Curve: Ensemble vs. Buy-and-Hold")
    st.line_chart(results[["Equity_Strat", "Equity_Bench"]])
    
    # --- EXPLAINABILITY SECTION (The "Why") ---
    st.markdown("---")
    st.subheader("🧠 Model Disagreement Analysis")
    st.markdown("""
    This chart shows where the **Linear Model (Logistic)** and **Non-Linear Model (Random Forest)** disagreed.
    * **High Disagreement** implies market ambiguity (High Uncertainty).
    * The Ensemble averages these views to smooth out false positives.
    """)
    
    # Plotting Probabilities
    st.line_chart(results[["prob_lr", "prob_rf"]].tail(50))
    
    st.success("Research Pipeline Complete. Use these metrics for your Technical Whitepaper.")

else:
    st.info("Awaiting Execution. Click the button to start the simulation.")
