import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =============================================================================
# 1. CONFIGURATION & PAGE SETUP
# =============================================================================
st.set_page_config(page_title="T20 Quant Research Lab", layout="wide")
st.title("Institutional-Grade Adaptive Asset Allocation")
st.markdown("""
**Research Objective:** Mitigating tail-risk in high-beta portfolios using regime-switching algorithms.
* **Methodology:** Walk-Forward Optimization with Hyperparameter Tuning.
* **Risk Management:** Class-Weighted Learning to handle rare 'Crash' events.
""")

# Sidebar settings
st.sidebar.header("Model Hyperparameters")
ticker_list = st.sidebar.text_input("Universe", "QQQ, SHY, SPY") 
lookback = st.sidebar.slider("Training Window (Months)", 24, 120, 48)
optimize_model = st.sidebar.checkbox("Run GridSearch Optimization?", value=True)

# =============================================================================
# 2. DATA INGESTION (ROBUST)
# =============================================================================
@st.cache_data
def get_data(tickers):
    clean_tickers = [t.strip().upper() for t in tickers.split(',')]
    data = yf.download(clean_tickers, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    
    if "Adj Close" in data: data = data["Adj Close"]
    elif "Close" in data: data = data["Close"]
    
    if isinstance(data, pd.Series): data = data.to_frame()
    data = data.dropna(axis=1, how='all')
    
    # Resample to Monthly
    data_monthly = data.resample("M").last()
    rets_monthly = data_monthly.pct_change()
    
    return data_monthly, rets_monthly

try:
    with st.spinner("Fetching Market Data..."):
        prices, monthly_rets = get_data(ticker_list)
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# =============================================================================
# 3. FEATURE ENGINEERING (THE "WHY")
# =============================================================================
def engineer_features(rets, prices):
    feat = pd.DataFrame(index=rets.index)
    
    # 1. Volatility (Fear Gauge)
    feat["Volatility_6M"] = rets["QQQ"].rolling(6).std()
    
    # 2. Momentum (Trend)
    feat["Momentum_6M"] = prices["QQQ"].pct_change(6)
    
    # 3. Relative Strength (Spread)
    feat["Spread_Tech_vs_Safe"] = rets["QQQ"].rolling(3).mean() - rets["SHY"].rolling(3).mean()
    
    # Target: 1 if Tech beats Cash next month, 0 if Crash/Safe is better
    # We use 'shift(-1)' to align today's features with TOMORROW's result
    target = (rets["QQQ"].shift(-1) > rets["SHY"].shift(-1)).astype(int)
    
    return feat.dropna(), target.dropna()

features, target = engineer_features(monthly_rets, prices)

# Align indexes
common_idx = features.index.intersection(target.index)
features = features.loc[common_idx]
target = target.loc[common_idx]

# =============================================================================
# 4. WALK-FORWARD OPTIMIZATION ENGINE
# =============================================================================
def run_institutional_backtest(X, y, start_window=48):
    predictions = []
    
    # "Walk-Forward" Logic:
    # We start at month 48, train on 0-48, predict month 49.
    # Then we move to month 49, train on 0-49, predict month 50.
    # This prevents "Look-Ahead Bias."
    
    tscv = TimeSeriesSplit(n_splits=5) # For GridSearch internal validation
    
    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    
    # Store the "Best Params" to show the user
    best_params_log = []
    
    progress_bar = st.progress(0)
    total_steps = len(X) - start_window
    
    for i in range(start_window, len(X)):
        # Update Progress
        prog = (i - start_window) / total_steps
        progress_bar.progress(min(prog, 1.0))
        
        # 1. Expanding Window Train/Test Split
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]] # Next single month
        
        # 2. Model Selection (GridSearch or Standard)
        # Note: We use class_weight='balanced' to handle Imbalanced Data (Crashes are rare)
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        if optimize_model and i % 12 == 0: # Re-optimize once a year to save time
            search = GridSearchCV(clf, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params_log.append(search.best_params_)
        else:
            # Use default or last optimized model
            model = clf
            model.fit(X_train, y_train)
            
        # 3. Predict Next Month
        pred = model.predict(X_test)[0]
        predictions.append({"date": X_test.index[0], "regime": pred})
        
    progress_bar.empty()
    return pd.DataFrame(predictions).set_index("date"), best_params_log, model

if st.button("🚀 Run Institutional Backtest"):
    with st.spinner("Running Walk-Forward Optimization... (This simulates real trading)"):
        backtest_results, params_log, final_model = run_institutional_backtest(features, target, start_window=lookback)
        
        if backtest_results.empty:
            st.error("Not enough data. Decrease window or check tickers.")
            st.stop()
            
        # =============================================================================
        # 5. RESULTS & INTERPRETABILITY
        # =============================================================================
        
        # --- Calculate Returns ---
        aligned_rets = monthly_rets.loc[backtest_results.index]
        strat_rets = []
        for date, row in backtest_results.iterrows():
            if row['regime'] == 1:
                strat_rets.append(aligned_rets.loc[date, "QQQ"]) # Risk On
            else:
                strat_rets.append(aligned_rets.loc[date, "SHY"]) # Risk Off (Cash)
                
        # Cumulative Returns
        strat_curve = (1 + pd.Series(strat_rets, index=backtest_results.index)).cumprod()
        bench_curve = (1 + aligned_rets["SPY"]).cumprod()
        
        # Metrics
        total_ret = strat_curve.iloc[-1] - 1
        bench_ret = bench_curve.iloc[-1] - 1
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy Return", f"{total_ret*100:.1f}%", delta=f"{total_ret-bench_ret:.1%}")
        
        # --- Interpretability (XAI) ---
        st.subheader("Explainable AI: Feature Importance")
        st.markdown("""
        **Why does the model switch regimes?** We use Feature Importance to "open the black box." 
        * If **Volatility** is high, the model learns to prioritize safety.
        * If **Momentum** is strong, it chases the trend.
        """)
        
        # Feature Importance Chart
        importances = pd.DataFrame({
            'Feature': features.columns,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.bar_chart(importances.set_index("Feature"))
        
        # --- Plot Performance ---
        st.subheader("Performance vs Benchmark")
        chart_data = pd.DataFrame({"Adaptive AI": strat_curve, "S&P 500": bench_curve})
        st.line_chart(chart_data)
        
        # --- Technical Diagnostics (The "Quant Intern" Section) ---
        st.markdown("---")
        st.subheader("Model Diagnostics (For Research Paper)")
        
        d1, d2 = st.columns(2)
        with d1:
            st.write("**Hyperparameter Evolution:**")
            if params_log:
                st.json(params_log[-1]) # Show the latest "best" params
            else:
                st.write("Optimization was disabled.")
                
        with d2:
            st.write("**Class Imbalance Handling:**")
            st.info("Model used `class_weight='balanced'`. This automatically adjusts weights inversely proportional to class frequencies, ensuring the model pays attention to rare 'Crash' events.")

else:
    st.info("Adjust parameters in the sidebar and click 'Run' to start the simulation.")
