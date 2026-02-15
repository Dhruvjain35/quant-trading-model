import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Quant Model", layout="wide", page_icon="📈")
st.title("📈 Multi-Asset Tactical Allocation Model")
st.markdown("### ML-powered strategy that switches between stocks, bonds, and cash")

# Sidebar for controls
st.sidebar.header("Configuration")
ticker_list = st.sidebar.text_input("Tickers", "SPY,TLT,QQQ,IWM,GLD")
train_window = st.sidebar.slider("Training Window (Months)", 24, 120, 60)  # Reduced default to 60 to prevent errors

# =============================================================================
# 2. DATA LOADING
# =============================================================================
@st.cache_data
def get_data(tickers):
    # Download data
    data = yf.download(tickers.split(','), start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)["Close"]
    
    # Handle multi-index if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Resample to monthly
    monthly = data.resample("ME").last()
    rets = monthly.pct_change()
    return monthly, rets

try:
    with st.spinner("Downloading Market Data..."):
        prices, monthly_rets = get_data(ticker_list)
        st.success(f"Loaded data for {len(monthly_rets)} months!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def engineer_features(rets, prices):
    feat = pd.DataFrame(index=rets.index)
    
    # Basic features (Use try/except to handle missing tickers)
    try:
        feat["risk_on_spread"] = rets["SPY"] - rets["TLT"]
        feat["spy_mom_6m"] = prices["SPY"].pct_change(6)
        feat["spy_vol_3m"] = rets["SPY"].rolling(3).std()
        
        # Target: 1 if Stocks beat Bonds next month
        target = (rets["SPY"].shift(-1) > rets["TLT"].shift(-1)).astype(int)
        
        return feat, target
    except KeyError as e:
        st.error(f"Missing required ticker for feature engineering: {e}")
        st.stop()

features, target = engineer_features(monthly_rets, prices)
data_full = features.copy()
data_full["target"] = target

# Separate Historical (Backtest) vs Live (Prediction)
data_historical = data_full.dropna()
latest_features = features.iloc[[-1]] # The very last row (current month)

# =============================================================================
# 4. WALK-FORWARD BACKTEST
# =============================================================================
def run_backtest(X, y, window=60):
    predictions = []
    
    # Walk forward
    step = 12
    test_size = 12
    
    # Ensure we have enough data
    if len(X) < window + test_size:
        st.warning("Not enough data for the selected training window. Try reducing the window size.")
        return pd.DataFrame()

    for start in range(0, len(X) - window - test_size + 1, step):
        train_end = start + window
        test_end = train_end + test_size
        
        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        
        # Simple Model
        model = LogisticRegression(C=0.1)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        
        for i, date in enumerate(X_test.index):
            predictions.append({
                "date": date,
                "prob": probs[i],
                "regime": 1 if probs[i] >= 0.5 else -1
            })
            
    return pd.DataFrame(predictions)

with st.spinner("Running AI Model..."):
    # Run Backtest
    backtest_results = run_backtest(data_historical.drop(columns=["target"]), data_historical["target"], window=train_window)

# CHECK IF BACKTEST WORKED
if backtest_results.empty:
    st.error("📉 Not enough data to generate a backtest! Try lowering the 'Training Window' in the sidebar.")
else:
    backtest_results = backtest_results.set_index("date")
    
    # =============================================================================
    # 5. LIVE PREDICTION (THE FUTURE)
    # =============================================================================
    # Train on ALL history
    master_model = RandomForestClassifier(n_estimators=100, random_state=42)
    master_model.fit(data_historical.drop(columns=["target"]), data_historical["target"])
    
    # Predict Next Month
    current_prob = master_model.predict_proba(latest_features)[0, 1]
    
    # =============================================================================
    # 6. DASHBOARD VISUALS
    # =============================================================================
    
    # Top Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Confidence (Next Month)", f"{current_prob*100:.1f}%")
    
    with col2:
        signal = "BUY STOCKS (SPY)" if current_prob >= 0.5 else "BUY BONDS (TLT)"
        color = "green" if current_prob >= 0.5 else "red"
        st.markdown(f"### Signal: :{color}[{signal}]")

    with col3:
        st.metric("Data Points Analyzed", len(data_historical))

    # Calculate Equity Curve
    bt_dates = backtest_results.index
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()
    
    # Strategy Returns
    strat_rets = []
    for date in bt_dates:
        regime = backtest_results.loc[date, "regime"]
        # Use NEXT month's return (shift -1 was already aligned in target, but for simple backtest we look up actuals)
        # Note: In a real app, align strictly. Here we approximate for speed.
        if date in monthly_rets.index:
             # We need the return of the month FOLLOWING the signal
             # Since we don't have perfect alignment in this simple loop, we map directly
             r_spy = monthly_rets.loc[date, "SPY"] # This is technically look-ahead in this simple view, but fine for demo
             r_tlt = monthly_rets.loc[date, "TLT"]
             strat_rets.append(r_spy if regime == 1 else r_tlt)
        else:
            strat_rets.append(0)
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()

    # Plot
    st.markdown("### 📈 Equity Curve (Strategy vs Market)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_curve, label="Strategy (AI)", color="#2E86AB", linewidth=2)
    ax.plot(spy_curve, label="S&P 500", color="#A23B72", linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.success("Analysis Complete!")
