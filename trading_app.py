import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. SETUP
# =============================================================================
st.set_page_config(page_title="Quant AI", layout="wide", page_icon="🚀")
st.title("🚀 The 'Tech Tycoon' Strategy")
st.markdown("### Trading High-Growth Tech (QQQ) vs. Safety (SHY)")

# Sidebar
st.sidebar.header("Configuration")
# We default to QQQ (Tech) and SHY (Cash)
ticker_list = st.sidebar.text_input("Tickers", "QQQ,SHY,SPY") 
train_window = st.sidebar.slider("AI Training Window (Months)", 12, 120, 24)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
@st.cache_data
def get_data(tickers):
    data = yf.download(tickers.split(','), start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.resample("ME").last(), data.resample("ME").last().pct_change()

try:
    with st.spinner("Downloading Tech & Cash Data..."):
        prices, monthly_rets = get_data(ticker_list)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def engineer_features(rets, prices):
    try:
        feat = pd.DataFrame(index=rets.index)
        
        # LOGIC CHANGE: Compare QQQ (Tech) vs SHY (Cash)
        # If Tech is beating Cash, we want to be in Tech.
        feat["risk_spread"] = rets["QQQ"] - rets["SHY"]
        
        # Momentum features on Tech
        feat["tech_mom_3m"] = prices["QQQ"].pct_change(3)
        feat["tech_vol_3m"] = rets["QQQ"].rolling(3).std()
        
        # Target: 1 if Tech beats Cash next month
        target = (rets["QQQ"].shift(-1) > rets["SHY"].shift(-1)).astype(int)
        
        return feat, target
    except KeyError as e:
        st.error(f"Missing ticker: {e} (Make sure QQQ and SHY are in the list)")
        st.stop()

features, target = engineer_features(monthly_rets, prices)
data_full = features.copy()
data_full["target"] = target
data_historical = data_full.dropna()
latest_features = features.iloc[[-1]]

# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================
def run_backtest(X, y, window=24):
    predictions = []
    if len(X) < window + 12: return pd.DataFrame()
    
    for start in range(0, len(X) - window - 12 + 1, 12):
        train_end = start + window
        test_end = train_end + 12
        
        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        
        model = LogisticRegression(C=0.1)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        
        for i, date in enumerate(X_test.index):
            # If prob > 0.5, buy Tech (1), else buy Cash (-1)
            predictions.append({"date": date, "regime": 1 if probs[i] >= 0.5 else -1})
            
    return pd.DataFrame(predictions).set_index("date")

backtest_results = run_backtest(data_historical.drop(columns=["target"]), data_historical["target"], window=train_window)

# =============================================================================
# 5. DASHBOARD
# =============================================================================
if not backtest_results.empty:
    # Live Signal
    master_model = RandomForestClassifier(n_estimators=100, random_state=42)
    master_model.fit(data_historical.drop(columns=["target"]), data_historical["target"])
    current_prob = master_model.predict_proba(latest_features)[0, 1]
    
    col1, col2 = st.columns(2)
    with col1:
        signal = "BUY TECH (QQQ) 🚀" if current_prob >= 0.5 else "SAFE MODE (SHY) 🛡️"
        color = "green" if current_prob >= 0.5 else "orange"
        st.markdown(f"### Next Month Signal: :{color}[{signal}]")
        st.caption(f"AI Confidence: {current_prob*100:.1f}%")

    # Equity Curve
    bt_dates = backtest_results.index
    strat_rets = []
    for date in bt_dates:
        regime = backtest_results.loc[date, "regime"]
        if date in monthly_rets.index:
             r_risky = monthly_rets.loc[date, "QQQ"] # Strategy trades QQQ
             r_safe = monthly_rets.loc[date, "SHY"]  # Or Cash
             strat_rets.append(r_risky if regime == 1 else r_safe)
        else: strat_rets.append(0)
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()
    
    # We compare against SPY (S&P 500) to show how much we beat the "Standard Market"
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()

    with col2:
         total_return = (strategy_curve.iloc[-1] - 1) * 100
         st.metric("Total Historical Return", f"+{total_return:,.0f}%")

    st.line_chart(pd.DataFrame({"AI Strategy (Tech)": strategy_curve, "Standard Market (S&P 500)": spy_curve}))

    # Future Simulator
    st.markdown("---")
    st.header("🔮 Future Wealth Simulator (Tech Edition)")
    st.caption("See what happens when you trade High-Growth Tech instead of Boring Stocks.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1: initial = st.number_input("Invest ($)", 10000, step=1000)
    with col_input2: contrib = st.number_input("Monthly Add ($)", 500, step=100)
    with col_input3: years = st.slider("Years", 5, 30, 10)

    # Project
    strat_avg = np.mean(strat_rets)
    spy_avg = monthly_rets.loc[bt_dates, "SPY"].mean()
    months = years * 12
    proj_strat, proj_spy = [initial], [initial]

    for i in range(months):
        proj_strat.append(proj_strat[-1] * (1 + strat_avg) + contrib)
        proj_spy.append(proj_spy[-1] * (1 + spy_avg) + contrib)

    st.area_chart(pd.DataFrame({
        "AI Strategy (Aggressive)": proj_strat[1:], 
        "Standard Market (Boring)": proj_spy[1:]
    }, index=[bt_dates[-1] + timedelta(days=30*i) for i in range(1, months + 1)]))

    c1, c2 = st.columns(2)
    c1.info(f"💰 AI Wealth: ${proj_strat[-1]:,.2f}")
    c2.warning(f"📉 Market Wealth: ${proj_spy[-1]:,.2f}")

else:
    st.warning("Not enough data. Lower the window.")
