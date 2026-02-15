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
st.set_page_config(page_title="Quant AI", layout="wide", page_icon="🛡️")
st.title("🛡️ The 'Cash is King' Strategy")
st.markdown("### AI that switches between Stocks (Risky) and Short-Term Bonds (Safe Cash)")

# Sidebar
st.sidebar.header("Configuration")
# We use SHY (1-3 Year Treasury) as our "Cash" substitute because it has longer history than BIL
ticker_list = st.sidebar.text_input("Tickers", "SPY,SHY,QQQ,IWM") 
train_window = st.sidebar.slider("AI Training Window (Months)", 12, 120, 24)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
@st.cache_data
def get_data(tickers):
    # Download data
    data = yf.download(tickers.split(','), start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.resample("ME").last(), data.resample("ME").last().pct_change()

try:
    with st.spinner("Downloading Data (Replacing TLT with SHY)..."):
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
        
        # KEY CHANGE: Compare SPY to SHY (Cash) instead of TLT (Long Bonds)
        feat["risk_on_spread"] = rets["SPY"] - rets["SHY"]
        
        # Momentum features
        feat["spy_mom_3m"] = prices["SPY"].pct_change(3)
        feat["spy_vol_3m"] = rets["SPY"].rolling(3).std()
        
        # Target: 1 if Stocks beat Cash next month
        target = (rets["SPY"].shift(-1) > rets["SHY"].shift(-1)).astype(int)
        
        return feat, target
    except KeyError as e:
        st.error(f"Missing ticker: {e}")
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
        signal = "BUY STOCKS (SPY) 🚀" if current_prob >= 0.5 else "STAY IN CASH (SHY) 💤"
        color = "green" if current_prob >= 0.5 else "orange"
        st.markdown(f"### Next Month Signal: :{color}[{signal}]")
        st.caption(f"AI Confidence: {current_prob*100:.1f}%")

    # Equity Curve Calculation
    bt_dates = backtest_results.index
    strat_rets = []
    for date in bt_dates:
        regime = backtest_results.loc[date, "regime"]
        if date in monthly_rets.index:
             r_spy = monthly_rets.loc[date, "SPY"]
             r_shy = monthly_rets.loc[date, "SHY"] # Replaced TLT with SHY
             strat_rets.append(r_spy if regime == 1 else r_shy)
        else: strat_rets.append(0)
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()

    with col2:
         total_return = (strategy_curve.iloc[-1] - 1) * 100
         st.metric("Total Historical Return", f"+{total_return:,.0f}%")

    st.line_chart(pd.DataFrame({"Strategy (Cash-Safe)": strategy_curve, "S&P 500": spy_curve}))

    # Future Simulator
    st.markdown("---")
    st.header("🔮 Future Wealth Simulator (Cash-Safe Edition)")
    
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
        "AI Strategy": proj_strat[1:], 
        "S&P 500": proj_spy[1:]
    }, index=[bt_dates[-1] + timedelta(days=30*i) for i in range(1, months + 1)]))

    c1, c2 = st.columns(2)
    c1.info(f"💰 AI Wealth: ${proj_strat[-1]:,.2f}")
    c2.warning(f"📉 Market Wealth: ${proj_spy[-1]:,.2f}")

else:
    st.warning("Not enough data. Lower the window.")
