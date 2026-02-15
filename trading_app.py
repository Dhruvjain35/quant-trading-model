import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Quant AI", layout="wide", page_icon="🤖")
st.title("AI Financial Advisor & Time Machine")
st.markdown("### Historical Backtest + Future Wealth Projector")

# Sidebar
st.sidebar.header("Configuration")
ticker_list = st.sidebar.text_input("Tickers", "SPY,TLT,QQQ,IWM,GLD")
train_window = st.sidebar.slider("AI Training Window (Months)", 24, 120, 48)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
@st.cache_data
def get_data(tickers):
    # Force download until today
    data = yf.download(tickers.split(','), start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=True, progress=False)["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.resample("ME").last(), data.resample("ME").last().pct_change()

try:
    with st.spinner("Downloading Market Data..."):
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
        feat["risk_on_spread"] = rets["SPY"] - rets["TLT"]
        feat["spy_mom_6m"] = prices["SPY"].pct_change(6)
        feat["spy_vol_3m"] = rets["SPY"].rolling(3).std()
        target = (rets["SPY"].shift(-1) > rets["TLT"].shift(-1)).astype(int)
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
def run_backtest(X, y, window=48):
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
# 5. DASHBOARD - PART 1: HISTORY & SIGNAL
# =============================================================================
if not backtest_results.empty:
    # Live Signal
    master_model = RandomForestClassifier(n_estimators=100, random_state=42)
    master_model.fit(data_historical.drop(columns=["target"]), data_historical["target"])
    current_prob = master_model.predict_proba(latest_features)[0, 1]
    
    col1, col2 = st.columns(2)
    with col1:
        signal = "BUY STOCKS (SPY) 🚀" if current_prob >= 0.5 else "BUY BONDS (TLT) 🛡️"
        color = "green" if current_prob >= 0.5 else "red"
        st.markdown(f"### Next Month Signal: :{color}[{signal}]")
        st.caption(f"AI Confidence: {current_prob*100:.1f}%")

    # Equity Curve
    bt_dates = backtest_results.index
    strat_rets = []
    for date in bt_dates:
        regime = backtest_results.loc[date, "regime"]
        if date in monthly_rets.index:
             r_spy = monthly_rets.loc[date, "SPY"]
             r_tlt = monthly_rets.loc[date, "TLT"]
             strat_rets.append(r_spy if regime == 1 else r_tlt)
        else: strat_rets.append(0)
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()

    with col2:
         total_return = (strategy_curve.iloc[-1] - 1) * 100
         st.metric("Total Historical Return", f"+{total_return:,.0f}%")

    st.line_chart(pd.DataFrame({"Strategy (AI)": strategy_curve, "S&P 500": spy_curve}))

    # =============================================================================
    # 6. DASHBOARD - PART 2: FUTURE SIMULATOR (THE TIME MACHINE)
    # =============================================================================
    st.markdown("---")
    st.header("🔮 Future Wealth Simulator")
    st.markdown("Project your potential growth based on the strategy's past performance.")

    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        initial_investment = st.number_input("Initial Investment ($)", value=10000, step=1000)
    with col_input2:
        monthly_contribution = st.number_input("Monthly Contribution ($)", value=500, step=100)
    with col_input3:
        years = st.slider("Years to Project", 5, 30, 10)

    # Calculate Stats for Projection
    strat_avg_monthly = np.mean(strat_rets)
    spy_avg_monthly = monthly_rets.loc[bt_dates, "SPY"].mean()

    # Generate Future Dates
    future_months = years * 12
    last_date = bt_dates[-1]
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, future_months + 1)]

    # Project Growth
    proj_strat = [initial_investment]
    proj_spy = [initial_investment]

    for i in range(future_months):
        # Strategy Projection
        prev_strat = proj_strat[-1]
        new_strat = prev_strat * (1 + strat_avg_monthly) + monthly_contribution
        proj_strat.append(new_strat)

        # SPY Projection
        prev_spy = proj_spy[-1]
        new_spy = prev_spy * (1 + spy_avg_monthly) + monthly_contribution
        proj_spy.append(new_spy)

    # Plot Projection
    future_df = pd.DataFrame({
        "AI Strategy (Projected)": proj_strat[1:],
        "S&P 500 (Projected)": proj_spy[1:]
    }, index=future_dates)

    st.area_chart(future_df)

    # Results
    final_strat = proj_strat[-1]
    final_spy = proj_spy[-1]
    
    c1, c2 = st.columns(2)
    c1.info(f"💰 **Projected Strategy Wealth:** ${final_strat:,.2f}")
    c2.warning(f"📉 **Projected Market Wealth:** ${final_spy:,.2f}")
    
else:
    st.warning("Not enough data to run backtest. Lower the training window.")
