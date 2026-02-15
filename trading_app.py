import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Quant AI", layout="wide", page_icon="💰")
st.title("💰 The $9 Million AI Strategy")
st.markdown("### High-Growth Tech (QQQ) with a Cash Safety Net (SHY)")

# Sidebar
st.sidebar.header("Settings")
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
    with st.spinner("Crunching the numbers..."):
        prices, monthly_rets = get_data(ticker_list)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# =============================================================================
# 3. AI BRAIN (Feature Engineering)
# =============================================================================
def engineer_features(rets, prices):
    try:
        feat = pd.DataFrame(index=rets.index)
        feat["risk_spread"] = rets["QQQ"] - rets["SHY"]
        feat["tech_mom_3m"] = prices["QQQ"].pct_change(3)
        feat["tech_vol_3m"] = rets["QQQ"].rolling(3).std()
        target = (rets["QQQ"].shift(-1) > rets["SHY"].shift(-1)).astype(int)
        return feat, target
    except KeyError:
        st.error("Error: Tickers must include QQQ and SHY")
        st.stop()

features, target = engineer_features(monthly_rets, prices)
data_full = features.copy()
data_full["target"] = target
data_historical = data_full.dropna()
latest_features = features.iloc[[-1]]

# =============================================================================
# 4. BACKTESTING
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
# 5. RESULTS DASHBOARD
# =============================================================================
if not backtest_results.empty:
    # --- Live Signal ---
    master_model = RandomForestClassifier(n_estimators=100, random_state=42)
    master_model.fit(data_historical.drop(columns=["target"]), data_historical["target"])
    current_prob = master_model.predict_proba(latest_features)[0, 1]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        signal = "BUY TECH (QQQ) 🚀" if current_prob >= 0.5 else "CASH SAFETY (SHY) 🛡️"
        color = "green" if current_prob >= 0.5 else "red"
        st.markdown(f"### Signal: :{color}[{signal}]")
    
    # --- Equity Curve Calculation ---
    bt_dates = backtest_results.index
    strat_rets = []
    
    # New Trade Log Logic
    trade_logs = []
    current_trade_start = bt_dates[0]
    current_regime = backtest_results.loc[bt_dates[0], "regime"]
    
    for i, date in enumerate(bt_dates):
        regime = backtest_results.loc[date, "regime"]
        
        # Calculate Returns for Curve
        if date in monthly_rets.index:
             r_risky = monthly_rets.loc[date, "QQQ"] 
             r_safe = monthly_rets.loc[date, "SHY"]
             strat_rets.append(r_risky if regime == 1 else r_safe)
        else: strat_rets.append(0)

        # Detect Trade Switch or End of Data
        if regime != current_regime or i == len(bt_dates) - 1:
            # Calculate performance for the period we just finished
            period_data = monthly_rets.loc[current_trade_start:date]
            
            # Strategy Return for this specific period
            if current_regime == 1:
                period_strat = (1 + period_data["QQQ"]).prod() - 1
                asset_name = "Held Tech (QQQ)"
            else:
                period_strat = (1 + period_data["SHY"]).prod() - 1
                asset_name = "Sat in Cash (SHY)"
            
            # Market (SPY) Return for same period
            period_spy = (1 + period_data["SPY"]).prod() - 1
            
            # Did we win?
            diff = period_strat - period_spy
            outcome = "✅ Beat Mkt" if diff > 0 else "❌ Lagged"
            
            trade_logs.append({
                "Start": current_trade_start.strftime('%Y-%m-%d'),
                "End": date.strftime('%Y-%m-%d'),
                "Action": asset_name,
                "Our Return": f"{period_strat*100:.1f}%",
                "Market Return": f"{period_spy*100:.1f}%",
                "Outcome": f"{outcome} ({diff*100:.1f}%)"
            })
            
            # Reset for next trade
            current_trade_start = date
            current_regime = regime
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()
    
    with c2:
        total_ret = (strategy_curve.iloc[-1] - 1) * 100
        st.metric("Total Return", f"+{total_ret:,.0f}%")
        
    st.line_chart(pd.DataFrame({"AI Strategy": strategy_curve, "S&P 500": spy_curve}))

    # --- Trade Diary (Enhanced) ---
    st.markdown("### 📜 AI Scorecard: Did the Move Pay Off?")
    st.markdown("Comparing each trade decision against just holding the S&P 500.")
    trades_df = pd.DataFrame(trade_logs).iloc[::-1]
    st.dataframe(trades_df.head(10), use_container_width=True)

    # --- Wealth Simulator ---
    st.markdown("---")
    st.header("💸 The $9 Million Projection")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1: 
        initial = st.number_input("Invest ($)", min_value=100, value=1000, step=100)
    with col_input2: 
        contrib = st.number_input("Monthly Add ($)", min_value=10, value=100, step=10)
    with col_input3: 
        years = st.slider("Years", min_value=1, max_value=30, value=10)

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

    # --- FINAL COMPARISON BOXES ---
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success(f"💰 Projected AI Wealth: ${proj_strat[-1]:,.2f}")
    with col_res2:
        st.warning(f"📉 Projected Market Wealth: ${proj_spy[-1]:,.2f}")

else:
    st.warning("Not enough data. Lower the window.")
