import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# =============================================================================
# 1. PROFESSIONAL CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Quant AI Research Platform", layout="wide", page_icon="📈")
st.title("📈 Quant AI: Adaptive Asset Allocation Research")
st.markdown("### A Machine Learning approach to minimizing Max Drawdown in High-Beta regimes")

# Sidebar
st.sidebar.header("Model Parameters")
ticker_list = st.sidebar.text_input("Universe (Risky, Safe, Benchmark)", "QQQ,SHY,SPY") 
train_window = st.sidebar.slider("Lookback Window (Months)", 12, 120, 24)

# =============================================================================
# 2. DATA INGESTION (FIXED & ROBUST)
# =============================================================================
@st.cache_data
def get_data(tickers):
    # 1. Clean the ticker text (remove spaces)
    clean_tickers = [t.strip().upper() for t in tickers.split(',')]
    
    # 2. Download data (Safe Mode)
    # We ask for "Adj Close" to account for dividends/splits
    data = yf.download(clean_tickers, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    
    # 3. Handle yfinance column messiness
    if "Adj Close" in data:
        data = data["Adj Close"]
    elif "Close" in data:
        data = data["Close"]
    
    # 4. Ensure it's a DataFrame, not a Series (if only 1 ticker)
    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    # 5. Drop any columns that are all NaN (bad tickers)
    data = data.dropna(axis=1, how='all')
    
    # 6. Resample to Monthly (using 'M' for max compatibility)
    data_monthly = data.resample("M").last()
    rets_monthly = data_monthly.pct_change()
    
    return data_monthly, rets_monthly

try:
    with st.spinner("Fetching Institutional Data..."):
        prices, monthly_rets = get_data(ticker_list)
        
        # Check if we actually got data for the required tickers
        required_tickers = [t.strip() for t in ticker_list.split(',')]
        missing = [t for t in required_tickers if t not in prices.columns]
        if missing:
            st.warning(f"⚠️ Could not find data for: {missing}. Check spelling!")
            
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# =============================================================================
# 3. FEATURE ENGINEERING & ML LOGIC
# =============================================================================
def engineer_features(rets, prices):
    try:
        feat = pd.DataFrame(index=rets.index)
        # Feature 1: The Spread (Is Tech outperforming Cash?)
        feat["Risk_Spread"] = rets["QQQ"] - rets["SHY"]
        # Feature 2: Momentum (Is the trend up?)
        feat["Momentum_3M"] = prices["QQQ"].pct_change(3)
        # Feature 3: Volatility (Is the market shaking?)
        feat["Volatility_3M"] = rets["QQQ"].rolling(3).std()
        
        # Target: 1 if Tech > Cash next month
        target = (rets["QQQ"].shift(-1) > rets["SHY"].shift(-1)).astype(int)
        return feat, target
    except KeyError:
        st.error("Error: Ensure QQQ and SHY are in the ticker list.")
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
# 5. INSTITUTIONAL METRICS (The "Ivy" Section)
# =============================================================================
def calculate_metrics(series):
    total_ret = (series.iloc[-1] - 1)
    # Annualized Sharpe Ratio (Risk-Adjusted Return)
    annualized_vol = series.pct_change().std() * np.sqrt(12)
    cagr = (series.iloc[-1]) ** (12 / len(series)) - 1
    sharpe = cagr / annualized_vol if annualized_vol != 0 else 0
    
    # Max Drawdown (Worst Crash)
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    max_dd = drawdown.min()
    
    return total_ret, sharpe, max_dd

if not backtest_results.empty:
    # --- Live Signal ---
    master_model = RandomForestClassifier(n_estimators=100, random_state=42)
    master_model.fit(data_historical.drop(columns=["target"]), data_historical["target"])
    current_prob = master_model.predict_proba(latest_features)[0, 1]
    
    # --- Calculate Equity Curves ---
    bt_dates = backtest_results.index
    strat_rets = []
    
    for date in bt_dates:
        regime = backtest_results.loc[date, "regime"]
        if date in monthly_rets.index:
             r_risky = monthly_rets.loc[date, "QQQ"] 
             r_safe = monthly_rets.loc[date, "SHY"]
             strat_rets.append(r_risky if regime == 1 else r_safe)
        else: strat_rets.append(0)
            
    strategy_curve = (1 + pd.Series(strat_rets, index=bt_dates)).cumprod()
    spy_curve = (1 + monthly_rets.loc[bt_dates, "SPY"]).cumprod()
    
    # --- Metrics Table ---
    strat_tot, strat_sharpe, strat_dd = calculate_metrics(strategy_curve)
    spy_tot, spy_sharpe, spy_dd = calculate_metrics(spy_curve)
    
    st.markdown("### 📊 Institutional Performance Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("Total Return", f"{strat_tot*100:.0f}%", delta=f"vs Mkt: {(strat_tot-spy_tot)*100:.0f}%")
    with col_m2: st.metric("Sharpe Ratio (Risk Adj.)", f"{strat_sharpe:.2f}", help="Higher is better. >1.0 is excellent.")
    with col_m3: st.metric("Max Drawdown (Risk)", f"{strat_dd*100:.1f}%", delta=f"Saved: {(spy_dd - strat_dd)*100:.1f}%", delta_color="inverse")
    with col_m4: 
        signal = "BUY TECH" if current_prob >= 0.5 else "DEFENSIVE CASH"
        st.metric("Current Regime", signal)

    # --- Charts ---
    st.area_chart(pd.DataFrame({"AI Strategy": strategy_curve, "Benchmark (SPY)": spy_curve}))

    # --- Explainable AI (XAI) ---
    st.markdown("---")
    st.header("🧠 Model Interpretability (Why did it choose this?)")
    
    # Feature Importance Visualization
    importances = master_model.feature_importances_
    feature_names = data_historical.drop(columns=["target"]).columns
    
    col_xai1, col_xai2 = st.columns([1, 2])
    with col_xai1:
        st.write("""
        **Feature Importance Analysis:**
        This chart shows what the AI is 'thinking' about most.
        * **Momentum:** Is the trend your friend?
        * **Volatility:** Is the market too scary?
        * **Risk_Spread:** Is Tech actually beating cash right now?
        """)
    with col_xai2:
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
        st.bar_chart(feat_df.set_index("Feature"))

    # --- Wealth Simulator (The Closer) ---
    st.markdown("---")
    st.header("💸 Long-Term Wealth Projection")
    c1, c2, c3 = st.columns(3)
    with c1: initial = st.number_input("Initial Capital ($)", 1000, step=1000)
    with c2: contrib = st.number_input("Monthly Add ($)", 500, step=100)
    with c3: years = st.slider("Projection Years", 1, 30, 20)

    strat_avg = np.mean(strat_rets)
    spy_avg = monthly_rets.loc[bt_dates, "SPY"].mean()
    months = years * 12
    proj_strat, proj_spy = [initial], [initial]

    for i in range(months):
        proj_strat.append(proj_strat[-1] * (1 + strat_avg) + contrib)
        proj_spy.append(proj_spy[-1] * (1 + spy_avg) + contrib)

    # Comparison Metrics
    final_ai = proj_strat[-1]
    final_mkt = proj_spy[-1]
    
    st.line_chart(pd.DataFrame({
        "AI Strategy": proj_strat, 
        "S&P 500": proj_spy
    }))
    
    st.success(f"💰 AI Wealth: ${final_ai:,.0f} | 📉 Market Wealth: ${final_mkt:,.0f}")
    st.caption("Past performance is not indicative of future results. This is a research simulation.")

else:
    st.warning("Insufficient data for analysis. Adjust parameters.")
