import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==========================================
# UI CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Quant ML Engine", layout="wide")
st.title("Machine Learning Strategy Backtester")
st.markdown("Strict Out-of-Sample testing. Zero lookahead bias. Path-dependent tax/friction logic.")

# ==========================================
# 1. ENGINE: DATA & ML (CACHED FOR SPEED)
# ==========================================
@st.cache_data
def fetch_and_engineer_data(risk_ticker, safe_ticker):
    df_risk = yf.download(risk_ticker, period='15y', interval='1d')['Close'].rename('Close_Risk').to_frame()
    df_safe = yf.download(safe_ticker, period='15y', interval='1d')['Close'].rename('Close_Safe').to_frame()
    
    df = df_risk.join(df_safe, how='inner').dropna()
    df['Ret_Risk'] = df['Close_Risk'].pct_change()
    df['Ret_Safe'] = df['Close_Safe'].pct_change()
    
    # Features
    df['Mom_1M'] = df['Close_Risk'].pct_change(21)
    df['Mom_3M'] = df['Close_Risk'].pct_change(63)
    df['Vol_1M'] = df['Ret_Risk'].rolling(21).std() * np.sqrt(252)
    
    delta = df['Close_Risk'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['SMA_50_Dist'] = (df['Close_Risk'] / df['Close_Risk'].rolling(50).mean()) - 1
    
    df.dropna(inplace=True)
    df['Target'] = (df['Ret_Risk'].shift(-1) > 0).astype(int) # Target is tomorrow's direction
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_and_predict(df):
    features = ['Mom_1M', 'Mom_3M', 'Vol_1M', 'RSI_14', 'SMA_50_Dist']
    split_idx = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df['Target']
    X_test, y_test = test_df[features], test_df['Target']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    rf_probs = rf.predict_proba(X_test)[:, 1]
    gb_probs = gb.predict_proba(X_test)[:, 1]
    
    test_df = test_df.copy()
    test_df['Prob_Up'] = (rf_probs + gb_probs) / 2
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] 
        
    return test_df, X_test, shap_values

# ==========================================
# 2. ENGINE: PATH-DEPENDENT BACKTESTER
# ==========================================
def run_backtest(test_df, start_capital, cost_bps, tax_st, tax_lt):
    capital = start_capital
    position = 0 
    days_held = 0
    equity_curve, benchmark_curve = [], []
    benchmark_capital = start_capital
    cost_pct = cost_bps / 10000.0 
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        prob_up = row['Prob_Up']
        
        if i > 0: benchmark_capital *= (1 + row['Ret_Risk'])
        benchmark_curve.append(benchmark_capital)
        
        target_position = 1 if prob_up > 0.50 else 0
        
        if position == 1 and target_position == 0:
            gross_sale = capital * (1 + row['Ret_Risk'])
            gross_sale -= (gross_sale * cost_pct)
            gain = gross_sale - capital
            if gain > 0:
                tax_rate = tax_lt if days_held >= 252 else tax_st
                gross_sale -= (gain * tax_rate)
            capital = gross_sale
            position = 0
            days_held = 0
            
        elif position == 0 and target_position == 1:
            capital = capital * (1 + row['Ret_Safe'])
            capital -= (capital * cost_pct)
            position = 1
            days_held = 1
            
        elif position == 1 and target_position == 1:
            capital *= (1 + row['Ret_Risk'])
            days_held += 1
            
        elif position == 0 and target_position == 0:
            capital *= (1 + row['Ret_Safe'])
            
        equity_curve.append(capital)
        
    test_df['Strategy_Equity'] = equity_curve
    test_df['Benchmark_Equity'] = benchmark_curve
    test_df['Strategy_Ret'] = test_df['Strategy_Equity'].pct_change().fillna(0)
    test_df['Benchmark_Ret'] = test_df['Benchmark_Equity'].pct_change().fillna(0)
    
    return test_df

# ==========================================
# 3. SIDEBAR UI 
# ==========================================
with st.sidebar:
    st.header("Model Parameters")
    risk_ticker = st.text_input("Risk Asset", value="QQQ")
    safe_ticker = st.text_input("Safe Asset", value="SHY")
    
    st.header("Friction & Taxes")
    start_capital = st.number_input("Starting Capital", value=100000)
    cost_bps = st.number_input("Slippage/Fees (BPS)", value=5)
    tax_st = st.slider("Short-Term Tax %", 0, 50, 22) / 100.0
    tax_lt = st.slider("Long-Term Tax %", 0, 50, 15) / 100.0
    
    run_button = st.button("Run Simulation", type="primary")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
if run_button:
    with st.spinner("Fetching data and training ML Models out-of-sample..."):
        # 1. Run Pipeline
        df = fetch_and_engineer_data(risk_ticker, safe_ticker)
        test_df, X_test, shap_values = train_and_predict(df)
        res_df = run_backtest(test_df, start_capital, cost_bps, tax_st, tax_lt)
        
        # 2. Calculate Metrics
        days = len(res_df)
        ann_strat = (res_df['Strategy_Equity'].iloc[-1] / start_capital) ** (252/days) - 1
        ann_bench = (res_df['Benchmark_Equity'].iloc[-1] / start_capital) ** (252/days) - 1
        vol_strat = res_df['Strategy_Ret'].std() * np.sqrt(252)
        sharpe = ann_strat / vol_strat if vol_strat != 0 else 0
        max_dd = ((res_df['Strategy_Equity'] / res_df['Strategy_Equity'].cummax()) - 1).min()
        t_stat, p_val = stats.ttest_ind(res_df['Strategy_Ret'], res_df['Benchmark_Ret'], equal_var=False)

        # 3. Build UI Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Equity & Stats", "Monte Carlo", "SHAP Output", "Crisis Alpha"])
        
        with tab1:
            st.subheader("Out-of-Sample Equity Curve")
            st.line_chart(res_df[['Strategy_Equity', 'Benchmark_Equity']])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Strategy CAGR", f"{ann_strat*100:.2f}%")
            col2.metric("Benchmark CAGR", f"{ann_bench*100:.2f}%")
            col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col5.metric("P-Value", f"{p_val:.4f}", help="< 0.05 means statistically significant alpha")
            
        with tab2:
            st.subheader("Monte Carlo Path Simulation (500 Runs)")
            sims = 500
            strat_ret = res_df['Strategy_Ret'].values
            sim_df = pd.DataFrame()
            for i in range(sims):
                random_path = np.random.choice(strat_ret, size=len(strat_ret), replace=True)
                sim_df[f'Sim_{i}'] = (1 + random_path).cumprod() * start_capital
            
            # Plot subset for performance
            st.line_chart(sim_df.iloc[:, :50])
            st.caption("Displaying 50 out of 500 simulated paths based on randomized OOS returns.")
            
        with tab3:
            st.subheader("Feature Importance (SHAP)")
            st.markdown("Shows how each feature influenced the ML model's prediction to go Long.")
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)
            
        with tab4:
            st.subheader("Crisis Alpha: 2022 Tech Crash")
            try:
                crash_data = res_df.loc['2022-01-01':'2022-12-31']
                if not crash_data.empty:
                    strat_crash = (crash_data['Strategy_Equity'].iloc[-1] / crash_data['Strategy_Equity'].iloc[0]) - 1
                    bench_crash = (crash_data['Benchmark_Equity'].iloc[-1] / crash_data['Benchmark_Equity'].iloc[0]) - 1
                    c1, c2 = st.columns(2)
                    c1.metric("Strategy 2022 Return", f"{strat_crash*100:.2f}%")
                    c2.metric("Benchmark 2022 Return", f"{bench_crash*100:.2f}%")
                    st.line_chart(crash_data[['Strategy_Equity', 'Benchmark_Equity']])
                else:
                    st.warning("OOS period does not cover 2022. Increase the yfinance fetch period.")
            except KeyError:
                st.warning("Date index issue extracting 2022 data.")
