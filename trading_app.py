import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from data_loader import fetch_data # You will need a simple yfinance wrapper here
from feature_engineering import engineer_features
from models import walk_forward_ensemble
from backtest_engine import run_backtest
from evaluation import calculate_metrics

# Load Config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

st.set_page_config(page_title="AMCE Research Platform", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔬 AMCE Parameters")
tc_slider = st.sidebar.slider("Transaction Costs (bps)", 0, 50, config['backtest']['transaction_cost_bps'])
config['backtest']['transaction_cost_bps'] = tc_slider

if st.sidebar.button("Execute Out-of-Sample Backtest"):
    with st.spinner("Compiling Walk-Forward Data..."):
        
        # 1. Load Data
        prices = fetch_data([config['data']['risk_asset'], config['data']['safe_asset']], config['data']['start_date'])
        
        # 2. Engineer Features (No Leakage)
        X, y, returns = engineer_features(prices)
        
        # 3. Model Training (OOS only)
        probs, last_model, scaler = walk_forward_ensemble(X, y, config['backtest']['walk_forward_train_days'])
        
        # 4. Backtest Engine (with costs)
        bt_data = run_backtest(probs, returns, config)
        
        # 5. Evaluation Math
        metrics = calculate_metrics(bt_data)
        
        # --- DASHBOARD RENDER ---
        st.title("Adaptive Macro-Conditional Ensemble (AMCE)")
        
        # Executive Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OOS Sharpe Ratio", f"{metrics['Sharpe']:.2f}", f"95% CI: [{metrics['Sharpe_95CI'][0]:.2f}, {metrics['Sharpe_95CI'][1]:.2f}]")
        c2.metric("Annualized Alpha", f"{metrics['Alpha']*100:.2f}%", f"t-stat: {metrics['Alpha_t_stat']:.2f}")
        c3.metric("Max Drawdown", f"{metrics['Max_DD']*100:.2f}%")
        c4.metric("Ann. Turnover", f"{metrics['Avg_Turnover']:.1f}x")
        
        # Equity Curve
        st.subheader("Cumulative OOS Performance vs Benchmark")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.plot(bt_data.index, bt_data['Cum_Strat'], label='AMCE Strategy (Net of Costs)', color='#2ea043', linewidth=2)
        ax.plot(bt_data.index, bt_data['Cum_Bench'], label='Benchmark', color='#8b949e', alpha=0.6)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#0d1117', labelcolor='white')
        st.pyplot(fig)
        
        # Stat Rigor Readout
        st.subheader("Statistical Validation")
        if metrics['Alpha_p_value'] < 0.05:
            st.success(f"✅ Strategy Alpha is statistically significant (p = {metrics['Alpha_p_value']:.4f}). This indicates the excess return is likely not due to random chance.")
        else:
            st.warning(f"⚠️ Strategy Alpha is NOT statistically significant (p = {metrics['Alpha_p_value']:.4f}). Performance may be curve-fitted or reliant on market beta.")

else:
    st.info("Awaiting execution parameters from sidebar.")
