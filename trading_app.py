"""
Multi-Asset Tactical Allocation Dashboard
A Streamlit app for visualizing and running the trading strategy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import os

# Import your model functions
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

# Page config
st.set_page_config(
    page_title="Tactical Allocation Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .signal-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .signal-spy {
        background-color: #d4edda;
        color: #155724;
        border: 3px solid #28a745;
    }
    .signal-tlt {
        background-color: #fff3cd;
        color: #856404;
        border: 3px solid #ffc107;
    }
    .signal-cash {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 3px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Multi-Asset Tactical Allocation</p>', unsafe_allow_html=True)
st.markdown("#### An ML-powered strategy that switches between stocks, bonds, and cash")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2E86AB/FFFFFF?text=Trading+Model", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    
    train_window = st.slider("Training Window (months)", 60, 180, 120, 12)
    test_window = st.slider("Test Window (months)", 6, 24, 12, 6)
    
    st.markdown("---")
    st.markdown("### 📅 Model Info")
    st.info(f"""
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
    
    **Assets Tracked:**
    - SPY (Stocks)
    - TLT (Bonds)
    - QQQ, IWM, GLD
    
    **Models:**
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    """)
    
    run_analysis = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# Helper functions
@st.cache_data(ttl=3600)
def download_data():
    """Download market data"""
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    prices = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    return prices

def create_features(monthly_prices, monthly_rets):
    """Create all features"""
    features = pd.DataFrame(index=monthly_rets.index)
    
    # Spreads
    features["risk_on_spread"] = monthly_rets["SPY"] - monthly_rets["TLT"]
    features["growth_lead"] = monthly_rets["QQQ"] - monthly_rets["SPY"]
    features["smallcaps_lead"] = monthly_rets["IWM"] - monthly_rets["SPY"]
    features["gold_lead"] = monthly_rets["GLD"] - monthly_rets["SPY"]
    
    # Momentum
    features["spy_mom_3m"] = monthly_prices["SPY"].pct_change(3)
    features["tlt_mom_3m"] = monthly_prices["TLT"].pct_change(3)
    features["gld_mom_3m"] = monthly_prices["GLD"].pct_change(3)
    features["spy_mom_6m"] = monthly_prices["SPY"].pct_change(6)
    features["tlt_mom_6m"] = monthly_prices["TLT"].pct_change(6)
    features["spy_mom_12m"] = monthly_prices["SPY"].pct_change(12)
    
    # Volatility
    features["spy_vol_3m"] = monthly_rets["SPY"].rolling(3).std()
    features["spy_vol_6m"] = monthly_rets["SPY"].rolling(6).std()
    features["tlt_vol_3m"] = monthly_rets["TLT"].rolling(3).std()
    features["spy_tlt_vol_ratio"] = features["spy_vol_3m"] / (features["tlt_vol_3m"] + 1e-6)
    
    # Trend
    spy_ma6 = monthly_prices["SPY"].rolling(6).mean()
    features["spy_ma_ratio_6m"] = (monthly_prices["SPY"] / spy_ma6) - 1
    tlt_ma6 = monthly_prices["TLT"].rolling(6).mean()
    features["tlt_ma_ratio_6m"] = (monthly_prices["TLT"] / tlt_ma6) - 1
    
    # Relative strength
    features["spy_tlt_mom_diff_3m"] = features["spy_mom_3m"] - features["tlt_mom_3m"]
    features["spy_tlt_mom_diff_6m"] = features["spy_mom_6m"] - features["tlt_mom_6m"]
    
    # Correlation
    cov_spy_tlt = monthly_rets["SPY"].rolling(12).cov(monthly_rets["TLT"])
    var_tlt = monthly_rets["TLT"].rolling(12).var()
    features["spy_tlt_beta_12m"] = cov_spy_tlt / (var_tlt + 1e-6)
    
    return features

def run_model(X, y, monthly_rets):
    """Run walk-forward validation"""
    TRAIN_WINDOW = 120
    TEST_WINDOW = 12
    STEP_SIZE = 12
    
    oos_predictions = []
    window_metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_windows = len(range(0, len(X) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE))
    
    for i, start_idx in enumerate(range(0, len(X) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE)):
        train_end = start_idx + TRAIN_WINDOW
        test_end = train_end + TEST_WINDOW
        
        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        
        if y_test.nunique() < 2:
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        model_lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        model_lr.fit(X_train_scaled, y_train)
        proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
        
        model_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_rf.fit(X_train_scaled, y_train)
        proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
        
        model_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model_gb.fit(X_train_scaled, y_train)
        proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
        
        proba_ensemble = (proba_lr + proba_rf + proba_gb) / 3
        
        # Calculate thresholds
        proba_train_lr = model_lr.predict_proba(X_train_scaled)[:, 1]
        proba_train_rf = model_rf.predict_proba(X_train_scaled)[:, 1]
        proba_train_gb = model_gb.predict_proba(X_train_scaled)[:, 1]
        proba_train_ensemble = (proba_train_lr + proba_train_rf + proba_train_gb) / 3
        
        upper_threshold = np.percentile(proba_train_ensemble, 70)
        lower_threshold = np.percentile(proba_train_ensemble, 30)
        
        regime_test = np.zeros(len(proba_ensemble))
        regime_test[proba_ensemble >= upper_threshold] = 1
        regime_test[proba_ensemble <= lower_threshold] = -1
        
        auc = roc_auc_score(y_test, proba_ensemble)
        
        for j, date in enumerate(X_test.index):
            oos_predictions.append({
                'date': date,
                'proba': proba_ensemble[j],
                'regime': regime_test[j],
                'true_label': y_test.iloc[j]
            })
        
        window_metrics.append({'window': i+1, 'auc': auc})
        
        progress = (i + 1) / total_windows
        progress_bar.progress(progress)
        status_text.text(f"Processing window {i+1}/{total_windows}...")
    
    progress_bar.empty()
    status_text.empty()
    
    df_predictions = pd.DataFrame(oos_predictions).set_index('date')
    df_metrics = pd.DataFrame(window_metrics)
    
    # Run backtest
    bt_dates = df_predictions.index
    spy_returns = monthly_rets.loc[bt_dates, "SPY"].shift(-1)
    tlt_returns = monthly_rets.loc[bt_dates, "TLT"].shift(-1)
    regimes = df_predictions['regime'].values
    
    strategy_returns = pd.Series(0.0, index=bt_dates)
    prev_regime = 0
    
    for i, date in enumerate(bt_dates):
        current_regime = regimes[i]
        
        if current_regime == 1:
            base_return = spy_returns.iloc[i]
        elif current_regime == -1:
            base_return = tlt_returns.iloc[i]
        else:
            base_return = 0.0
        
        if current_regime != prev_regime and i > 0:
            tc_cost = 0.0005 * 2
            strategy_returns.iloc[i] = base_return - tc_cost if not pd.isna(base_return) else -tc_cost
        else:
            strategy_returns.iloc[i] = base_return if not pd.isna(base_return) else 0.0
        
        prev_regime = current_regime
    
    bt = pd.DataFrame({
        'strategy': strategy_returns,
        'spy': spy_returns,
        'tlt': tlt_returns,
        'regime': regimes
    }).dropna()
    
    bt['strategy_growth'] = (1 + bt['strategy']).cumprod()
    bt['spy_growth'] = (1 + bt['spy']).cumprod()
    bt['tlt_growth'] = (1 + bt['tlt']).cumprod()
    
    return df_predictions, df_metrics, bt

# Main app
if run_analysis:
    with st.spinner("📥 Downloading market data..."):
        prices = download_data()
        monthly_prices = prices.resample("ME").last()
        monthly_rets = monthly_prices.pct_change()
    
    with st.spinner("🔧 Engineering features..."):
        features = create_features(monthly_prices, monthly_rets)
        target = (monthly_rets["SPY"].shift(-1) > monthly_rets["TLT"].shift(-1)).astype(int)
        data = features.copy()
        data["target"] = target
        data_clean = data.dropna()
        
        X = data_clean.drop(columns=["target"])
        y = data_clean["target"]
    
    with st.spinner("🤖 Training models and running backtest..."):
        df_predictions, df_metrics, bt = run_model(X, y, monthly_rets)
    
    st.success("✅ Analysis complete!")
    
    # Current Signal
    st.markdown("---")
    st.markdown("## 🎯 Current Signal")
    
    latest_regime = df_predictions['regime'].iloc[-1]
    latest_proba = df_predictions['proba'].iloc[-1]
    latest_date = df_predictions.index[-1]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if latest_regime == 1:
            signal_class = "signal-spy"
            signal_text = "BUY SPY (Stocks)"
            emoji = "🟢"
        elif latest_regime == -1:
            signal_class = "signal-tlt"
            signal_text = "BUY TLT (Bonds)"
            emoji = "🟡"
        else:
            signal_class = "signal-cash"
            signal_text = "HOLD CASH"
            emoji = "🔵"
        
        st.markdown(f'<div class="signal-box {signal_class}">{emoji} {signal_text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{latest_proba*100:.1f}%")
        st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")
    
    with col3:
        st.metric("Mean AUC", f"{df_metrics['auc'].mean():.3f}")
        st.caption("Model accuracy")
    
    # Performance Metrics
    st.markdown("---")
    st.markdown("## 📊 Performance Metrics")
    
    def calc_metrics(returns):
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (12 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        cum_returns = (1 + returns).cumprod()
        max_dd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
        return {
            'Total Return': f"{total_return*100:.1f}%",
            'Annual Return': f"{ann_return*100:.1f}%",
            'Annual Vol': f"{ann_vol*100:.1f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd*100:.1f}%",
            'Final $1': f"${(1+total_return):.2f}"
        }
    
    metrics_strategy = calc_metrics(bt['strategy'])
    metrics_spy = calc_metrics(bt['spy'])
    metrics_tlt = calc_metrics(bt['tlt'])
    
    metrics_df = pd.DataFrame({
        'Strategy': metrics_strategy,
        'SPY Buy&Hold': metrics_spy,
        'TLT Buy&Hold': metrics_tlt
    }).T
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Charts
    st.markdown("---")
    st.markdown("## 📈 Interactive Charts")
    
    # Equity Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt['strategy_growth'], name='Strategy', line=dict(color='#2E86AB', width=3)))
    fig.add_trace(go.Scatter(x=bt.index, y=bt['spy_growth'], name='SPY', line=dict(color='#A23B72', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=bt.index, y=bt['tlt_growth'], name='TLT', line=dict(color='#F18F01', width=2, dash='dash')))
    
    fig.update_layout(
        title="Equity Curve: Growth of $1",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown
    def calc_drawdown(growth):
        running_max = growth.cummax()
        return ((growth - running_max) / running_max) * 100
    
    dd_strategy = calc_drawdown(bt['strategy_growth'])
    dd_spy = calc_drawdown(bt['spy_growth'])
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt.index, y=dd_strategy, name='Strategy', fill='tozeroy', line=dict(color='#2E86AB')))
    fig2.add_trace(go.Scatter(x=bt.index, y=dd_spy, name='SPY', line=dict(color='#A23B72', dash='dash')))
    
    fig2.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Regime allocation
    regime_counts = bt['regime'].value_counts()
    regime_pct = (regime_counts / len(bt) * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = go.Figure(data=[go.Pie(
            labels=['SPY', 'CASH', 'TLT'],
            values=[regime_counts.get(1, 0), regime_counts.get(0, 0), regime_counts.get(-1, 0)],
            marker_colors=['#2E86AB', '#90A959', '#F18F01']
        )])
        fig3.update_layout(title="Regime Allocation", height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Allocation Summary")
        st.metric("SPY (Stocks)", f"{regime_pct.get(1, 0):.1f}%", "Growth focused")
        st.metric("CASH", f"{regime_pct.get(0, 0):.1f}%", "Risk management")
        st.metric("TLT (Bonds)", f"{regime_pct.get(-1, 0):.1f}%", "Safety focused")

else:
    # Landing page
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 What This Does")
        st.info("""
        Uses machine learning to predict whether stocks (SPY) or bonds (TLT) will perform better next month.
        
        Switches between assets to maximize returns and minimize risk.
        """)
    
    with col2:
        st.markdown("### 🤖 The Models")
        st.success("""
        - **Logistic Regression**
        - **Random Forest**
        - **Gradient Boosting**
        
        Ensemble of 3 models for robust predictions.
        """)
    
    with col3:
        st.markdown("### 📊 22 Features")
        st.warning("""
        - Momentum indicators
        - Volatility metrics
        - Trend signals
        - Cross-asset correlations
        
        Captures market behavior.
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("Click **'Run Full Analysis'** in the sidebar to see the latest signal and backtest results!")
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption("""
    This is for educational purposes only. Not financial advice. 
    Past performance does not guarantee future results. 
    Trading involves risk of loss.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Multi-Asset Tactical Allocation Model | Built with Python & Streamlit</p>
        <p>📧 Questions? Contact: your@email.com</p>
    </div>
    """,
    unsafe_allow_html=True
)