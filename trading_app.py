"""
ADAPTIVE MACRO-CONDITIONAL ENSEMBLE (AMCE) 
V8.4 - PRO EQUITY SCALING (MDD CONTROL)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AMCE PRO V8.4", page_icon="▲", layout="wide")

# PROFESSIONAL UI STYLING
st.markdown("""
<style>
    :root {--bg:#0A0E14;--accent:#00FFB2;--panel:#11151C;--text:#EBEEF5;}
    .stApp {background-color:var(--bg); color:var(--text);}
    [data-testid="stMetric"] {background-color:var(--panel); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:4px;}
    [data-testid="stMetricValue"] {color:var(--accent) !important;}
    h1 {color:var(--accent); font-family:'Space Grotesk',sans-serif; text-transform:uppercase; letter-spacing:-1px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(risk, safe):
    # Expanded lookback for 200MA stability
    tickers = [risk, safe, '^VIX', '^TNX']
    df = yf.download(tickers, start="1995-01-01", end="2026-01-01", progress=False)['Close']
    df.ffill().dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df.rename(columns={risk:'Risk', safe:'Safe', '^VIX':'VIX', '^TNX':'Yield'})

def engineer_features(df):
    data = df.copy()
    data['Fwd_Ret'] = data['Risk'].shift(-1) / data['Risk'] - 1
    data['Target'] = (data['Fwd_Ret'] > 0).astype(int)
    
    # Core Alpha Features (v8.2 set)
    data['Mom_1M'] = data['Risk'].pct_change(21)
    data['Mom_3M'] = data['Risk'].pct_change(63)
    data['MA_200_Dist'] = data['Risk'] / data['Risk'].rolling(200).mean() - 1
    data['Yield_Mom'] = data['Yield'].pct_change(21)
    
    delta = data['Risk'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -1 * delta.clip(upper=0).rolling(14).mean()
    data['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
    
    data['Vol_Ratio'] = data['Risk'].pct_change().rolling(21).std() / data['Safe'].pct_change().rolling(21).std()
    data['Vol_Ratio_MA'] = data['Vol_Ratio'].rolling(63).mean()
    
    data.dropna(inplace=True)
    features = ['Mom_1M', 'Mom_3M', 'MA_200_Dist', 'Yield_Mom', 'Vol_Ratio', 'RSI_14']
    return data, features

def run_pro_backtest(data, features, target_vol, mdd_limit, tc_bps):
    # 1. ENSEMBLE SIGNAL GENERATION
    split = int(len(data) * 0.40)
    train, test = data.iloc[:split], data.iloc[split:].copy()
    
    X_train, y_train = train[features], train['Target']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42).fit(X_train, y_train)
    
    # Probability fusion
    test['Prob'] = (rf.predict_proba(test[features])[:,1] + gb.predict_proba(test[features])[:,1]) / 2
    test['Signal'] = (test['Prob'].ewm(span=10).mean() > 0.48).astype(int)
    
    # 2. DYNAMIC EXECUTION LOOP (Equity Scaling)
    risk_ret = test['Risk'].pct_change().fillna(0).values
    safe_ret = test['Safe'].pct_change().fillna(0).values
    signals = test['Signal'].shift(1).fillna(0).values
    
    n = len(test)
    net_rets = np.zeros(n)
    portfolio_value = np.ones(n)
    weights = np.zeros(n)
    
    hwm = 1.0 # High Water Mark
    mdd_decimal = mdd_limit / 100.0
    
    for i in range(1, n):
        # Calculate current DD from Peak
        current_dd = (portfolio_value[i-1] / hwm) - 1
        
        # PRO EQUITY SCALER (Starts at -5%, hits 0 at mdd_limit)
        if current_dd < -0.05:
            scale_factor = max(0, 1 - (abs(current_dd) - 0.05) / (mdd_decimal - 0.05))
        else:
            scale_factor = 1.0
            
        # Realized Vol Scaling (v8.2)
        window = risk_ret[max(0, i-21):i]
        realized_vol = np.std(window) * np.sqrt(252) if len(window) > 0 else 0.15
        vol_scalar = (target_vol/100) / max(realized_vol, 0.001)
        vol_scalar = min(vol_scalar, 1.0)
        
        # Final Weight
        w = signals[i] * vol_scalar * scale_factor
        weights[i] = w
        
        # Friction
        turnover = abs(weights[i] - weights[i-1])
        cost = turnover * (tc_bps / 10000)
        
        # Daily Return
        daily_ret = (w * risk_ret[i]) + ((1 - w) * safe_ret[i]) - cost
        net_rets[i] = daily_ret
        portfolio_value[i] = portfolio_value[i-1] * (1 + daily_ret)
        
        # Update HWM
        if portfolio_value[i] > hwm:
            hwm = portfolio_value[i]
            
    test['Net_Ret'] = net_rets
    test['Strat_Eq'] = portfolio_value
    test['Weight'] = weights
    test['Bench_Eq'] = (1 + risk_ret).cumprod()
    
    return test

# --- DASHBOARD ---
st.title("AMCE PRO V8.4 ▲")
st.caption("DYNAMIC EQUITY SCALING ENGINE")

with st.sidebar:
    st.header("Terminal Controls")
    risk_ticker = st.text_input("Risk Asset", "^NDX")
    safe_ticker = st.text_input("Safe Asset", "TLT")
    st.divider()
    vol_target = st.slider("Target Volatility %", 5, 30, 15)
    mdd_soft_limit = st.slider("Max DD Soft Limit %", 10, 40, 25)
    tc = st.number_input("Transaction Costs (bps)", 0, 50, 5)
    run = st.button("EXECUTE PRO PIPELINE", use_container_width=True)

if run:
    with st.spinner("Processing High-Frequency Features..."):
        raw_df = load_data(risk_ticker, safe_ticker)
        data_df, feat_cols = engineer_features(raw_df)
        results = run_pro_backtest(data_df, feat_cols, vol_target, mdd_soft_limit, tc)
        
    # CALCULATE METRICS
    total_ret = (results['Strat_Eq'].iloc[-1] - 1) * 100
    bench_ret = (results['Bench_Eq'].iloc[-1] - 1) * 100
    ann_ret = (1 + results['Net_Ret'].mean())**252 - 1
    sharpe = (results['Net_Ret'].mean() / results['Net_Ret'].std()) * np.sqrt(252)
    mdd = (results['Strat_Eq'] / results['Strat_Eq'].cummax() - 1).min() * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{total_ret:.1f}%", f"{total_ret-bench_ret:.1f}% vs Bench")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Max Drawdown", f"{mdd:.1f}%")
    c4.metric("Ann. Return", f"{ann_ret*100:.1f}%")

    # MAIN CHART
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=results.index, y=results['Strat_Eq'], name="AMCE V8.4", line=dict(color='#00FFB2', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=results.index, y=results['Bench_Eq'], name="Benchmark", line=dict(color='#8B95A8', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=results.index, y=results['Weight'], name="Exposure", fill='tozeroy', line=dict(color='rgba(124,77,255,0.5)')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Breakdown")
    st.dataframe(results[['Risk', 'Safe', 'Prob', 'Signal', 'Weight', 'Strat_Eq']].tail(10), use_container_width=True)
else:
    st.info("Awaiting execution... Configure parameters in the sidebar and click Execute.")
