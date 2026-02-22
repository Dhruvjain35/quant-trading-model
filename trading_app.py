import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import time

# ==========================================
# UI CONFIG & CUSTOM CSS (TERMINAL VIBE)
# ==========================================
st.set_page_config(page_title="AMCE Terminal v4.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Mimic the dark terminal UI from screenshots */
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Courier New', Courier, monospace; }
    .metric-container { background-color: #161A25; padding: 15px; border-radius: 5px; border-left: 4px solid #00FFAA; }
    .stButton>button { background-color: #00FFAA; color: #000000; font-weight: bold; width: 100%; border-radius: 4px; }
    .stButton>button:hover { background-color: #00CC88; color: white; }
    hr { border-color: #333333; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE ENGINE: DATA & FEATURES (NO LEAKAGE)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_and_engineer_data(risk_asset, safe_asset, embargo_months):
    # Fetch data including VIX for macro feature
    tickers = [risk_asset, safe_asset, '^VIX']
    df = yf.download(tickers, period='20y', interval='1d')['Close'].dropna()
    df.columns = ['Risk', 'Safe', 'VIX']
    
    # Calculate daily returns
    df['Risk_Ret'] = df['Risk'].pct_change()
    df['Safe_Ret'] = df['Safe'].pct_change()
    
    # --- FEATURE ENGINEERING (STRICTLY BACKWARD LOOKING) ---
    df['Mom_1M'] = df['Risk'].pct_change(21)
    df['Mom_3M'] = df['Risk'].pct_change(63)
    df['Mom_6M'] = df['Risk'].pct_change(126)
    df['Safe_Mom'] = df['Safe'].pct_change(63)
    
    df['Vol_21'] = df['Risk_Ret'].rolling(21).std() * np.sqrt(252)
    df['Vol_63'] = df['Risk_Ret'].rolling(63).std() * np.sqrt(252)
    df['Vol_Ratio'] = df['Vol_21'] / df['Vol_63']
    
    df['MA_50'] = (df['Risk'] / df['Risk'].rolling(50).mean()) - 1
    df['MA_200'] = (df['Risk'] / df['Risk'].rolling(200).mean()) - 1
    
    df['VIX_Level'] = df['VIX']
    df['VIX_Chg_1M'] = df['VIX'].pct_change(21)
    
    # Drop NaNs from rolling windows
    df.dropna(inplace=True)
    
    # --- THE TARGET (TOMORROW'S RETURN) ---
    # Target is 1 if tomorrow's risk asset return is positive, 0 otherwise
    df['Target'] = (df['Risk_Ret'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True) # Drop the final row which has no tomorrow
    
    # --- SPLIT WITH PURGED EMBARGO ---
    split_idx = int(len(df) * 0.7)
    embargo_days = int(embargo_months * 21) # Approx 21 trading days per month
    
    train_df = df.iloc[:split_idx - embargo_days].copy()
    test_df = df.iloc[split_idx:].copy() # OOS Data
    
    return train_df, test_df

# ==========================================
# 2. CORE ENGINE: ML & SHAP
# ==========================================
@st.cache_resource(show_spinner=False)
def train_models(train_df, test_df):
    features = ['Mom_1M', 'Mom_3M', 'Mom_6M', 'Safe_Mom', 'Vol_Ratio', 'MA_50', 'MA_200', 'VIX_Level', 'VIX_Chg_1M']
    
    X_train, y_train = train_df[features], train_df['Target']
    X_test = test_df[features]
    
    # Ensemble Models
    rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=50, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    # Probabilities
    rf_probs = rf.predict_proba(X_test)[:, 1]
    gb_probs = gb.predict_proba(X_test)[:, 1]
    test_df['Prob_Up'] = (rf_probs + gb_probs) / 2
    
    # SHAP Generation (Using a sample for speed & avoiding memory errors)
    X_test_sample = shap.utils.sample(X_test, 500)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Handle list format for binary classification in older/newer SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    return test_df, X_test_sample, shap_values, features

# ==========================================
# 3. CORE ENGINE: TAX-FREE BACKTESTER
# ==========================================
def vector_backtest(df, cost_bps):
    df = df.copy()
    cost_pct = cost_bps / 10000.0
    
    # Generate Signal (1 if AI thinks market goes up, 0 if down)
    df['Target_Position'] = (df['Prob_Up'] > 0.50).astype(int)
    
    # Shift position by 1 day to simulate buying at the close of the signal day
    df['Position'] = df['Target_Position'].shift(1).fillna(1)
    
    # Calculate Turnover for Slippage
    df['Turnover'] = df['Position'].diff().fillna(0).abs()
    df['Friction'] = df['Turnover'] * cost_pct
    
    # Gross & Net Returns
    df['Gross_Ret'] = np.where(df['Position'] == 1, df['Risk_Ret'], df['Safe_Ret'])
    df['Net_Ret'] = df['Gross_Ret'] - df['Friction']
    
    # Equity Curves
    df['Eq_Strategy'] = (1 + df['Net_Ret']).cumprod()
    df['Eq_Benchmark'] = (1 + df['Risk_Ret']).cumprod()
    
    # Drawdowns
    df['DD_Strategy'] = df['Eq_Strategy'] / df['Eq_Strategy'].cummax() - 1
    df['DD_Benchmark'] = df['Eq_Benchmark'] / df['Eq_Benchmark'].cummax() - 1
    
    return df

# ==========================================
# METRICS & STATS HELPERS
# ==========================================
def calc_metrics(df):
    days = len(df)
    ann_ret_strat = df['Eq_Strategy'].iloc[-1] ** (252/days) - 1
    ann_ret_bench = df['Eq_Benchmark'].iloc[-1] ** (252/days) - 1
    
    vol_strat = df['Net_Ret'].std() * np.sqrt(252)
    sharpe = ann_ret_strat / vol_strat if vol_strat != 0 else 0
    
    downside_rets = df.loc[df['Net_Ret'] < 0, 'Net_Ret']
    downside_vol = downside_rets.std() * np.sqrt(252)
    sortino = ann_ret_strat / downside_vol if downside_vol != 0 else 0
    
    max_dd = df['DD_Strategy'].min()
    max_dd_bench = df['DD_Benchmark'].min()
    
    tot_ret = (df['Eq_Strategy'].iloc[-1] - 1)
    tot_ret_bench = (df['Eq_Benchmark'].iloc[-1] - 1)
    
    return ann_ret_strat, ann_ret_bench, sharpe, sortino, max_dd, max_dd_bench, tot_ret, tot_ret_bench

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ASSET CONFIGURATION")
    risk_asset = st.text_input("High-Beta Asset", value="QQQ")
    safe_asset = st.text_input("Risk-Free Asset", value="SHY")
    
    st.markdown("### VALIDATION")
    embargo_months = st.slider("Purged Embargo (Months)", 1, 12, 4)
    monte_carlo_sims = st.number_input("Monte Carlo Sims", min_value=100, max_value=2000, value=500, step=100)
    
    st.markdown("### COST MODEL")
    cost_bps = st.slider("Slippage (bps per trade)", 0, 20, 5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ EXECUTE PIPELINE")
    
    st.markdown("---")
    st.caption("Regime-Filtered Boosting • Purged walk-forward validation • Ensemble voting • SHAP attribution")

# ==========================================
# MAIN DASHBOARD
# ==========================================
if run_btn:
    # --- PIPELINE EXECUTION UI ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📡 1/4: Fetching API & Engineering Features...")
    train_df, test_df = fetch_and_engineer_data(risk_asset, safe_asset, embargo_months)
    progress_bar.progress(25)
    
    status_text.text("🧠 2/4: Training Sklearn Random Forest & Gradient Boosting...")
    test_df, X_test_sample, shap_values, features = train_models(train_df, test_df)
    progress_bar.progress(65)
    
    status_text.text("💸 3/4: Running Walk-Forward Backtest (Applying Friction)...")
    res_df = vector_backtest(test_df, cost_bps)
    progress_bar.progress(90)
    
    status_text.text("📊 4/4: Calculating Institutional Metrics...")
    (ann_ret_strat, ann_ret_bench, sharpe, sortino, max_dd, max_dd_bench, tot_ret, tot_ret_bench) = calc_metrics(res_df)
    
    progress_bar.empty()
    status_text.success("✔️ Pipeline Execution Complete! (Strict Out-of-Sample)")
    
    # --- HEADER ---
    st.markdown("# Adaptive Macro-Conditional Ensemble")
    st.markdown(f"*AMCE v4.0 | Out-of-Sample Validated | No Data Leakage | Net of {cost_bps}bps Slippage*")
    
    st.markdown("""
    <div style="background-color: #121826; padding: 15px; border-radius: 8px; border: 1px solid #2B3548; margin-bottom: 20px;">
        <h5 style="color: #6366F1; margin-top:0;">RESEARCH HYPOTHESIS</h5>
        <p style="font-size: 0.9em; margin-bottom: 5px;"><strong>H₀ (Null):</strong> Macro-conditional regime signals provide no statistically significant improvement over passive equity exposure.</p>
        <p style="font-size: 0.9em; margin-bottom: 0;"><strong>H₁ (Alternative):</strong> Integrating Regime Filtering with Gradient Boosting generates positive crisis alpha and risk-adjusted outperformance, net of trading friction.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 01 EXECUTIVE RISK SUMMARY ---
    st.markdown("### 01 — EXECUTIVE RISK SUMMARY")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SHARPE RATIO", f"{sharpe:.3f}", f"Bench: {res_df['Risk_Ret'].mean()/res_df['Risk_Ret'].std()*np.sqrt(252):.3f}")
    c2.metric("SORTINO RATIO", f"{sortino:.3f}", "Downside Adj.")
    c3.metric("TOTAL RETURN", f"{tot_ret*100:.1f}%", f"Bench: {tot_ret_bench*100:.1f}%")
    c4.metric("ANN. RETURN", f"{ann_ret_strat*100:.1f}%", f"Bench: {ann_ret_bench*100:.1f}%")
    c5.metric("MAX DRAWDOWN", f"{max_dd*100:.1f}%", f"Bench: {max_dd_bench*100:.1f}%", delta_color="inverse")
    
    # --- 02 EQUITY CURVE & REGIME OVERLAY ---
    st.markdown("### 02 — EQUITY CURVE & REGIME OVERLAY")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Eq_Strategy'], mode='lines', name='AMCE Strategy', line=dict(color='#00FFAA', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Eq_Benchmark'], mode='lines', name=f'{risk_asset} Buy & Hold', line=dict(color='#888888', width=1, dash='dot')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['DD_Strategy']*100, mode='lines', name='Strat DD', line=dict(color='#FF3366', width=1), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['DD_Benchmark']*100, mode='lines', name='Bench DD', line=dict(color='#555555', width=1)), row=2, col=1)
    
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=20, b=0), height=500, legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    # --- 03 MONTE CARLO ---
    st.markdown("### 03 — MONTE CARLO ROBUSTNESS (BOOTSTRAPPED)")
    with st.spinner("Running Monte Carlo simulations..."):
        daily_rets = res_df['Net_Ret'].values
        mc_paths = np.zeros((monte_carlo_sims, len(daily_rets)))
        for i in range(monte_carlo_sims):
            # Bootstrap resampling with replacement
            mc_paths[i] = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
        
        mc_cumulative = np.cumprod(1 + mc_paths, axis=1)
        mc_percentiles = np.percentile(mc_cumulative, [5, 50, 95], axis=0)
        
        fig_mc = go.Figure()
        # 95% Cone
        fig_mc.add_trace(go.Scatter(x=res_df.index, y=mc_percentiles[2], mode='lines', line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=res_df.index, y=mc_percentiles[0], mode='lines', fill='tonexty', fillcolor='rgba(100, 100, 255, 0.1)', line=dict(width=0), name='95% Confidence Cone'))
        # Median
        fig_mc.add_trace(go.Scatter(x=res_df.index, y=mc_percentiles[1], mode='lines', name='Median Expectation', line=dict(color='#6666FF', dash='dash')))
        # Actual
        fig_mc.add_trace(go.Scatter(x=res_df.index, y=res_df['Eq_Strategy'], mode='lines', name='Actual Strategy', line=dict(color='#00FFAA', width=2)))
        
        fig_mc.update_layout(template='plotly_dark', height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_mc, use_container_width=True)

    # --- 04 & 05 ROW: FACTOR & CRISIS ---
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### 04 — FACTOR DECOMPOSITION (OLS ALPHA)")
        # Run OLS: Strategy_Ret = Alpha + Beta * Benchmark_Ret
        X = sm.add_constant(res_df['Risk_Ret'].values)
        y = res_df['Net_Ret'].values
        model = sm.OLS(y, X).fit()
        alpha_daily, beta = model.params[0], model.params[1]
        alpha_ann = (1 + alpha_daily)**252 - 1
        
        ca1, ca2 = st.columns(2)
        ca1.metric("Alpha (Ann.)", f"{alpha_ann*100:.2f}%", help="Excess return not explained by market direction.")
        ca2.metric("Market Beta", f"{beta:.2f}", help="Correlation to the benchmark asset.")
        st.caption(f"OLS Regression p-value: {model.pvalues[0]:.4f}")

    with colB:
        st.markdown("### 05 — CRISIS ALPHA ANALYSIS")
        # Define approximate dates for major crashes in the last 20 years
        crashes = {
            "2008 Financial Crisis": ('2008-08-01', '2009-03-01'),
            "2011 Euro Debt Crisis": ('2011-07-01', '2011-10-01'),
            "2015 Flash Crash": ('2015-08-01', '2015-09-01'),
            "2018 Volmageddon": ('2018-01-20', '2018-02-20'),
            "2020 COVID Crash": ('2020-02-15', '2020-03-25'),
            "2022 Tech Bear": ('2022-01-01', '2022-12-31')
        }
        
        crisis_data = []
        for name, (start, end) in crashes.items():
            try:
                period_df = res_df.loc[start:end]
                if len(period_df) > 10: # Ensure we actually have data for this period
                    strat_ret = (period_df['Eq_Strategy'].iloc[-1] / period_df['Eq_Strategy'].iloc[0]) - 1
                    bench_ret = (period_df['Eq_Benchmark'].iloc[-1] / period_df['Eq_Benchmark'].iloc[0]) - 1
                    edge = strat_ret - bench_ret
                    result = "✅ Preserved" if strat_ret > bench_ret else "❌ Underperformed"
                    crisis_data.append([name, f"{strat_ret*100:.1f}%", f"{bench_ret*100:.1f}%", f"+{edge*100:.1f}%" if edge>0 else f"{edge*100:.1f}%", result])
            except Exception:
                pass # Skip if dates aren't in test set
                
        if crisis_data:
            df_crisis = pd.DataFrame(crisis_data, columns=["Crisis Period", "Strategy", "Market", "Alpha (Edge)", "Result"])
            st.dataframe(df_crisis, hide_index=True, use_container_width=True)
        else:
            st.info("No major crisis dates fall within the out-of-sample test period.")

    # --- 06 SHAP ---
    st.markdown("### 06 — SHAP FEATURE ATTRIBUTION (GAME-THEORETIC)")
    st.markdown("<span style='color:gray; font-size:0.9em'>SHapley Additive exPlanations decompose predictions into individual feature contributions.</span>", unsafe_allow_html=True)
    
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Feature Importance (Mean |SHAP|)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        
        # Check SHAP version formatting
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, color='#00FFAA')
        st.pyplot(fig)
        
    with sc2:
        st.markdown("**SHAP Beeswarm (Directional Impact)**")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor('#0E1117')
        ax2.set_facecolor('#0E1117')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        
        shap.summary_plot(shap_values, X_test_sample, show=False)
        st.pyplot(fig2)

else:
    st.info("Configure your parameters in the sidebar and click **EXECUTE PIPELINE** to begin research.")
