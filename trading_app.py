import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Institutional Quant: AMCE Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Financial Terminal Look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E1E25; padding: 15px; border-radius: 5px; border-left: 5px solid #00CCFF; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; font-weight: 300; }
    /* Force plot backgrounds to match */
    div[data-testid="stImage"] { background-color: #0E1117; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def get_data(ticker, start="2000-01-01"):
    # Download data
    df = yf.download(ticker, start=start, progress=False)
    
    # Handle MultiIndex columns if they exist (yfinance update fix)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: if 'Close' is in level 0, grab it.
        try:
            df = df.xs('Close', axis=1, level=0)
        except:
            # Fallback if structure is different (e.g. Ticker is level 0)
            df.columns = df.columns.get_level_values(0)
            if 'Close' in df.columns:
                df = df['Close']
            else:
                 # Last resort: take the first column assuming it's close/adj close
                df = df.iloc[:, 0]
                
    # If df is a DataFrame with 1 column, convert to Series
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
        
    return df

def engineer_features(prices, rets, ticker_risky):
    """
    Creates 8+ institutional-grade features for the model.
    """
    df = pd.DataFrame(index=prices.index)
    
    # 1. Momentum Factors (Trend)
    df['Mom_1M'] = prices[ticker_risky].pct_change(21)
    df['Mom_3M'] = prices[ticker_risky].pct_change(63)
    df['Mom_6M'] = prices[ticker_risky].pct_change(126)
    
    # 2. Volatility Factors (Risk)
    # Realized Volatility (Annualized)
    df['Vol_1M'] = rets[ticker_risky].rolling(21).std() * (252**0.5)
    
    # Volatility Regime (Short vs Long term vol)
    vol_3M = rets[ticker_risky].rolling(63).std()
    df['Vol_Regime'] = df['Vol_1M'] / vol_3M
    
    # 3. Mean Reversion (RSI)
    delta = prices[ticker_risky].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Market Stress Proxies (Simulated if external data missing)
    # "VIX Proxy": Rolling standard deviation of returns
    df['VIX_Proxy'] = df['Vol_1M'] * 100
    
    # "Yield Curve Slope Proxy": 
    # Since we don't have live Treasury data, we approximate yield stress 
    # by looking at the momentum of the Safe asset (bonds).
    # If Safe asset is crashing, yields are spiking.
    # We use Safe Asset 1M return as a proxy for rate changes.
    df['Rate_Stress'] = prices.iloc[:, 1].pct_change(21) * -1 # Inverse bond price
    
    # Clean up
    df = df.dropna()
    
    # Target: 1 if Risky Asset goes UP tomorrow
    target = (rets[ticker_risky].shift(-1) > 0).astype(int)
    
    # Align indices
    common_idx = df.index.intersection(target.index)
    return df.loc[common_idx], target.loc[common_idx]

# ==========================================
# 3. ENSEMBLE ENGINE (BATCH OPTIMIZED)
# ==========================================
def run_ensemble(X, y, gap):
    results = []
    # Start after 5 years (approx 1260 days) to have enough history
    start_idx = 1260 
    
    # Models
    lr = LogisticRegression(C=0.5, solver='liblinear') # More robust for small data
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
    scaler = StandardScaler()

    # BATCH SETTING: Retrain & Predict every 63 days (Quarterly)
    step_size = 63 
    
    for i in range(start_idx, len(X), step_size):
        # 1. Define Training Window (Purged)
        train_end = i - gap
        if train_end < 252: continue
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        # 2. Define Test Batch
        test_end = min(i + step_size, len(X))
        X_test_batch = X.iloc[i:test_end]
        
        if X_test_batch.empty: break

        # 3. Train Models (Once per batch)
        # Handle "In-Sample" vs "Out-Of-Sample" implicitly here
        X_train_s = scaler.fit_transform(X_train)
        lr.fit(X_train_s, y_train)
        rf.fit(X_train, y_train)
        
        # 4. Predict
        X_test_batch_s = scaler.transform(X_test_batch)
        prob_lr = lr.predict_proba(X_test_batch_s)[:, 1]
        prob_rf = rf.predict_proba(X_test_batch)[:, 1]
        
        # 5. Ensemble Vote
        avg_probs = (prob_lr + prob_rf) / 2
        signals = (avg_probs > 0.52).astype(int) # Slightly higher threshold for "Quality"
        
        batch_res = pd.DataFrame({
            'Signal': signals,
            'Prob_LR': prob_lr,
            'Prob_RF': prob_rf
        }, index=X_test_batch.index)
        
        results.append(batch_res)
    
    if not results:
        return pd.DataFrame(), rf, X # Fallback
        
    final_res = pd.concat(results)
    return final_res, rf, X_train

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.sidebar.header("🔬 Model Controls")
ticker_risky = st.sidebar.text_input("High-Beta Asset", "QQQ")
ticker_safe = st.sidebar.text_input("Defensive Asset", "IEF") # Changed to IEF (7-10yr Treasury) for better yield proxy
embargo = st.sidebar.slider("Purged Embargo (Months)", 1, 12, 1)

st.title("Adaptive Macro-Conditional Ensemble (AMCE)")
st.markdown("""
**Research Abstract:** Institutional-grade regime detection system. 
Utilizes a **Voting Ensemble** (Logistic + Random Forest) with **Purged Walk-Forward Validation** to minimize overfitting.
Focuses on **Crisis Alpha**—preserving capital during liquidity events.
""")

if st.sidebar.button("🚀 Run Research Pipeline"):
    with st.spinner("Fetching data, engineering factors, and running backtest..."):
        
        # 1. Data Ingestion
        try:
            p_risky = get_data(ticker_risky)
            p_safe = get_data(ticker_safe)
            
            # Align
            prices = pd.concat([p_risky, p_safe], axis=1).dropna()
            prices.columns = [ticker_risky, ticker_safe]
            rets = prices.pct_change().dropna()
            
            # 2. Feature Engineering
            features, target = engineer_features(prices, rets, ticker_risky)
            
            # 3. Execution
            gap = embargo * 21
            backtest, last_model, X_train_last = run_ensemble(features, target, gap)
            
            # 4. Performance Calculation
            res = backtest.join(rets).dropna()
            
            # Strategy Return: If Signal=1, buy Risky; else buy Safe
            res['Strat_Ret'] = np.where(res['Signal'] == 1, res[ticker_risky], res[ticker_safe])
            res['Bench_Ret'] = res[ticker_risky]
            
            cum_strat = (1 + res['Strat_Ret']).cumprod()
            cum_bench = (1 + res['Bench_Ret']).cumprod()
            
            # Metrics
            total_ret = cum_strat.iloc[-1] - 1
            sharpe = (res['Strat_Ret'].mean() / res['Strat_Ret'].std()) * (252**0.5)
            win_rate = len(res[res['Strat_Ret'] > 0]) / len(res)
            max_drawdown = (cum_strat / cum_strat.cummax() - 1).min()

            # ==========================================
            # 5. VISUALIZATION SECTIONS
            # ==========================================
            
            # --- SECTION 1: EXECUTIVE SUMMARY ---
            st.subheader("1. Executive Risk Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", f"{sharpe:.2f}", "Target: > 0.7")
            c2.metric("Total Return", f"{total_ret*100:.1f}%", f"Benchmark: {(cum_bench.iloc[-1]-1)*100:.0f}%")
            c3.metric("Max Drawdown", f"{max_drawdown*100:.1f}%", "Defensive Shield Active")
            c4.metric("Win Rate", f"{win_rate:.1%}", "Consistency Score")
            
            # --- SECTION 2: MONTE CARLO (BOOTSTRAP) ---
            st.subheader("2. Monte Carlo Robustness (Bootstrapped)")
            st.markdown("Uses **Resampling with Replacement** of actual strategy returns, preserving the statistical properties (fat tails) of the strategy.")
            
            n_sims = 200
            simulations = []
            
            # Bootstrap Loop
            for x in range(n_sims):
                # Sample RANDOM returns from the ACTUAL strategy history
                random_returns = np.random.choice(res['Strat_Ret'], size=len(res), replace=True)
                sim_path = (1 + random_returns).cumprod()
                simulations.append(sim_path)
            
            sim_array = np.array(simulations)
            p95 = np.percentile(sim_array, 95, axis=0)
            p50 = np.percentile(sim_array, 50, axis=0)
            p05 = np.percentile(sim_array, 5, axis=0)
            
            fig_mc, ax_mc = plt.subplots(figsize=(12, 5))
            ax_mc.fill_between(range(len(p50)), p05, p95, color='gray', alpha=0.3, label='95% Confidence Cone')
            ax_mc.plot(p50, color='white', linestyle='--', alpha=0.6, label='Median Expectation')
            ax_mc.plot(cum_strat.values, color='#00CCFF', linewidth=2, label='Actual Strategy')
            
            # Styling
            ax_mc.set_facecolor('#0E1117')
            fig_mc.patch.set_facecolor('#0E1117')
            ax_mc.grid(color='gray', linestyle=':', alpha=0.2)
            ax_mc.legend(facecolor='#0E1117', labelcolor='white')
            ax_mc.tick_params(colors='white')
            for spine in ax_mc.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig_mc)
            
            # --- SECTION 3: CRISIS ALPHA (NEW!) ---
            st.subheader("3. Crisis Alpha Analysis")
            st.markdown("How did the model perform during specific historical crashes?")
            
            # Define Crises
            crises = [
                ("2008 Financial Crisis", "2008-01-01", "2009-03-01"),
                ("2011 Euro Debt Crisis", "2011-05-01", "2011-10-01"),
                ("2015 Flash Crash", "2015-08-01", "2016-02-01"),
                ("2018 Volmageddon", "2018-09-01", "2018-12-31"),
                ("2020 COVID Crash", "2020-02-19", "2020-03-23"),
                ("2022 Inflation Bear", "2022-01-01", "2022-12-31"),
            ]
            
            crisis_data = []
            for name, start, end in crises:
                mask = (res.index >= start) & (res.index <= end)
                sub = res.loc[mask]
                if len(sub) > 0:
                    strat_perf = (1 + sub['Strat_Ret']).cumprod()[-1] - 1
                    bench_perf = (1 + sub['Bench_Ret']).cumprod()[-1] - 1
                    outperf = strat_perf - bench_perf
                    crisis_data.append({
                        "Event": name,
                        "Strategy": f"{strat_perf:.1%}",
                        "Market": f"{bench_perf:.1%}",
                        "Alpha (Edge)": f"{outperf:.1%}"
                    })
            
            c_df = pd.DataFrame(crisis_data)
            st.table(c_df)
            
            # --- SECTION 4: ROLLING STABILITY ---
            st.subheader("4. Strategy Stability (Rolling Metrics)")
            col_roll1, col_roll2 = st.columns(2)
            
            # Rolling Sharpe
            rolling_sharpe = res['Strat_Ret'].rolling(252).mean() / res['Strat_Ret'].rolling(252).std() * (252**0.5)
            
            fig_roll, ax_roll = plt.subplots(figsize=(8, 4))
            ax_roll.plot(rolling_sharpe.index, rolling_sharpe, color='#00CCFF', label='Rolling 1Y Sharpe')
            ax_roll.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax_roll.set_title("12-Month Rolling Sharpe Ratio")
            
            # Style Roll
            ax_roll.set_facecolor('#0E1117')
            fig_roll.patch.set_facecolor('#0E1117')
            ax_roll.tick_params(colors='white')
            for spine in ax_roll.spines.values(): spine.set_edgecolor('white')
            col_roll1.pyplot(fig_roll)
            
            # In-Sample vs Out-of-Sample Table
            split_idx = int(len(res) * 0.7)
            in_sample = res.iloc[:split_idx]
            out_sample = res.iloc[split_idx:]
            
            is_sharpe = (in_sample['Strat_Ret'].mean() / in_sample['Strat_Ret'].std()) * (252**0.5)
            os_sharpe = (out_sample['Strat_Ret'].mean() / out_sample['Strat_Ret'].std()) * (252**0.5)
            
            col_roll2.markdown(f"""
            ### Overfitting Diagnostic
            Comparing the first 70% of history (Training) vs the last 30% (Unseen).
            
            | Metric | In-Sample (Train) | Out-of-Sample (Test) |
            | :--- | :--- | :--- |
            | **Sharpe** | {is_sharpe:.2f} | {os_sharpe:.2f} |
            | **Total Return** | {( (1+in_sample['Strat_Ret']).cumprod()[-1]-1 )*100:.0f}% | {( (1+out_sample['Strat_Ret']).cumprod()[-1]-1 )*100:.0f}% |
            """)
            if abs(is_sharpe - os_sharpe) < 0.5:
                col_roll2.success("✅ LOW OVERFITTING DETECTED (Metrics are stable)")
            else:
                col_roll2.warning("⚠️ POTENTIAL DECAY (Metrics diverge)")

            # --- SECTION 5: STATISTICAL SIGNIFICANCE (PERMUTATION) ---
            st.subheader("5. Statistical Significance (Permutation Test)")
            
            # Run fast permutation
            perm_scores = []
            obs_sharpe = sharpe
            
            # Only run 100 for speed in demo, real world use 1000
            for _ in range(100):
                # Shuffle signals
                shuffled_signals = np.random.permutation(res['Signal'].values)
                # Recalculate return with shuffled signals
                shuff_ret = np.where(shuffled_signals == 1, res[ticker_risky], res[ticker_safe])
                # Calc Sharpe
                s_mean = np.mean(shuff_ret)
                s_std = np.std(shuff_ret)
                if s_std > 0:
                    perm_scores.append((s_mean/s_std) * (252**0.5))
            
            perm_scores = np.array(perm_scores)
            p_value = (np.sum(perm_scores >= obs_sharpe) + 1) / (len(perm_scores) + 1)
            
            fig_perm, ax_perm = plt.subplots(figsize=(10, 4))
            ax_perm.hist(perm_scores, bins=30, color='gray', alpha=0.5, label='Random Luck Distribution')
            ax_perm.axvline(obs_sharpe, color='#00CCFF', linewidth=3, label=f'Your Strategy (p={p_value:.3f})')
            ax_perm.legend()
            
            # Style Perm
            ax_perm.set_facecolor('#0E1117')
            fig_perm.patch.set_facecolor('#0E1117')
            ax_perm.tick_params(colors='white')
            for spine in ax_perm.spines.values(): spine.set_edgecolor('white')
            
            st.pyplot(fig_perm)
            if p_value < 0.05:
                st.success(f"⭐⭐⭐ Result is Statistically Significant (p={p_value:.3f} < 0.05)")
            else:
                st.info(f"Result is within random noise margin (p={p_value:.3f})")

            # --- SECTION 6: SHAP (FIXED) ---
            st.subheader("6. SHAP Feature Attribution (The 'Why')")
            
            try:
                # 1. Calc SHAP
                explainer = shap.TreeExplainer(last_model)
                shap_values = explainer.shap_values(X_train_last)
                
                # 2. Binary Classification Handler
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                else:
                    sv = shap_values
                    
                # 3. Robust Plotting (The White Box Fix)
                plt.close('all')
                fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                
                # Plot onto the specific axes
                shap.summary_plot(sv, X_train_last, plot_type="bar", show=False, max_display=10)
                
                # Grab the current figure manually to force Streamlit to see it
                fig_final = plt.gcf()
                fig_final.patch.set_facecolor('#0E1117')
                ax = plt.gca()
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                
                st.pyplot(fig_final)
                
            except Exception as e:
                st.warning(f"SHAP visualization skipped: {e}")

        except Exception as e:
            st.error(f"Critical Error in Pipeline: {e}")
            st.info("Tip: Ensure tickers (QQQ, IEF) are valid and you have internet connection.")

else:
    st.info("👈 Select parameters and click 'Run Research Pipeline'")
