import yfinance as yf
import pandas as pd
import numpy as np
import shap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATA & FEATURE ENGINEERING (NO LEAKAGE)
# ==========================================
def fetch_and_engineer_data(risk_ticker='QQQ', safe_ticker='SHY'):
    """Fetches data and creates features using ONLY strictly historical data."""
    df_risk = yf.download(risk_ticker, period='20y', interval='1d')['Close'].rename('Close_Risk').to_frame()
    df_safe = yf.download(safe_ticker, period='20y', interval='1d')['Close'].rename('Close_Safe').to_frame()
    
    df = df_risk.join(df_safe, how='inner').dropna()
    
    # Calculate daily returns
    df['Ret_Risk'] = df['Close_Risk'].pct_change()
    df['Ret_Safe'] = df['Close_Safe'].pct_change()
    
    # Features (Everything MUST be shifted later to prevent lookahead bias)
    df['Mom_1M'] = df['Close_Risk'].pct_change(21)
    df['Mom_3M'] = df['Close_Risk'].pct_change(63)
    df['Vol_1M'] = df['Ret_Risk'].rolling(21).std() * np.sqrt(252)
    df['RSI_14'] = compute_rsi(df['Close_Risk'], 14)
    df['SMA_50_Dist'] = (df['Close_Risk'] / df['Close_Risk'].rolling(50).mean()) - 1
    
    df.dropna(inplace=True)
    
    # Target: Did the risk asset go up TOMORROW? 
    # (We shift the target backwards, meaning today's row has tomorrow's outcome)
    df['Target'] = (df['Ret_Risk'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 2. ML TRAINING (STRICT OUT-OF-SAMPLE)
# ==========================================
def train_and_predict(df):
    """Trains on the past, predicts on the future. Zero data leakage."""
    features = ['Mom_1M', 'Mom_3M', 'Vol_1M', 'RSI_14', 'SMA_50_Dist']
    
    # Chronological split: 70% Train, 30% Out-of-Sample (OOS) Test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df['Target']
    X_test, y_test = test_df[features], test_df['Target']
    
    # Initialize Models
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # Train
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    # Predict probabilities on Out-Of-Sample data
    rf_probs = rf.predict_proba(X_test)[:, 1]
    gb_probs = gb.predict_proba(X_test)[:, 1]
    
    # Ensemble Vote
    test_df = test_df.copy()
    test_df['Prob_Up'] = (rf_probs + gb_probs) / 2
    
    # Generate SHAP values for the OOS data using the Random Forest
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification in newer SHAP versions, extract the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1] 
        
    return test_df, X_test, shap_values, explainer.expected_value

# ==========================================
# 3. PATH-DEPENDENT BACKTESTER (THE FIX)
# ==========================================
def run_backtest(test_df, start_capital=100000.0, cost_bps=5, tax_st=0.22, tax_lt=0.15):
    """
    Simulates a real brokerage account. Taxes are paid from cash, not percentage returns.
    """
    capital = start_capital
    position = 0 # 1 for Long QQQ, 0 for Cash/SHY
    entry_price = 0.0
    days_held = 0
    
    equity_curve = []
    benchmark_capital = start_capital
    benchmark_curve = []
    
    # Transaction cost as a decimal
    cost_pct = cost_bps / 10000.0 
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        today_price_risk = row['Close_Risk']
        today_price_safe = row['Close_Safe']
        prob_up = row['Prob_Up']
        
        # Benchmark update (Buy and hold QQQ)
        if i > 0:
            benchmark_capital = benchmark_capital * (1 + row['Ret_Risk'])
        benchmark_curve.append(benchmark_capital)
        
        # --- STRATEGY LOGIC ---
        # We trade at the close based on the signal generated today
        
        target_position = 1 if prob_up > 0.50 else 0
        
        # If we need to SELL
        if position == 1 and target_position == 0:
            # Calculate total sale value minus slippage
            gross_sale = capital * (1 + row['Ret_Risk']) # Capital grew today
            gross_sale -= (gross_sale * cost_pct) # Pay broker
            
            # Calculate Taxes
            gain = gross_sale - capital  # Simplified gain calculation
            if gain > 0:
                tax_rate = tax_lt if days_held >= 252 else tax_st
                tax_owed = gain * tax_rate
                gross_sale -= tax_owed
                
            capital = gross_sale
            position = 0
            days_held = 0
            
        # If we need to BUY
        elif position == 0 and target_position == 1:
            capital = capital * (1 + row['Ret_Safe']) # Earned safe yield today
            capital -= (capital * cost_pct) # Pay broker to enter risk asset
            position = 1
            days_held = 1
            
        # If we are HOLDING RISK
        elif position == 1 and target_position == 1:
            capital = capital * (1 + row['Ret_Risk'])
            days_held += 1
            
        # If we are HOLDING SAFE
        elif position == 0 and target_position == 0:
            capital = capital * (1 + row['Ret_Safe'])
            
        equity_curve.append(capital)
        
    test_df['Strategy_Equity'] = equity_curve
    test_df['Benchmark_Equity'] = benchmark_curve
    test_df['Strategy_Ret'] = test_df['Strategy_Equity'].pct_change().fillna(0)
    test_df['Benchmark_Ret'] = test_df['Benchmark_Equity'].pct_change().fillna(0)
    
    return test_df

# ==========================================
# 4. STATISTICAL VALIDATION & METRICS
# ==========================================
def calculate_metrics(test_df):
    """Calculates standard quant metrics on the OOS equity curves."""
    strat_ret = test_df['Strategy_Ret']
    bench_ret = test_df['Benchmark_Ret']
    
    # Annualized Returns
    days = len(test_df)
    ann_strat = (test_df['Strategy_Equity'].iloc[-1] / test_df['Strategy_Equity'].iloc[0]) ** (252/days) - 1
    ann_bench = (test_df['Benchmark_Equity'].iloc[-1] / test_df['Benchmark_Equity'].iloc[0]) ** (252/days) - 1
    
    # Volatility
    vol_strat = strat_ret.std() * np.sqrt(252)
    
    # Sharpe (assuming 0 risk-free rate for simplicity, or subtract SHY yield)
    sharpe = ann_strat / vol_strat if vol_strat != 0 else 0
    
    # Max Drawdown
    roll_max = test_df['Strategy_Equity'].cummax()
    drawdown = (test_df['Strategy_Equity'] / roll_max) - 1
    max_dd = drawdown.min()
    
    # P-Value (T-Test on daily returns vs benchmark)
    t_stat, p_val = stats.ttest_ind(strat_ret, bench_ret, equal_var=False)
    
    return {
        "Ann_Return": ann_strat,
        "Bench_Return": ann_bench,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "P_Value": p_val
    }

def run_monte_carlo(test_df, sims=500):
    """Resamples daily strategy returns to test robustness."""
    strat_ret = test_df['Strategy_Ret'].values
    sim_paths = []
    
    for _ in range(sims):
        # Sample with replacement
        random_path = np.random.choice(strat_ret, size=len(strat_ret), replace=True)
        # Rebuild equity curve
        sim_curve = (1 + random_path).cumprod() * 100000
        sim_paths.append(sim_curve)
        
    return sim_paths

def get_crisis_alpha(test_df):
    """Isolates performance during the 2022 tech crash as an example."""
    try:
        crash_data = test_df.loc['2022-01-01':'2022-12-31']
        if len(crash_data) == 0:
            return "No 2022 data in Out-of-Sample period"
            
        strat_crash = (crash_data['Strategy_Equity'].iloc[-1] / crash_data['Strategy_Equity'].iloc[0]) - 1
        bench_crash = (crash_data['Benchmark_Equity'].iloc[-1] / crash_data['Benchmark_Equity'].iloc[0]) - 1
        return {"Strategy_2022": strat_crash, "Benchmark_2022": bench_crash}
    except:
        return "Insufficient data"
