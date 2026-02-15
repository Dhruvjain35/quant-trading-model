# lesson1_spy_returns.py
# Multi-asset "market behavior" model + LIVE PREDICTION
# IMPROVED VERSION: Includes "Future Signal" for the current month

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("IMPROVED QUANT STRATEGY - With LIVE Future Prediction")
print("=" * 80)

# =============================================================================
# 0) DOWNLOAD LIVE DATA (Forces fresh download)
# =============================================================================
print("\n[1/9] Downloading LIVE price data...")

tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# Download until TODAY
prices = yf.download(
    tickers,
    start="2000-01-01",
    end=datetime.now().strftime('%Y-%m-%d'), # Forces fetch up to today
    auto_adjust=True,
    progress=False
)["Close"]

# Handle cases where yfinance returns MultiIndex columns
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

print(f"✓ Downloaded {len(prices)} days of data")
print(f"  Range: {prices.index.min().date()} to {prices.index.max().date()}")

# =============================================================================
# 1) RESAMPLE TO MONTHLY
# =============================================================================
print("\n[2/9] Resampling to monthly frequency...")

# 'ME' is Month End. We use last price of the month.
monthly_prices = prices.resample("ME").last()
monthly_rets = monthly_prices.pct_change()

# DROP the very last row if it's the current unfinished month? 
# Actually, for "Live Prediction", we need the current unfinished month's prices
# to predict the NEXT month. So we keep it.

print(f"✓ Monthly data points: {len(monthly_rets)}")

# =============================================================================
# 2) ENGINEER FEATURES
# =============================================================================
print("\n[3/9] Engineering features...")

features = pd.DataFrame(index=monthly_rets.index)

# --- Market Behavior ---
features["risk_on_spread"] = monthly_rets["SPY"] - monthly_rets["TLT"]
features["growth_lead"] = monthly_rets["QQQ"] - monthly_rets["SPY"]
features["smallcaps_lead"] = monthly_rets["IWM"] - monthly_rets["SPY"]
features["gold_lead"] = monthly_rets["GLD"] - monthly_rets["SPY"]

# --- Momentum ---
features["spy_mom_3m"] = monthly_prices["SPY"].pct_change(3)
features["tlt_mom_3m"] = monthly_prices["TLT"].pct_change(3)
features["gld_mom_3m"] = monthly_prices["GLD"].pct_change(3)
features["spy_mom_6m"] = monthly_prices["SPY"].pct_change(6)
features["spy_mom_12m"] = monthly_prices["SPY"].pct_change(12)

# --- Volatility ---
features["spy_vol_3m"] = monthly_rets["SPY"].rolling(3).std()
features["tlt_vol_3m"] = monthly_rets["TLT"].rolling(3).std()
features["spy_tlt_vol_ratio"] = features["spy_vol_3m"] / (features["tlt_vol_3m"] + 1e-6)

# --- Trend ---
spy_ma6 = monthly_prices["SPY"].rolling(6).mean()
features["spy_ma_ratio_6m"] = (monthly_prices["SPY"] / spy_ma6) - 1

# --- Correlation ---
cov_spy_tlt = monthly_rets["SPY"].rolling(12).cov(monthly_rets["TLT"])
var_tlt = monthly_rets["TLT"].rolling(12).var()
features["spy_tlt_beta_12m"] = cov_spy_tlt / (var_tlt + 1e-6)

print(f"✓ Created features. Total rows: {len(features)}")

# =============================================================================
# 3) SEPARATE "HISTORICAL" from "LIVE"
# =============================================================================
print("\n[4/9] Preparing Datasets...")

# The "Target" is what happens NEXT month.
# Shift(-1) moves next month's return "back" to this month's row.
target = (monthly_rets["SPY"].shift(-1) > monthly_rets["TLT"].shift(-1)).astype(int)

# Combine features and target
data = features.copy()
data["target"] = target

# THE FIX: Split the data BEFORE dropping NaNs
# 1. Historical Data (Backtesting): Rows where we know the outcome (Target is not NaN)
data_historical = data.dropna()

# 2. Live Data (Prediction): The VERY LAST row. 
# It has Features (current prices) but Target is NaN (we don't know next month yet)
latest_features = features.iloc[[-1]] 

print(f"✓ Historical Training Data: {len(data_historical)} months")
print(f"✓ Live Prediction Data: {latest_features.index[0].date()}")

# =============================================================================
# 4) WALK-FORWARD VALIDATION (BACKTEST)
# =============================================================================
print("\n[5/9] Running walk-forward validation (The Past)...")

TRAIN_WINDOW = 120
TEST_WINDOW = 12
STEP_SIZE = 12

X = data_historical.drop(columns=["target"])
y = data_historical["target"]

oos_predictions = []

for start_idx in range(0, len(X) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE):
    train_end = start_idx + TRAIN_WINDOW
    test_end = train_end + TEST_WINDOW
    
    X_train = X.iloc[start_idx:train_end]
    y_train = y.iloc[start_idx:train_end]
    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]
    
    if y_test.nunique() < 2: continue
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple Ensemble
    m1 = LogisticRegression(C=0.1, random_state=42).fit(X_train_scaled, y_train)
    m2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_train_scaled, y_train)
    
    p1 = m1.predict_proba(X_test_scaled)[:, 1]
    p2 = m2.predict_proba(X_test_scaled)[:, 1]
    p_ensemble = (p1 + p2) / 2
    
    for j, date in enumerate(X_test.index):
        oos_predictions.append({
            'date': date,
            'true_label': y_test.iloc[j],
            'proba': p_ensemble[j],
            'regime': 1 if p_ensemble[j] >= 0.5 else -1
        })

df_predictions = pd.DataFrame(oos_predictions).set_index('date')
print(f"✓ Walk-forward complete. {len(df_predictions)} predictions generated.")

# =============================================================================
# 5) CALCULATE RETURNS
# =============================================================================
print("\n[6/9] Calculating performance...")

# Align dates
bt_dates = df_predictions.index
spy_returns = monthly_rets.loc[bt_dates, "SPY"].shift(-1)
tlt_returns = monthly_rets.loc[bt_dates, "TLT"].shift(-1)
regimes = df_predictions['regime'].values

# Transaction Costs (0.10% per trade)
TC = 0.001 
strategy_returns = []
prev_regime = 0

for i in range(len(bt_dates)):
    curr = regimes[i]
    ret = spy_returns.iloc[i] if curr == 1 else tlt_returns.iloc[i]
    
    # Subtract cost if we switched assets
    if curr != prev_regime and i > 0:
        ret -= TC
        
    strategy_returns.append(ret)
    prev_regime = curr

bt = pd.DataFrame({
    'strategy': strategy_returns,
    'spy': spy_returns,
    'tlt': tlt_returns
}, index=bt_dates).dropna()

# Cumulative Growth
bt['strategy_growth'] = (1 + bt['strategy']).cumprod()
bt['spy_growth'] = (1 + bt['spy']).cumprod()

final_return = bt['strategy_growth'].iloc[-1]
print(f"✓ Strategy Final $1 Growth: ${final_return:.2f}")

# =============================================================================
# 6) GENERATE LIVE PREDICTION (THE FUTURE)
# =============================================================================
print("\n" + "="*80)
print("[7/9] GENERATING LIVE FUTURE SIGNAL")
print("="*80)

# 1. Train a "Master Model" on ALL historical data available
scaler_live = StandardScaler()
X_all_scaled = scaler_live.fit_transform(X)
y_all = y

# Train robust models
master_lr = LogisticRegression(C=0.1, random_state=42).fit(X_all_scaled, y_all)
master_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_all_scaled, y_all)

# 2. Prepare the Latest Data (Today's Data)
latest_scaled = scaler_live.transform(latest_features)

# 3. Predict
prob_lr = master_lr.predict_proba(latest_scaled)[0, 1]
prob_rf = master_rf.predict_proba(latest_scaled)[0, 1]
final_prob = (prob_lr + prob_rf) / 2

print(f"\nDate of Analysis: {datetime.now().date()}")
print(f"Latest Data Point Used: {latest_features.index[0].date()}")
print("-" * 40)
print(f"Model Confidence (SPY vs TLT): {final_prob*100:.1f}%")
print("-" * 40)

if final_prob >= 0.55:
    signal = "BULLISH - BUY STOCKS (SPY)"
    emoji = "🚀"
elif final_prob <= 0.45:
    signal = "BEARISH - BUY BONDS (TLT)"
    emoji = "🛡️"
else:
    signal = "NEUTRAL - CASH / HEDGE"
    emoji = "⚖️"

print(f"OFFICIAL SIGNAL FOR NEXT MONTH: {emoji} {signal} {emoji}")
print("="*80)

# =============================================================================
# 7) SAVE PLOTS
# =============================================================================
print("\n[8/9] Saving updated plots...")
plot_dir = "backtest_plots"
os.makedirs(plot_dir, exist_ok=True)

# Equity Curve
plt.figure(figsize=(12, 6))
plt.plot(bt.index, bt['strategy_growth'], label='Your Strategy', color='#2E86AB', linewidth=2)
plt.plot(bt.index, bt['spy_growth'], label='S&P 500', color='#A23B72', alpha=0.6)
plt.title(f'Strategy Performance (Live Update: {datetime.now().date()})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f'{plot_dir}/1_equity_curve.png')
plt.close()

# Prediction Gauge (Simple Bar)
plt.figure(figsize=(6, 2))
plt.barh(['Signal'], [final_prob], color='#2E86AB' if final_prob > 0.5 else '#F18F01')
plt.xlim(0, 1)
plt.axvline(0.5, color='red', linestyle='--')
plt.title(f'Next Month Forecast: {final_prob:.2f}')
plt.tight_layout()
plt.savefig(f'{plot_dir}/forecast_gauge.png')
plt.close()

print(f"✓ Plots saved to {plot_dir}/")
print("\n[9/9] COMPLETE.")
