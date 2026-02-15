# lesson1_spy_returns.py
# Multi-asset "market behavior" model + walk-forward probs + SPY/TLT/CASH backtest
# IMPROVED VERSION: Fixed lookahead bias, added features, ensemble models, transaction costs
# NOW WITH VISUALIZATIONS! (Compatible version)

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    classification_report, brier_score_loss
)

# Set style for better-looking plots
plt.style.use('default')  # Use default style for compatibility
sns.set_palette("husl")

print("=" * 80)
print("IMPROVED QUANT STRATEGY - Walk-Forward Validation with Proper Backtesting")
print("=" * 80)
print("RUNNING FILE:", os.path.abspath(__file__))

# =============================================================================
# 0) DOWNLOAD MULTI-ASSET PRICES
# =============================================================================
print("\n[1/8] Downloading price data...")

tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

prices = yf.download(
    tickers,
    start="2000-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

print(f"✓ Downloaded {len(prices)} days of data for {len(tickers)} tickers")
print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")

# =============================================================================
# 1) RESAMPLE TO MONTHLY & CALCULATE RETURNS
# =============================================================================
print("\n[2/8] Resampling to monthly frequency...")

monthly_prices = prices.resample("ME").last()
monthly_rets = monthly_prices.pct_change()

print(f"✓ Monthly data: {len(monthly_rets)} months")

# =============================================================================
# 2) ENGINEER FEATURES - EXPANDED SET
# =============================================================================
print("\n[3/8] Engineering features...")

features = pd.DataFrame(index=monthly_rets.index)

# --- Original spread features (market behavior) ---
features["risk_on_spread"] = monthly_rets["SPY"] - monthly_rets["TLT"]
features["growth_lead"] = monthly_rets["QQQ"] - monthly_rets["SPY"]
features["smallcaps_lead"] = monthly_rets["IWM"] - monthly_rets["SPY"]
features["gold_lead"] = monthly_rets["GLD"] - monthly_rets["SPY"]

# --- Momentum features (trailing returns) ---
features["spy_mom_3m"] = monthly_prices["SPY"].pct_change(3)
features["tlt_mom_3m"] = monthly_prices["TLT"].pct_change(3)
features["gld_mom_3m"] = monthly_prices["GLD"].pct_change(3)
features["spy_mom_6m"] = monthly_prices["SPY"].pct_change(6)
features["tlt_mom_6m"] = monthly_prices["TLT"].pct_change(6)
features["spy_mom_12m"] = monthly_prices["SPY"].pct_change(12)

# --- Volatility features ---
features["spy_vol_3m"] = monthly_rets["SPY"].rolling(3).std()
features["spy_vol_6m"] = monthly_rets["SPY"].rolling(6).std()
features["tlt_vol_3m"] = monthly_rets["TLT"].rolling(3).std()
features["spy_tlt_vol_ratio"] = features["spy_vol_3m"] / (features["tlt_vol_3m"] + 1e-6)

# --- Trend features ---
spy_ma6 = monthly_prices["SPY"].rolling(6).mean()
features["spy_ma_ratio_6m"] = (monthly_prices["SPY"] / spy_ma6) - 1
tlt_ma6 = monthly_prices["TLT"].rolling(6).mean()
features["tlt_ma_ratio_6m"] = (monthly_prices["TLT"] / tlt_ma6) - 1

# --- Relative strength ---
features["spy_tlt_mom_diff_3m"] = features["spy_mom_3m"] - features["tlt_mom_3m"]
features["spy_tlt_mom_diff_6m"] = features["spy_mom_6m"] - features["tlt_mom_6m"]

# --- Cross-asset correlation ---
cov_spy_tlt = monthly_rets["SPY"].rolling(12).cov(monthly_rets["TLT"])
var_tlt = monthly_rets["TLT"].rolling(12).var()
features["spy_tlt_beta_12m"] = cov_spy_tlt / (var_tlt + 1e-6)

print(f"✓ Created {len(features.columns)} features")

# =============================================================================
# 3) DEFINE TARGET
# =============================================================================
print("\n[4/8] Defining target variable...")

target = (monthly_rets["SPY"].shift(-1) > monthly_rets["TLT"].shift(-1)).astype(int)
data = features.copy()
data["target"] = target
data_clean = data.dropna()

print(f"✓ Target created: predict if SPY > TLT next month")
print(f"  Clean dataset: {len(data_clean)} months")

# =============================================================================
# 4) WALK-FORWARD VALIDATION
# =============================================================================
print("\n[5/8] Running walk-forward validation...")

TRAIN_WINDOW = 120
TEST_WINDOW = 12
STEP_SIZE = 12

X = data_clean.drop(columns=["target"])
y = data_clean["target"]

oos_predictions = []
window_metrics = []

print(f"\nWalk-forward parameters:")
print(f"  Train window: {TRAIN_WINDOW} months")
print(f"  Test window: {TEST_WINDOW} months")
print(f"  Step size: {STEP_SIZE} months\n")

for i, start_idx in enumerate(range(0, len(X) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE)):
    train_end = start_idx + TRAIN_WINDOW
    test_end = train_end + TEST_WINDOW
    
    X_train = X.iloc[start_idx:train_end]
    y_train = y.iloc[start_idx:train_end]
    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]
    
    train_dates = X_train.index
    test_dates = X_test.index
    
    if y_test.nunique() < 2:
        print(f"Window {i+1}: SKIPPED")
        continue
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    model_lr = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
    
    model_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    model_gb.fit(X_train_scaled, y_train)
    proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
    
    proba_ensemble = (proba_lr + proba_rf + proba_gb) / 3
    
    # Calculate thresholds from training data only
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
    acc = accuracy_score(y_test, (proba_ensemble >= 0.5).astype(int))
    brier = brier_score_loss(y_test, proba_ensemble)
    
    for j, date in enumerate(test_dates):
        oos_predictions.append({
            'date': date,
            'true_label': y_test.iloc[j],
            'proba': proba_ensemble[j],
            'regime': regime_test[j],
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'window': i + 1
        })
    
    window_metrics.append({
        'window': i + 1,
        'train_start': train_dates[0],
        'train_end': train_dates[-1],
        'test_start': test_dates[0],
        'test_end': test_dates[-1],
        'auc': auc,
        'accuracy': acc,
        'brier': brier,
        'n_test': len(y_test)
    })
    
    print(f"Window {i+1:2d}: {test_dates[0].date()} to {test_dates[-1].date()} | AUC={auc:.3f}")

df_predictions = pd.DataFrame(oos_predictions).set_index('date')
df_metrics = pd.DataFrame(window_metrics)

print(f"\n✓ Completed {len(df_metrics)} windows")
print(f"\nMean AUC: {df_metrics['auc'].mean():.3f}")

# =============================================================================
# 5) MODEL EVALUATION
# =============================================================================
print("\n[6/8] Evaluating model performance...")

y_true_all = df_predictions['true_label'].values
proba_all = df_predictions['proba'].values
pred_all = (proba_all >= 0.5).astype(int)

cm = confusion_matrix(y_true_all, pred_all)
print("\n--- Confusion Matrix ---")
print(f"Actual TLT>SPY: {cm[0,0]:6d} correct, {cm[0,1]:6d} wrong")
print(f"Actual SPY>TLT: {cm[1,0]:6d} wrong, {cm[1,1]:6d} correct")

# =============================================================================
# 6) BACKTEST WITH TRANSACTION COSTS
# =============================================================================
print("\n[7/8] Running backtest...")

bt_dates = df_predictions.index
spy_returns = monthly_rets.loc[bt_dates, "SPY"].shift(-1)
tlt_returns = monthly_rets.loc[bt_dates, "TLT"].shift(-1)
regimes = df_predictions['regime'].values

TC_BPS = 5
TC_RATE = TC_BPS / 10000

strategy_returns = pd.Series(0.0, index=bt_dates)
regime_series = pd.Series(regimes, index=bt_dates)

prev_regime = 0
n_trades = 0

for i, date in enumerate(bt_dates):
    current_regime = regimes[i]
    position_change = (current_regime != prev_regime)
    
    if current_regime == 1:
        base_return = spy_returns.iloc[i]
    elif current_regime == -1:
        base_return = tlt_returns.iloc[i]
    else:
        base_return = 0.0
    
    if position_change and i > 0:
        tc_cost = TC_RATE * 2
        strategy_returns.iloc[i] = base_return - tc_cost if not pd.isna(base_return) else -tc_cost
        n_trades += 1
    else:
        strategy_returns.iloc[i] = base_return if not pd.isna(base_return) else 0.0
    
    prev_regime = current_regime

buyhold_spy = spy_returns.copy()
buyhold_tlt = tlt_returns.copy()

bt = pd.DataFrame({
    'strategy': strategy_returns,
    'spy': buyhold_spy,
    'tlt': buyhold_tlt,
    'regime': regime_series
}).dropna()

bt['strategy_growth'] = (1 + bt['strategy']).cumprod()
bt['spy_growth'] = (1 + bt['spy']).cumprod()
bt['tlt_growth'] = (1 + bt['tlt']).cumprod()

def calc_metrics(returns_series):
    total_return = (1 + returns_series).prod() - 1
    ann_return = (1 + total_return) ** (12 / len(returns_series)) - 1
    ann_vol = returns_series.std() * np.sqrt(12)
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    calmar = abs(ann_return / max_dd) if max_dd < 0 else 0
    
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar
    }

metrics_strategy = calc_metrics(bt['strategy'])
metrics_spy = calc_metrics(bt['spy'])
metrics_tlt = calc_metrics(bt['tlt'])

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print(f"\nPeriod: {bt.index[0].date()} to {bt.index[-1].date()}")
print(f"Number of months: {len(bt)}")
print(f"Number of trades: {n_trades}")

print("\n--- Performance Comparison ---")
print(f"{'Metric':<20} {'Strategy':>12} {'SPY B&H':>12} {'TLT B&H':>12}")
print("-" * 60)
print(f"{'Final $1':<20} ${metrics_strategy['total_return']+1:>11.3f} "
      f"${metrics_spy['total_return']+1:>11.3f} ${metrics_tlt['total_return']+1:>11.3f}")
print(f"{'Annual Return':<20} {metrics_strategy['ann_return']*100:>11.1f}% "
      f"{metrics_spy['ann_return']*100:>11.1f}% {metrics_tlt['ann_return']*100:>11.1f}%")
print(f"{'Sharpe Ratio':<20} {metrics_strategy['sharpe']:>11.2f} "
      f"{metrics_spy['sharpe']:>11.2f} {metrics_tlt['sharpe']:>11.2f}")
print(f"{'Max Drawdown':<20} {metrics_strategy['max_dd']*100:>11.1f}% "
      f"{metrics_spy['max_dd']*100:>11.1f}% {metrics_tlt['max_dd']*100:>11.1f}%")

# =============================================================================
# 7) CREATE VISUALIZATIONS
# =============================================================================
print("\n[8/8] Creating visualizations...")

plot_dir = "backtest_plots"
os.makedirs(plot_dir, exist_ok=True)

try:
    # PLOT 1: Equity Curve
    print("  Creating equity curve...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt.index, bt['strategy_growth'], label='Strategy', linewidth=2, color='#2E86AB')
    ax.plot(bt.index, bt['spy_growth'], label='SPY', linewidth=2, color='#A23B72', linestyle='--')
    ax.plot(bt.index, bt['tlt_growth'], label='TLT', linewidth=2, color='#F18F01', linestyle='--')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.set_title('Equity Curve: Growth of $1', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/1_equity_curve.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/1_equity_curve.png")
    
    # PLOT 2: Drawdown
    print("  Creating drawdown chart...")
    def calc_drawdown_series(growth_series):
        running_max = growth_series.cummax()
        return ((growth_series - running_max) / running_max) * 100
    
    dd_strategy = calc_drawdown_series(bt['strategy_growth'])
    dd_spy = calc_drawdown_series(bt['spy_growth'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(bt.index, dd_strategy, 0, alpha=0.3, color='#2E86AB', label='Strategy')
    ax.plot(bt.index, dd_strategy, linewidth=2, color='#2E86AB')
    ax.plot(bt.index, dd_spy, linewidth=1.5, color='#A23B72', linestyle='--', label='SPY')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/2_drawdown.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/2_drawdown.png")
    
    # PLOT 3: Monthly Returns Heatmap
    print("  Creating returns heatmap...")
    returns_for_heatmap = bt['strategy'].copy()
    returns_pivot = pd.DataFrame({
        'Year': returns_for_heatmap.index.year,
        'Month': returns_for_heatmap.index.month,
        'Return': returns_for_heatmap.values * 100
    })
    returns_matrix = returns_pivot.pivot(index='Year', columns='Month', values='Return')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returns_matrix.columns = [month_names[int(m)-1] for m in returns_matrix.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(returns_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, linewidths=0.5, ax=ax, vmin=-10, vmax=10)
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/3_monthly_heatmap.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/3_monthly_heatmap.png")
    
    # PLOT 4: Regime Allocation
    print("  Creating regime allocation...")
    fig, ax = plt.subplots(figsize=(12, 6))
    regime_df = pd.DataFrame({
        'SPY': (bt['regime'] == 1).astype(int),
        'CASH': (bt['regime'] == 0).astype(int),
        'TLT': (bt['regime'] == -1).astype(int)
    }, index=bt.index)
    
    ax.fill_between(regime_df.index, 0, regime_df['TLT'], label='TLT', alpha=0.7, color='#F18F01')
    ax.fill_between(regime_df.index, regime_df['TLT'], regime_df['TLT'] + regime_df['CASH'],
                    label='CASH', alpha=0.7, color='#90A959')
    ax.fill_between(regime_df.index, regime_df['TLT'] + regime_df['CASH'], 1, 
                    label='SPY', alpha=0.7, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Allocation', fontsize=11)
    ax.set_title('Strategy Allocation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/4_regime_allocation.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/4_regime_allocation.png")
    
    # PLOT 5: AUC Over Time
    print("  Creating AUC chart...")
    window_centers = []
    for _, row in df_metrics.iterrows():
        center_date = row['test_start'] + (row['test_end'] - row['test_start']) / 2
        window_centers.append(center_date)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(window_centers, df_metrics['auc'], marker='o', linewidth=2, markersize=6, color='#6A4C93')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random (0.5)')
    mean_auc = df_metrics['auc'].mean()
    ax.axhline(y=mean_auc, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Mean={mean_auc:.3f}')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('AUC Score', fontsize=11)
    ax.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/5_auc_over_time.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/5_auc_over_time.png")
    
    # PLOT 6: Annual Returns
    print("  Creating annual returns...")
    bt_annual = bt.copy()
    bt_annual['year'] = bt_annual.index.year
    annual_returns = bt_annual.groupby('year')[['strategy', 'spy', 'tlt']].apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(annual_returns))
    width = 0.25
    ax.bar(x - width, annual_returns['strategy'], width, label='Strategy', color='#2E86AB', alpha=0.8)
    ax.bar(x, annual_returns['spy'], width, label='SPY', color='#A23B72', alpha=0.8)
    ax.bar(x + width, annual_returns['tlt'], width, label='TLT', color='#F18F01', alpha=0.8)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=11)
    ax.set_title('Annual Returns Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(annual_returns.index, rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/6_annual_returns.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {plot_dir}/6_annual_returns.png")
    
    print(f"\n✓ All 6 plots saved to '{plot_dir}/' directory")
    print(f"\nTo view plots:")
    print(f"  Mac:     open {plot_dir}/")
    print(f"  Or in VS Code: Click on the {plot_dir} folder in the sidebar")
    
except Exception as e:
    print(f"\n⚠ Error creating plots: {e}")
    print("  But don't worry - the backtest results are still valid!")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\n✓ Script completed successfully!")

