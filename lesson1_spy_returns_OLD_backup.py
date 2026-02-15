# lesson1_spy_returns.py
# Multi-asset "market behavior" model + walk-forward probs + SPY/TLT/CASH backtest

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

print("RUNNING FILE:", os.path.abspath(__file__))
print("RUNNING...")

# -----------------------------
# 0) Download multi-asset prices
# -----------------------------
tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

prices = yf.download(
    tickers,
    start="2000-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

print("\n=== MULTI-TICKER CHECK ===")
print("Columns in prices:", list(prices.columns))
print("Prices shape:", prices.shape)
print(prices.head())
print("=== END MULTI-TICKER CHECK ===\n")

# -----------------------------
# 1) Daily -> Monthly returns
# -----------------------------
monthly_prices = prices.resample("ME").last()
monthly_rets = monthly_prices.pct_change()

print("Monthly returns preview:")
print(monthly_rets.head(), "\n")

# -----------------------------
# 2) Features: market behavior
# -----------------------------
features = pd.DataFrame(index=monthly_rets.index)

# risk-on vs risk-off
features["risk_on_spread"] = monthly_rets["SPY"] - monthly_rets["TLT"]
# growth leadership
features["growth_lead"] = monthly_rets["QQQ"] - monthly_rets["SPY"]
# small cap appetite
features["smallcaps_lead"] = monthly_rets["IWM"] - monthly_rets["SPY"]
# gold vs stocks (fear/hedge proxy)
features["gold_lead"] = monthly_rets["GLD"] - monthly_rets["SPY"]
# volatility regime
features["spy_vol_3m"] = monthly_rets["SPY"].rolling(3).std()

print("Feature preview:")
print(features.head(10), "\n")

# -----------------------------
# 3) Target: next-month SPY up?
# -----------------------------
# Target: will SPY outperform TLT next month?
target = (monthly_rets["SPY"].shift(-1) > monthly_rets["TLT"].shift(-1)).astype(int)


data = features.copy()
data["target"] = target

# Drop NaNs (early months + rolling + ETF starts)
data = data.dropna()

print("Dataset shape:", data.shape)
print(data.head(10), "\n")

# -----------------------------
# 4) Simple time split baseline
# -----------------------------
split_date = "2015-01-01"
train = data.loc[data.index < split_date]
test  = data.loc[data.index >= split_date]

X_train = train.drop(columns=["target"])
y_train = train["target"]
X_test  = test.drop(columns=["target"])
y_test  = test["target"]

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(
    C=0.2,            # try 0.05, 0.1, 0.2, 0.5, 1.0
    penalty="l2",
    solver="lbfgs",
    max_iter=2000
)

model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
proba_up_test = model.predict_proba(X_test_s)[:, 1]

print("\n=== SIMPLE TEST (time split) ===")
print("Test accuracy:", round(acc, 3))
print("First 10 predicted probs (SPY up):", proba_up_test[:10])

weights = pd.Series(model.coef_[0], index=X_train.columns).sort_values(ascending=False)
print("\nFeature weights (higher pushes prob UP):")
print(weights)
print("=== END SIMPLE TEST ===\n")

# -----------------------------
# 5) Walk-forward validation
#    (collect probs for EACH test window)
# -----------------------------
window_train = 120   # months (~10 years)
window_test  = 24    # months (1 year)

X = data.drop(columns=["target"])
y = data["target"]

auc_scores = []
all_proba = pd.Series(dtype=float)  # predicted probs indexed by date
all_y     = pd.Series(dtype=int)    # true labels indexed by date

print("=== WALK-FORWARD VALIDATION ===")
for start in range(0, len(X) - window_train - window_test, window_test):
    train_slice = slice(start, start + window_train)
    test_slice  = slice(start + window_train, start + window_train + window_test)

    X_tr = X.iloc[train_slice]
    y_tr = y.iloc[train_slice]
    X_te = X.iloc[test_slice]
    y_te = y.iloc[test_slice]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    wf_model = LogisticRegression(C=0.5, max_iter=1000)
    wf_model.fit(X_tr_s, y_tr)

    proba = wf_model.predict_proba(X_te_s)[:, 1]
    test_dates = X_te.index

    # store probabilities + true labels with the correct dates
    all_proba = pd.concat([all_proba, pd.Series(proba, index=test_dates)])
    all_y     = pd.concat([all_y, y_te])

    # AUC for this window only if both classes present
    if y_te.nunique() == 2:
        auc = roc_auc_score(y_te, proba)
        auc_scores.append(auc)
        print(f"{test_dates[0].date()} -> AUC: {auc:.3f}")
    else:
        print(f"{test_dates[0].date()} -> AUC: skipped (only one class)")

if len(auc_scores) == 0:
    print("\nMean AUC: skipped (no valid windows with both classes)")
else:
    auc_series = pd.Series(auc_scores)
    print("\nMean AUC:", round(auc_series.mean(), 3))
    print("Min  AUC:", round(auc_series.min(), 3))
    print("Max  AUC:", round(auc_series.max(), 3))
print("=== END WALK-FORWARD ===\n")

# Clean up duplicates / ordering (important!)
proba_series = all_proba.sort_index()
y_series     = all_y.sort_index()

# If any duplicate dates got created, keep the last value
proba_series = proba_series[~proba_series.index.duplicated(keep="last")]
y_series     = y_series[~y_series.index.duplicated(keep="last")]

# Align both series on shared dates only
common_idx = proba_series.index.intersection(y_series.index)
proba_series = proba_series.loc[common_idx]
y_series     = y_series.loc[common_idx]

# -----------------------------
# 6) Backtest: SPY vs TLT vs CASH
# -----------------------------
test_index = proba_series.index  # use the dates we actually have probs for

# Next-month returns (what we'd earn if we take position at month-end)
spy_next = monthly_rets.loc[proba_series.index, "SPY"].shift(-1)
tlt_next = monthly_rets.loc[proba_series.index, "TLT"].shift(-1)


# -----------------------------
# SPY vs TLT only (no cash)
# -----------------------------

threshold = proba_series.median()
print("\nUsing median threshold:", round(threshold, 3))

regime = pd.Series(index=proba_series.index)
regime[proba_series >= threshold] = 1    # SPY
regime[proba_series < threshold] = -1    # TLT


print("\nProba summary:")
print(proba_series.describe())

print("\nRegime counts (months):")
print(regime.value_counts().sort_index())
print("\nRegime %:")
print((regime.value_counts(normalize=True).sort_index() * 100).round(1))

# Strategy returns
strategy_ret = pd.Series(0.0, index=test_index)
strategy_ret.loc[regime == 1]  = spy_next.loc[regime == 1]
strategy_ret.loc[regime == -1] = tlt_next.loc[regime == -1]
# regime == 0 stays 0 (cash)

# Buy & hold baseline (SPY)
buyhold_ret = spy_next.copy()

# Drop NaNs from shift(-1) at the end
bt = pd.DataFrame({
    "strategy": strategy_ret,
    "buyhold_spy": buyhold_ret
}).dropna()

# Growth of $1
bt["strategy_growth"] = (1 + bt["strategy"]).cumprod()
bt["buyhold_growth"]  = (1 + bt["buyhold_spy"]).cumprod()

print("\n=== BACKTEST RESULTS ===")
print("Start:", bt.index.min(), "End:", bt.index.max())
print("Final $1 (strategy):", round(bt["strategy_growth"].iloc[-1], 3))
print("Final $1 (buy&hold SPY):", round(bt["buyhold_growth"].iloc[-1], 3))

def sharpe(monthly_returns: pd.Series) -> float:
    if monthly_returns.std() == 0:
        return float("nan")
    return (monthly_returns.mean() / monthly_returns.std()) * (12 ** 0.5)

print("\nAvg monthly return (strategy):", round(bt["strategy"].mean(), 4))
print("Avg monthly return (SPY):", round(bt["buyhold_spy"].mean(), 4))
print("Sharpe (strategy):", round(sharpe(bt["strategy"]), 3))
print("Sharpe (SPY):", round(sharpe(bt["buyhold_spy"]), 3))

def max_drawdown(growth_series: pd.Series) -> float:
    peak = growth_series.cummax()
    dd = growth_series / peak - 1
    return dd.min()

print("\nMax drawdown (strategy):", round(max_drawdown(bt["strategy_growth"]), 3))
print("Max drawdown (SPY):", round(max_drawdown(bt["buyhold_growth"]), 3))
print("=== END BACKTEST ===")
