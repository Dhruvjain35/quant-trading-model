import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class InstitutionalQuantEngine:
    def __init__(self, risk_asset, safe_asset, start_date="2005-01-01"):
        self.risk_asset = risk_asset
        self.safe_asset = safe_asset
        self.start_date = start_date
        self.data = None
        
    def fetch_data(self):
        """Fetches data and handles yfinance MultiIndex issues."""
        df = yf.download([self.risk_asset, self.safe_asset], start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Adj Close'] if 'Adj Close' in df.columns.levels[0] else df['Close']
        else:
            df = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        self.data = df.dropna()
        return self.data

    def engineer_features(self):
        """
        STRICT NO DATA LEAKAGE PROTOCOL.
        All features use strictly backward-looking rolling windows.
        """
        df = self.data.copy()
        X = pd.DataFrame(index=df.index)
        returns = df.pct_change().dropna()
        
        # 1. Macro & Momentum Features (Available at time t)
        X['Mom_3M'] = df[self.risk_asset].pct_change(63)
        X['Mom_6M'] = df[self.risk_asset].pct_change(126)
        X['Vol_1M'] = returns[self.risk_asset].rolling(21, min_periods=21).std() * np.sqrt(252)
        X['Vol_3M'] = returns[self.risk_asset].rolling(63, min_periods=63).std() * np.sqrt(252)
        X['Yield_Proxy'] = df[self.safe_asset].pct_change(63)
        X['Risk_Spread'] = X['Mom_3M'] - X['Yield_Proxy']
        
        # 2. TARGET CREATION (Available at time t+1)
        # We shift the target BACKWARD so that row 't' contains the feature at 't' and the target for 't+1'
        target_returns = returns[self.risk_asset].shift(-1)
        y = (target_returns > 0).astype(int)
        
        # Drop NaNs to ensure perfectly aligned indices
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        self.X = X.loc[valid_idx]
        self.y = y.loc[valid_idx]
        self.actual_returns = returns.loc[valid_idx]
        
        return self.X, self.y

    def walk_forward_validation(self, train_window=1000, step=252):
        """
        Expanding Window Walk-Forward Validation.
        The model never sees test data during training.
        """
        predictions = pd.Series(index=self.X.index, dtype=float)
        all_shap_values = []
        
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, 
            objective='binary:logistic', eval_metric='logloss', random_state=42
        )
        
        for i in range(train_window, len(self.X), step):
            # Train strictly on past data [0 to i]
            X_train = self.X.iloc[:i]
            y_train = self.y.iloc[:i]
            
            # Test strictly on future, unseen data [i to i+step]
            test_end = min(i + step, len(self.X))
            X_test = self.X.iloc[i:test_end]
            
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            predictions.iloc[i:test_end] = probs
            
            # SHAP Explainability (on the test set)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            all_shap_values.append(shap_vals)
            
        self.predictions = predictions.dropna()
        return self.predictions

    def backtest_with_frictions(self, tc_bps, slippage_bps, tax_rate, threshold=0.52):
        """
        Realistic execution modeling.
        """
        bt = pd.DataFrame(index=self.predictions.index)
        bt['Prob'] = self.predictions
        
        # Signal Generation
        bt['Target_Weight'] = np.where(bt['Prob'] > threshold, 1.0, 0.0)
        
        # Shift weight to t+1 to simulate executing at tomorrow's close based on today's signal
        bt['Exec_Weight'] = bt['Target_Weight'].shift(1).fillna(0)
        bt['Turnover'] = bt['Exec_Weight'].diff().abs().fillna(0)
        
        # Costs
        friction = (tc_bps + slippage_bps) / 10000.0
        
        # Returns
        bt['Gross_Ret'] = (bt['Exec_Weight'] * self.actual_returns[self.risk_asset].loc[bt.index]) + \
                          ((1 - bt['Exec_Weight']) * self.actual_returns[self.safe_asset].loc[bt.index])
        
        bt['Net_Ret'] = bt['Gross_Ret'] - (bt['Turnover'] * friction)
        
        # Crude Tax Approximation (applied to positive returns when turnover occurs)
        tax_impact = np.where((bt['Turnover'] > 0) & (bt['Net_Ret'] > 0), bt['Net_Ret'] * tax_rate, 0)
        bt['Post_Tax_Ret'] = bt['Net_Ret'] - tax_impact
        
        bt['Bench_Ret'] = self.actual_returns[self.risk_asset].loc[bt.index]
        bt['Cum_Strat'] = (1 + bt['Post_Tax_Ret']).cumprod()
        bt['Cum_Bench'] = (1 + bt['Bench_Ret']).cumprod()
        
        # Drawdown
        bt['Strat_Peak'] = bt['Cum_Strat'].cummax()
        bt['Strat_DD'] = (bt['Cum_Strat'] / bt['Strat_Peak']) - 1
        
        self.bt_results = bt
        return bt

    def calculate_statistics(self, n_bootstraps=1000):
        """Rigorous Institutional Stats."""
        rets = self.bt_results['Post_Tax_Ret']
        bench = self.bt_results['Bench_Ret']
        
        ann_ret = rets.mean() * 252
        vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol != 0 else 0
        
        # Bootstrapped Sharpe Confidence Interval
        boot_sharpes = []
        rets_arr = rets.values
        for _ in range(n_bootstraps):
            sample = np.random.choice(rets_arr, size=len(rets_arr), replace=True)
            s_sharpe = (np.mean(sample) * 252) / (np.std(sample) * np.sqrt(252))
            boot_sharpes.append(s_sharpe)
            
        ci_lower = np.percentile(boot_sharpes, 2.5)
        ci_upper = np.percentile(boot_sharpes, 97.5)
        
        # Advanced Stats
        skew = stats.skew(rets)
        kurt = stats.kurtosis(rets)
        max_dd = self.bt_results['Strat_DD'].min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "CAGR": ann_ret, "Vol": vol, "Sharpe": sharpe, 
            "Sharpe_95CI": (ci_lower, ci_upper),
            "Max_DD": max_dd, "Calmar": calmar,
            "Skew": skew, "Kurtosis": kurt,
            "Total_Ret": self.bt_results['Cum_Strat'].iloc[-1] - 1,
            "Bench_Total": self.bt_results['Cum_Bench'].iloc[-1] - 1
        }
