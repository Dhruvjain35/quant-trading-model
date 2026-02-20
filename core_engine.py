import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

class InstitutionalAMCE:
    def __init__(self, risk_asset="QQQ", safe_asset="SHY"):
        self.risk_asset = risk_asset
        self.safe_asset = safe_asset
        self.tickers = [self.risk_asset, self.safe_asset]
        self.raw_data = None
        self.full_data = None
        self.results = None
        self.bt_data = None

    def fetch_data(self):
        # Fetching adjusted closing prices
        data = yf.download(self.tickers, start="2008-01-01", progress=False)['Close']
        self.raw_data = data.ffill().dropna()
        return self.raw_data

    def engineer_features(self):
        df = self.raw_data.copy()
        risk = df[self.risk_asset]
        safe = df[self.safe_asset]
        
        # Base Returns
        df['Risk_Ret'] = risk.pct_change()
        
        # 1. Volatility Regime
        df['Vol_1M'] = df['Risk_Ret'].rolling(21).std() * np.sqrt(252)
        
        # 2. Multi-Timeframe Momentum (1M, 3M, 6M)
        df['Mom_1M'] = risk / risk.shift(21) - 1
        df['Mom_3M'] = risk / risk.shift(63) - 1
        df['Mom_6M'] = risk / risk.shift(126) - 1
        
        # 3. Cross Asset Strength (Is Risk outperforming Safe?)
        df['Risk_Safe_Spread'] = (risk / risk.shift(63)) - (safe / safe.shift(63))
        
        # 4. MACD-style Trend Proxy (Fast vs Slow Moving Average)
        df['Trend_Oscillator'] = (risk.rolling(21).mean() / risk.rolling(63).mean()) - 1
        
        # TARGET: Predict the 21-day (1 month) forward return direction to minimize turnover
        # 1 if price in 21 days is strictly higher than today, else 0
        df['Target'] = (risk.shift(-21) > risk).astype(int)
        
        self.full_data = df.dropna()
        return self.full_data

    def purged_walk_forward_backtest(self, train_window=756, step_size=63):
        df = self.full_data.copy()
        # Using the upgraded feature set
        features = ['Vol_1M', 'Mom_1M', 'Mom_3M', 'Mom_6M', 'Risk_Safe_Spread', 'Trend_Oscillator']
        
        X = df[features]
        y = df['Target']
        
        predictions = []
        indices = []
        disagreements = []
        
        # Heavily regularized models to prevent overfitting to noise
        models = {
            'LR': LogisticRegression(C=0.1, random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=10, random_state=42),
            'GB': GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
        }
        
        for i in range(train_window, len(df) - step_size, step_size):
            train_X = X.iloc[i - train_window : i]
            train_y = y.iloc[i - train_window : i]
            test_X = X.iloc[i : i + step_size]
            
            probs = []
            for name, model in models.items():
                model.fit(train_X, train_y)
                probs.append(model.predict_proba(test_X)[:, 1])
            
            probs_array = np.array(probs)
            ensemble_prob = probs_array.mean(axis=0)
            disagreement = probs_array.var(axis=0)
            
            predictions.extend(ensemble_prob)
            disagreements.extend(disagreement)
            indices.extend(test_X.index)
            
        self.results = pd.DataFrame({
            'Risk_On_Prob': predictions,
            'Model_Disagreement': disagreements
        }, index=indices)
        
        return self.results

    def simulate_portfolio(self, tc_bps=5, slip_bps=2, tax_rate=0.0):
        """Simulates capital allocation with real-world frictions and strict confidence thresholds."""
        df = self.raw_data.loc[self.results.index].copy()
        df['Risk_Ret'] = df[self.risk_asset].pct_change()
        df['Safe_Ret'] = df[self.safe_asset].pct_change()
        df['Prob'] = self.results['Risk_On_Prob']
        
        # STRICT POSITION SIZING TO REDUCE TURNOVER
        # Requires high confidence to take risk; otherwise stays in cash/safe asset
        def size_position(prob):
            if prob > 0.60: return 1.0  
            elif prob < 0.45: return 0.0 
            else: return (prob - 0.45) / 0.15 

        df['Target_Weight_Risk'] = df['Prob'].apply(size_position)
        df['Target_Weight_Safe'] = 1 - df['Target_Weight_Risk']

        df['Actual_Weight_Risk'] = df['Target_Weight_Risk'].shift(1).fillna(1.0)
        df['Actual_Weight_Safe'] = df['Target_Weight_Safe'].shift(1).fillna(0.0)

        # FRICTIONS & RETURNS
        df['Turnover'] = abs(df['Actual_Weight_Risk'].diff()).fillna(0)
        friction_decimal = (tc_bps + slip_bps) / 10000.0
        df['Execution_Cost'] = df['Turnover'] * friction_decimal

        df['Gross_Ret'] = (df['Actual_Weight_Risk'] * df['Risk_Ret']) + (df['Actual_Weight_Safe'] * df['Safe_Ret'])
        df['Pre_Tax_Ret'] = df['Gross_Ret'] - df['Execution_Cost']

        df['Tax_Hit'] = np.where(df['Pre_Tax_Ret'] > 0, df['Pre_Tax_Ret'] * tax_rate, 0)
        df['Net_Ret'] = df['Pre_Tax_Ret'] - df['Tax_Hit']
        
        df['Bench_Ret'] = df['Risk_Ret']

        self.bt_data = df.dropna()
        return self.bt_data
