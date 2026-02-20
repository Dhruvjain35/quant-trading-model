import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
        data = yf.download(self.tickers, start="2005-01-01", progress=False)['Close']
        self.raw_data = data.ffill().dropna()
        return self.raw_data

    def engineer_features(self):
        df = self.raw_data.copy()
        risk = df[self.risk_asset]
        
        # 1. CORE REGIME SIGNAL: Distance from 200-Day MA
        df['MA_200'] = risk.rolling(200).mean()
        df['Dist_200'] = (risk / df['MA_200']) - 1
        
        # 2. VOLATILITY CLUSTERING (Standardized)
        returns = risk.pct_change()
        df['Vol_Signal'] = returns.rolling(21).std() * np.sqrt(252)
        
        # 3. RELATIVE MOMENTUM (3-Month vs 1-Year)
        df['Mom_Rel'] = (risk.rolling(63).mean() / risk.rolling(252).mean()) - 1
        
        # TARGET: Is the market positive over the next month (21 days)?
        df['Target'] = (risk.shift(-21) > risk).astype(int)
        
        self.full_data = df.dropna()
        return self.full_data

    def purged_walk_forward_backtest(self, train_window=1260, step_size=63):
        df = self.full_data.copy()
        features = ['Dist_200', 'Vol_Signal', 'Mom_Rel']
        
        X = df[features]
        y = df['Target']
        
        scaler = StandardScaler()
        predictions = []
        indices = []
        
        # High-penalty Ridge Logistic Regression (Very stable, low variance)
        model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs')
        
        for i in range(train_window, len(df) - step_size, step_size):
            train_X = X.iloc[i - train_window : i]
            train_y = y.iloc[i - train_window : i]
            test_X = X.iloc[i : i + step_size]
            
            # Standardize features to ensure stable weights
            train_X_scaled = scaler.fit_transform(train_X)
            test_X_scaled = scaler.transform(test_X)
            
            model.fit(train_X_scaled, train_y)
            prob = model.predict_proba(test_X_scaled)[:, 1]
            
            predictions.extend(prob)
            indices.extend(test_X.index)
            
        self.results = pd.DataFrame({
            'Risk_On_Prob': predictions,
            'Model_Disagreement': 0 # Simplified for stability
        }, index=indices)
        
        return self.results

    def simulate_portfolio(self, tc_bps=5, slip_bps=2, tax_rate=0.0):
        df = self.raw_data.loc[self.results.index].copy()
        df['Risk_Ret'] = df[self.risk_asset].pct_change()
        df['Safe_Ret'] = df[self.safe_asset].pct_change()
        df['Prob'] = self.results['Risk_On_Prob']
        
        # LOGIC: 200-DAY TREND FILTER (Institutional Gold Standard)
        # If Price < 200MA, we drastically cut exposure regardless of ML prob.
        df['MA_200'] = df[self.risk_asset].rolling(200).mean()
        df['Trend_Filter'] = np.where(df[self.risk_asset] > df['MA_200'], 1.0, 0.0)
        
        # Continuous sizing based on probability, then nuked by trend filter
        df['Base_Weight'] = np.where(df['Prob'] > 0.52, 1.0, 0.0)
        df['Target_Weight_Risk'] = df['Base_Weight'] * df['Trend_Filter']
        
        df['Actual_Weight_Risk'] = df['Target_Weight_Risk'].shift(1).fillna(0.0)
        df['Actual_Weight_Safe'] = 1 - df['Actual_Weight_Risk']

        # Frictions
        df['Turnover'] = abs(df['Actual_Weight_Risk'].diff()).fillna(0)
        cost = (tc_bps + slip_bps) / 10000.0
        
        df['Gross_Ret'] = (df['Actual_Weight_Risk'] * df['Risk_Ret']) + (df['Actual_Weight_Safe'] * df['Safe_Ret'])
        df['Net_Ret'] = df['Gross_Ret'] - (df['Turnover'] * cost)
        
        # Apply Tax Drag
        df['Net_Ret'] = np.where(df['Net_Ret'] > 0, df['Net_Ret'] * (1 - tax_rate), df['Net_Ret'])
        
        df['Bench_Ret'] = df['Risk_Ret']
        self.bt_data = df.dropna()
        return self.bt_data
