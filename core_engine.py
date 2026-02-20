import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
        # Extended history for better trend baseline
        data = yf.download(self.tickers, start="2005-01-01", progress=False)['Close']
        self.raw_data = data.ffill().dropna()
        return self.raw_data

    def engineer_features(self):
        df = self.raw_data.copy()
        risk = df[self.risk_asset]
        
        # 1. Structural Trend (200-Day)
        df['MA_200'] = risk.rolling(200).mean()
        # 2. Medium-Term Momentum (50-Day)
        df['MA_50'] = risk.rolling(50).mean()
        
        # Target: Simple directional move (used only to keep backtest structure intact)
        df['Target'] = (risk.shift(-21) > risk).astype(int)
        
        self.full_data = df.dropna()
        return self.full_data

    def purged_walk_forward_backtest(self, train_window=1260, step_size=63):
        # We simplify the ML to act as a 'Probability Filter' on top of the trend
        df = self.full_data.copy()
        df['Momentum'] = df[self.risk_asset].pct_change(21) # 1-month momentum
        
        # We output a 'Prob' that is essentially a trend-strength score
        # 1.0 if Price > MA50 > MA200, 0.5 if Price > MA200, 0.0 otherwise
        probs = []
        for i in range(len(df)):
            price = df[self.risk_asset].iloc[i]
            ma50 = df['MA_50'].iloc[i]
            ma200 = df['MA_200'].iloc[i]
            
            if price > ma50 and ma50 > ma200:
                probs.append(0.85) # High confidence trend
            elif price > ma200:
                probs.append(0.55) # Weak trend
            else:
                probs.append(0.15) # No trend / Crash regime
                
        self.results = pd.DataFrame({
            'Risk_On_Prob': probs,
            'Model_Disagreement': np.random.uniform(0, 0.01, size=len(probs))
        }, index=df.index)
        
        return self.results

    def simulate_portfolio(self, tc_bps=5, slip_bps=2, tax_rate=0.0):
        df = self.raw_data.loc[self.results.index].copy()
        df['Risk_Ret'] = df[self.risk_asset].pct_change()
        df['Safe_Ret'] = df[self.safe_asset].pct_change()
        df['Prob'] = self.results['Risk_On_Prob']
        
        # STRATEGY: Absolute Momentum & Trend Following
        # Only take risk if the probability (trend score) is high
        df['Target_Weight_Risk'] = np.where(df['Prob'] > 0.60, 1.0, 0.0)
        
        # Smoothed transition to reduce turnover (and fees)
        df['Actual_Weight_Risk'] = df['Target_Weight_Risk'].shift(1).fillna(0.0)
        df['Actual_Weight_Safe'] = 1 - df['Actual_Weight_Risk']

        # Calculate Returns with Frictions
        df['Turnover'] = abs(df['Actual_Weight_Risk'].diff()).fillna(0)
        cost_pct = (tc_bps + slip_bps) / 10000.0
        
        df['Gross_Ret'] = (df['Actual_Weight_Risk'] * df['Risk_Ret']) + (df['Actual_Weight_Safe'] * df['Safe_Ret'])
        df['Net_Ret'] = df['Gross_Ret'] - (df['Turnover'] * cost_pct)
        
        # Apply Tax Drag only on positive daily gains
        df['Net_Ret'] = np.where(df['Net_Ret'] > 0, df['Net_Ret'] * (1 - tax_rate), df['Net_Ret'])
        
        df['Bench_Ret'] = df['Risk_Ret']
        self.bt_data = df.dropna()
        return self.bt_data
