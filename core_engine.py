import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class InstitutionalAMCE:
    def __init__(self, risk_asset="QQQ", safe_asset="SHY", horizon=5):
        self.risk_asset = risk_asset
        self.safe_asset = safe_asset
        self.horizon = horizon # Predict 5 days out to avoid daily noise
        self.raw_data = None
        self.features = None
        self.target = None
        self.models = {}

    def fetch_data(self):
        """Fetches base assets and generates a proxy for macro stress (VIX alternative)."""
        tickers = [self.risk_asset, self.safe_asset]
      data = yf.download(tickers, start="2008-01-01", progress=False)['Close']
        # Forward fill missing data, drop entirely empty rows
        self.raw_data = data.ffill().dropna()

    def engineer_features(self):
        """Builds a stationary, institution-grade feature space."""
        df = pd.DataFrame(index=self.raw_data.index)
        risk_price = self.raw_data[self.risk_asset]
        safe_price = self.raw_data[self.safe_asset]

        # 1. Cross-Asset Momentum (Equity vs Bond strength)
        df['Risk_Ret_1M'] = risk_price.pct_change(21)
        df['Safe_Ret_1M'] = safe_price.pct_change(21)
        df['Cross_Asset_Strength'] = df['Risk_Ret_1M'] - df['Safe_Ret_1M']

        # 2. Volatility Term Structure (Short vs Long Term Vol)
        df['Vol_1W'] = risk_price.pct_change().rolling(5).std() * np.sqrt(252)
        df['Vol_1M'] = risk_price.pct_change().rolling(21).std() * np.sqrt(252)
        df['Vol_3M'] = risk_price.pct_change().rolling(63).std() * np.sqrt(252)
        df['Vol_Term_Structure'] = df['Vol_1W'] / df['Vol_1M'] # > 1 means acute panic

        # 3. Macro Correlation (When equities and bonds fall together = liquidity crisis)
        df['Eq_Bond_Corr'] = risk_price.pct_change().rolling(63).corr(safe_price.pct_change())

        # 4. Trend Oscillators (Z-Scored to maintain stationarity)
        df['Trend_60D'] = (risk_price - risk_price.rolling(60).mean()) / risk_price.rolling(60).std()
        df['Trend_200D'] = (risk_price - risk_price.rolling(200).mean()) / risk_price.rolling(200).std()

        # TARGET VARIABLE: Will Risk Asset beat Safe Asset over the next 'horizon' days?
        future_risk_ret = risk_price.pct_change(self.horizon).shift(-self.horizon)
        future_safe_ret = safe_price.pct_change(self.horizon).shift(-self.horizon)
        
        # 1 if Equity Risk Premium is positive, 0 otherwise
        df['Target'] = np.where(future_risk_ret > future_safe_ret, 1, 0)

        self.full_data = df.dropna()
        self.X = self.full_data.drop(columns=['Target'])
        self.y = self.full_data['Target']

    def train_ensemble(self, X_train, y_train):
        """Constructs the tri-factor ensemble model."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Base Models
        lr = LogisticRegression(C=0.1, class_weight='balanced', random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)

        # Soft Voting Ensemble (averages probabilities)
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        return ensemble, scaler

    def purged_walk_forward_backtest(self, train_window=756, step_size=63, embargo=10):
        """
        Walk-forward validation with an embargo period to strictly prevent data leakage.
        train_window: ~3 years of trading days
        step_size: Retrain every ~3 months
        embargo: Days skipped between train and test to purge overlapping return horizons
        """
        predictions = pd.Series(index=self.full_data.index, dtype=float)
        disagreement = pd.Series(index=self.full_data.index, dtype=float)

        # Step through time
        for start_idx in range(0, len(self.full_data) - train_window - embargo, step_size):
            end_train_idx = start_idx + train_window
            start_test_idx = end_train_idx + embargo
            end_test_idx = min(start_test_idx + step_size, len(self.full_data))

            X_train = self.X.iloc[start_idx:end_train_idx]
            y_train = self.y.iloc[start_idx:end_train_idx]
            X_test = self.X.iloc[start_test_idx:end_test_idx]

            if len(np.unique(y_train)) > 1: # Ensure we have both classes in training data
                ensemble, scaler = self.train_ensemble(X_train, y_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Get soft voting probability of class 1 (Risk-On)
                probs = ensemble.predict_proba(X_test_scaled)[:, 1]
                predictions.iloc[start_test_idx:end_test_idx] = probs
                
                # Calculate Disagreement (Variance between the 3 models' predictions)
                # This is an advanced alpha feature for the UI later
                ind_preds = np.array([model.predict_proba(X_test_scaled)[:, 1] for model in ensemble.estimators_])
                disagreement.iloc[start_test_idx:end_test_idx] = np.std(ind_preds, axis=0)

        # Clean up NaNs from the initial training window
        self.results = pd.DataFrame({
            'Risk_On_Prob': predictions,
            'Model_Disagreement': disagreement
        }).dropna()
        
        return self.results

def simulate_portfolio(self, tc_bps=5, slip_bps=2, tax_rate=0.0):
        """Simulates capital allocation with real-world frictions (Fees, Slippage, Taxes)."""
        df = self.raw_data.loc[self.results.index].copy()
        df['Risk_Ret'] = df[self.risk_asset].pct_change()
        df['Safe_Ret'] = df[self.safe_asset].pct_change()
        df['Prob'] = self.results['Risk_On_Prob']
        
        # CONTINUOUS POSITION SIZING
        def size_position(prob):
            if prob > 0.55: return 1.0  
            elif prob < 0.45: return 0.0 
            else: return (prob - 0.45) / 0.10 

        df['Target_Weight_Risk'] = df['Prob'].apply(size_position)
        df['Target_Weight_Safe'] = 1 - df['Target_Weight_Risk']

        df['Actual_Weight_Risk'] = df['Target_Weight_Risk'].shift(1).fillna(1.0)
        df['Actual_Weight_Safe'] = df['Target_Weight_Safe'].shift(1).fillna(0.0)

        # 1. CALCULATE TURNOVER
        df['Turnover'] = abs(df['Actual_Weight_Risk'].diff()).fillna(0)
        
        # 2. EXECUTION FRICTIONS (Transaction Costs + Market Slippage)
        friction_decimal = (tc_bps + slip_bps) / 10000.0
        df['Execution_Cost'] = df['Turnover'] * friction_decimal

        # 3. GROSS RETURNS
        df['Gross_Ret'] = (df['Actual_Weight_Risk'] * df['Risk_Ret']) + (df['Actual_Weight_Safe'] * df['Safe_Ret'])
        df['Pre_Tax_Ret'] = df['Gross_Ret'] - df['Execution_Cost']

        # 4. TAX DRAG (Proxy for Short-Term Capital Gains)
        df['Tax_Hit'] = np.where(df['Pre_Tax_Ret'] > 0, df['Pre_Tax_Ret'] * tax_rate, 0)
        
        # 5. FINAL NET RETURN
        df['Net_Ret'] = df['Pre_Tax_Ret'] - df['Tax_Hit']
        
        # Benchmark Returns
        df['Bench_Ret'] = df['Risk_Ret']

        self.bt_data = df.dropna()
        return self.bt_data
