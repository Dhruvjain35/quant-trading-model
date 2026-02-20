import pandas as pd
import numpy as np

def engineer_features(price_df):
    """
    Constructs institutional-grade features avoiding look-ahead bias.
    price_df expects columns: ['SPY', 'TLT']
    """
    df = pd.DataFrame(index=price_df.index)
    risk_asset = price_df.columns[0]
    safe_asset = price_df.columns[1]
    
    # 1. Momentum / Trend (Lagged automatically by rolling)
    df['Mom_3M'] = price_df[risk_asset].pct_change(63)
    df['Mom_6M'] = price_df[risk_asset].pct_change(126)
    df['Mom_12M'] = price_df[risk_asset].pct_change(252)
    
    # 2. Volatility (Realized rolling)
    returns = price_df.pct_change()
    df['Vol_1M'] = returns[risk_asset].rolling(21).std() * np.sqrt(252)
    df['Vol_3M'] = returns[risk_asset].rolling(63).std() * np.sqrt(252)
    
    # 3. Cross-Asset Dynamics / Macro Proxies
    # Flight to safety proxy (Safe asset momentum relative to Risk asset)
    df['Risk_On_Spread'] = df['Mom_3M'] - price_df[safe_asset].pct_change(63)
    
    # Term Structure / Rate proxy (using Safe Asset volatility)
    df['Rate_Vol_Proxy'] = returns[safe_asset].rolling(21).std() * np.sqrt(252)

    # 4. Drawdowns (Information available at time t)
    rolling_max = price_df[risk_asset].rolling(252, min_periods=1).max()
    df['Drawdown'] = (price_df[risk_asset] / rolling_max) - 1.0

    # Clean NaNs caused by rolling windows
    df = df.dropna()
    
    # 5. TARGET VARIABLE (STRICTLY SHIFTED - NO LEAKAGE)
    # Target: 1 if risk asset return at t+1 is > 0
    target_returns = returns[risk_asset].shift(-1)
    target = (target_returns > 0).astype(int)
    
    # Align indices (drops the last row where target is NaN)
    common_idx = df.index.intersection(target.dropna().index)
    
    return df.loc[common_idx], target.loc[common_idx], returns.loc[common_idx]
