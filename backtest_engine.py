import pandas as pd
import numpy as np

def run_backtest(probs_df, returns_df, config):
    """
    Vectorized backtest with Transaction Costs and Slippage.
    """
    tc_bps = config['backtest']['transaction_cost_bps'] / 10000
    slip_bps = config['backtest']['slippage_bps'] / 10000
    total_friction = tc_bps + slip_bps
    
    risk_asset = returns_df.columns[0]
    safe_asset = returns_df.columns[1]
    
    # Align dates
    bt_data = probs_df.join(returns_df).dropna()
    
    # Allocation Logic: Thresholding based on probabilities
    # 1 = Risk On (SPY), 0 = Risk Off (TLT)
    threshold = config['model']['voting_soft_threshold']
    bt_data['Target_Weight'] = np.where(bt_data['Prob_Risk_On'] >= threshold, 1.0, 0.0)
    
    # Calculate Turnover (Change in position)
    bt_data['Weight_Shift'] = bt_data['Target_Weight'].shift(1)
    bt_data['Turnover'] = bt_data['Target_Weight'].diff().abs().fillna(0)
    
    # Calculate Gross Returns (Using Shifted Weights to avoid look-ahead)
    bt_data['Gross_Ret'] = (bt_data['Weight_Shift'] * bt_data[risk_asset]) + \
                           ((1 - bt_data['Weight_Shift']) * bt_data[safe_asset])
    
    # Apply Transaction Costs (Applied only on days where turnover > 0)
    bt_data['Net_Ret'] = bt_data['Gross_Ret'] - (bt_data['Turnover'] * total_friction)
    
    # Benchmark is Buy and Hold Risk Asset
    bt_data['Benchmark_Ret'] = bt_data[risk_asset]
    
    # Cumulative curves
    bt_data['Cum_Strat'] = (1 + bt_data['Net_Ret']).cumprod()
    bt_data['Cum_Bench'] = (1 + bt_data['Benchmark_Ret']).cumprod()
    
    return bt_data
