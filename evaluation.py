import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def calculate_metrics(bt_data):
    """Calculates institutional risk-adjusted metrics."""
    strat_rets = bt_data['Net_Ret'].dropna()
    bench_rets = bt_data['Benchmark_Ret'].dropna()
    
    ann_factor = 252
    
    # Basic Metrics
    ann_ret = strat_rets.mean() * ann_factor
    vol = strat_rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / vol if vol != 0 else 0
    
    # Drawdown
    cum_returns = (1 + strat_rets).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    max_dd = drawdowns.min()
    
    # OLS Alpha and Beta (Statistical Significance)
    # y = alpha + beta * x
    X = sm.add_constant(bench_rets)
    model = sm.OLS(strat_rets, X).fit()
    
    alpha_daily = model.params.iloc[0]
    beta = model.params.iloc[1]
    alpha_ann = alpha_daily * ann_factor
    p_value_alpha = model.pvalues.iloc[0]
    t_stat_alpha = model.tvalues.iloc[0]
    
    # Bootstrapped Sharpe Confidence Interval
    n_bootstraps = 1000
    boot_sharpes = []
    rets_array = strat_rets.values
    
    for _ in range(n_bootstraps):
        sample = np.random.choice(rets_array, size=len(rets_array), replace=True)
        s_sharpe = (np.mean(sample) / np.std(sample)) * np.sqrt(ann_factor)
        boot_sharpes.append(s_sharpe)
        
    sharpe_5th = np.percentile(boot_sharpes, 5)
    sharpe_95th = np.percentile(boot_sharpes, 95)
    
    return {
        "Ann_Ret": ann_ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sharpe_95CI": (sharpe_5th, sharpe_95th),
        "Max_DD": max_dd,
        "Alpha": alpha_ann,
        "Beta": beta,
        "Alpha_t_stat": t_stat_alpha,
        "Alpha_p_value": p_value_alpha,
        "Avg_Turnover": bt_data['Turnover'].mean() * ann_factor
    }
