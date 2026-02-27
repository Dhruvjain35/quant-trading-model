# Adaptive Macro-Regime Trading Engine (V8.3)

A quantitative trading backtester that shifts market exposure based on macroeconomic regimes. Built to prioritize out-of-sample robustness and absolute returns by avoiding common retail pitfalls like overfitting and cash drag.

## The V8.3 Architecture
Earlier versions of this model attempted to dynamically scale out of positions during drawdowns. While that capped max losses, empirical testing showed it created massive cash drag during V-shaped market recoveries. V8.3 reverts to a strict binary (1 or 0) risk-on/risk-off allocation. It’s a simpler, more aggressive approach that captures momentum significantly better while still defending capital during structural breakdowns.

## Core Methodology
* **Macro Regime Shifting:** Uses a state-space model to toggle exposure based on macroeconomic indicators.
* **Anti-Overfitting:** Implements Purged Embargo walk-forward validation to prevent data leakage between training and testing sets.
* **Statistical Rigor:** Uses Permutation Testing to generate a p-value for the Sharpe ratio, proving the performance isn't just random luck.
* **Interpretable AI:** Integrates SHAP (SHapley Additive exPlanations) to visualize exactly which macro variables drive the model's decisions—no black boxes.
* **Real-World Friction:** Explicitly accounts for dynamic transaction costs and bid-ask slippage to ensure out-of-sample realism.

## Tech Stack
* **Core:** `pandas`, `numpy`, `scipy`
* **ML:** `scikit-learn`, `xgboost`, `shap`
* **UI:** `streamlit`, `plotly`

## Quick Start

1. Clone the repo and navigate to the directory:
   ```bash
   git clone [https://github.com/yourusername/macro-regime-engine.git](https://github.com/yourusername/macro-regime-engine.git)
   cd macro-regime-engine

