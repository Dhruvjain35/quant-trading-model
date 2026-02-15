# Multi-Asset Tactical Allocation Model

An ML-powered trading strategy that dynamically switches between stocks (SPY), bonds (TLT), and cash based on market conditions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

##  Overview

This quantitative trading model uses ensemble machine learning to predict whether stocks or bonds will outperform in the next month. It combines three algorithms:
- **Logistic Regression** - Linear baseline
- **Random Forest** - Captures non-linear patterns  
- **Gradient Boosting** - Sequential learning

##  Features

- **22 Market Indicators**: Momentum, volatility, trend, and correlation features
- **Walk-Forward Validation**: Proper out-of-sample testing (no lookahead bias)
- **Transaction Costs**: Realistic backtesting with 5 bps trading costs
- **Interactive Dashboard**: Beautiful Streamlit web app
- **Professional Visualizations**: Equity curves, drawdowns, heatmaps

##  Performance

- **Backtest Period**: 2013-2025 (131 months)
- **Strategy**: Dynamic allocation between SPY/TLT/CASH
- **Mean AUC**: ~0.49
- **Sharpe Ratio**: 0.17 (strategy) vs 1.02 (SPY)
- **Max Drawdown**: -20.6% (strategy) vs -23.9% (SPY)

##  Installation
```bash
# Clone the repository
git clone https://github.com/Inkspire-Custom-Arts/quant-trading-model.git
cd quant-trading-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

##  Usage

### Option 1: Terminal Version (CLI)
```bash
python3 lesson1_spy_returns.py
```

### Option 2: Web App (Interactive Dashboard)
```bash
streamlit run trading_app.py
```
Opens at `http://localhost:8501` with interactive charts!

##  What You Get

1. **Equity Curve** - Growth of $1 over time
2. **Drawdown Chart** - Risk visualization
3. **Monthly Returns Heatmap** - Calendar view
4. **Regime Allocation** - Strategy decisions over time
5. **Model AUC** - Prediction quality
6. **Annual Returns** - Year-by-year comparison

##  How It Works

1. **Data Collection**: Downloads historical prices (SPY, QQQ, IWM, TLT, GLD)
2. **Feature Engineering**: Creates 22 technical indicators
3. **Model Training**: Walk-forward validation (no lookahead!)
4. **Signal Generation**: Ensemble prediction → regime decision
5. **Backtesting**: Realistic simulation with transaction costs

## Disclaimer

**This project is for educational purposes only.**
- Not financial advice
- Past performance ≠ future results
- Trading involves risk of loss

## License

MIT License

---

**Built with Python, scikit-learn, and Streamlit** 
