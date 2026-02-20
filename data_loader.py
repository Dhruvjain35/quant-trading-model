import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date):
    """
    Fetches adjusted close prices for the given tickers from yfinance.
    """
    # Download data
    data = yf.download(tickers, start=start_date)
    
    # Extract 'Adj Close' (or 'Close' if Adj Close isn't available)
    if 'Adj Close' in data.columns:
        df = data['Adj Close']
    else:
        df = data['Close']
        
    # Drop rows with NaN values to ensure clean data
    df = df.dropna()
    
    return df
