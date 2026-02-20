"""
Data Loader Module
Handles downloading and preprocessing financial data with proper date handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Optional
from datetime import datetime


class DataLoader:
    """
    Handles financial data ingestion with proper error handling
    and date alignment.
    """
    
    def __init__(self, start_date: str = "2000-01-01"):
        self.start_date = start_date
        self.data_cache = {}
    
    def download_asset(self, ticker: str) -> pd.Series:
        """
        Download a single asset's price history.
        
        Args:
            ticker: Asset ticker symbol
            
        Returns:
            pd.Series: Daily closing prices
        """
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        
        try:
            df = yf.download(ticker, start=self.start_date, progress=False, auto_adjust=True)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            elif 'Close' in df.columns:
                df = df['Close']
                
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0]
            
            self.data_cache[ticker] = df
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to download {ticker}: {e}")
    
    def download_vix(self) -> Optional[pd.Series]:
        """
        Download VIX volatility index.
        
        Returns:
            pd.Series or None: VIX daily values
        """
        try:
            return self.download_asset("^VIX")
        except:
            return None
    
    def create_dataset(
        self, 
        risky_asset: str, 
        safe_asset: str,
        include_vix: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """
        Create aligned price and return datasets.
        
        Args:
            risky_asset: High-beta asset ticker
            safe_asset: Risk-free asset ticker
            include_vix: Whether to download VIX
            
        Returns:
            prices: DataFrame with both assets
            returns: DataFrame with daily returns
            vix: VIX series if requested
        """
        # Download assets
        p_risky = self.download_asset(risky_asset)
        p_safe = self.download_asset(safe_asset)
        
        # Align dates
        prices = pd.concat([p_risky, p_safe], axis=1).dropna()
        prices.columns = [risky_asset, safe_asset]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # VIX
        vix = None
        if include_vix:
            vix = self.download_vix()
            if vix is not None:
                vix = vix.reindex(prices.index, method='ffill')
        
        return prices, returns, vix
    
    def get_date_range(self, data: pd.DataFrame) -> Tuple[datetime, datetime]:
        """Get start and end dates of dataset"""
        return data.index[0], data.index[-1]
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        train_pct: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test maintaining temporal order.
        
        Args:
            data: Time-series data
            train_pct: Fraction for training
            
        Returns:
            train_data, test_data
        """
        split_idx = int(len(data) * train_pct)
        return data.iloc[:split_idx], data.iloc[split_idx:]
