"""
Zeus Trader Data Fetcher
========================
Fetches historical market data and macro assets from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys

from config import CONFIG


def fetch_target_data(symbol: str = None, days: int = 1095) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for the target asset.
    
    Args:
        symbol: Ticker symbol (default: from CONFIG)
        days: Number of days of history to fetch (default: 3 years)
    
    Returns:
        DataFrame with OHLCV data
    """
    if symbol is None:
        symbol = CONFIG["default_target"]
    
    print(f"📡 Fetching data for {symbol}...")
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    df = yf.download(symbol, start=start_date, progress=False)
    
    if df.empty:
        print(f"❌ Error: No data found for {symbol}")
        sys.exit(1)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"✅ Fetched {len(df)} days of data for {symbol}")
    return df


def fetch_macro_assets(start_date: str) -> pd.DataFrame:
    """
    Fetch macro correlated assets (Oil, Gold, Silver, Copper, US10Y, DXY, Nifty).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with macro asset prices aligned by date
    """
    print("📡 Fetching Macro Assets...")
    macro_data = {}
    
    for name, ticker in CONFIG["macro_assets"].items():
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if not data.empty:
                macro_data[name] = data['Close']
                print(f"   ✅ {name}: {len(data)} days")
            else:
                print(f"   ⚠️ {name}: No data")
                
        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")
    
    return pd.DataFrame(macro_data)


def fetch_data(symbol: str = None, days: int = 1095) -> pd.DataFrame:
    """
    Fetch target asset + all macro correlators, merged and forward-filled.
    
    Args:
        symbol: Target ticker symbol
        days: Number of days of history
    
    Returns:
        DataFrame with target OHLCV + all macro prices
    """
    # Fetch target
    target_df = fetch_target_data(symbol, days)
    
    # Fetch macro assets
    start_date = target_df.index.min().strftime('%Y-%m-%d')
    macro_df = fetch_macro_assets(start_date)
    
    # Align macro data to target dates and forward-fill
    macro_df = macro_df.reindex(target_df.index)
    macro_df = macro_df.ffill().bfill()
    
    # Merge
    df = pd.concat([target_df, macro_df], axis=1)
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    print(f"✅ Total merged data: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_quote(symbol: str) -> dict:
    """
    Get current quote for a symbol.
    
    Args:
        symbol: Ticker symbol
    
    Returns:
        Dict with current price info
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "price": info.get("regularMarketPrice", 0),
            "change": info.get("regularMarketChange", 0),
            "changePercent": info.get("regularMarketChangePercent", 0),
        }
    except Exception as e:
        print(f"❌ Error fetching quote for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_data()
    print("\n📊 Data Sample:")
    print(df.tail())
