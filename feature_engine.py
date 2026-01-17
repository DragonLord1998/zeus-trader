"""
Zeus Trader Enhanced Feature Engine
====================================
Comprehensive technical indicator library with 15+ indicators for LSTM prediction.
Based on research: best-performing LSTM stock prediction features for Indian NSE.
"""

import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict


# Feature Categories for Benchmarking
FEATURE_SETS = {
    "minimal": ["Close", "Volume"],
    
    "basic": ["Close", "Volume", "RSI", "MACD", "SMA_50"],
    
    "momentum": ["Close", "Volume", "RSI", "MACD", "Stochastic_K", "Stochastic_D", 
                 "ROC", "Williams_R"],
    
    "trend": ["Close", "Volume", "SMA_20", "SMA_50", "EMA_12", "EMA_26", 
              "ADX", "Aroon_Up", "Aroon_Down"],
    
    "volatility": ["Close", "Volume", "BB_Upper", "BB_Lower", "BB_Width", 
                   "ATR", "Keltner_High", "Keltner_Low"],
    
    "volume_based": ["Close", "Volume", "OBV", "MFI", "VWAP", "CMF", "Force_Index"],
    
    "full_technical": [
        "Close", "Volume", "RSI", "MACD", "MACD_Signal", 
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "BB_Upper", "BB_Lower", "BB_Width",
        "ATR", "ADX", "Stochastic_K", "Stochastic_D",
        "OBV", "MFI", "CCI", "Williams_R", "ROC"
    ],
    
    "macro_only": ["Close", "Volume", "NIFTY", "OIL", "GOLD", "SILVER", 
                   "COPPER", "US10Y", "DXY"],
    
    "full_macro": [
        "Close", "Volume", "RSI", "MACD", "SMA_50",
        "BB_Upper", "BB_Lower", "ATR", "ADX",
        "NIFTY", "OIL", "GOLD", "SILVER", "COPPER", "US10Y", "DXY"
    ],
    
    "kitchen_sink": [  # All available features
        "Close", "Volume", "RSI", "MACD", "MACD_Signal",
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "BB_Upper", "BB_Lower", "BB_Width",
        "ATR", "ADX", "Stochastic_K", "Stochastic_D",
        "OBV", "MFI", "CCI", "Williams_R", "ROC",
        "Aroon_Up", "Aroon_Down", "Force_Index",
        "NIFTY", "OIL", "GOLD", "SILVER", "COPPER", "US10Y", "DXY"
    ],
}


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to the dataframe.
    
    Categories:
        - Trend indicators (SMA, EMA, ADX, Aroon)
        - Momentum indicators (RSI, MACD, Stochastic, ROC, Williams %R)
        - Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
        - Volume indicators (OBV, MFI, VWAP, CMF, Force Index)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with 25+ indicator columns
    """
    print("⚙️  Calculating Comprehensive Technical Indicators...")
    df = df.copy()
    
    # Ensure we have required columns
    close = df['Close']
    high = df['High'] if 'High' in df.columns else df['Close']
    low = df['Low'] if 'Low' in df.columns else df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series([0] * len(df), index=df.index)
    
    # ==================== TREND INDICATORS ====================
    print("   📈 Adding Trend Indicators...")
    
    # Simple Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    df['SMA_200'] = ta.trend.sma_indicator(close, window=200)
    
    # Exponential Moving Averages
    df['EMA_12'] = ta.trend.ema_indicator(close, window=12)
    df['EMA_26'] = ta.trend.ema_indicator(close, window=26)
    
    # ADX (Average Directional Index) - Trend Strength
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df['ADX'] = adx_indicator.adx()
    df['DI_Plus'] = adx_indicator.adx_pos()
    df['DI_Minus'] = adx_indicator.adx_neg()
    
    # Aroon Indicator - Trend Direction
    aroon = ta.trend.AroonIndicator(high, low, window=25)
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    
    # ==================== MOMENTUM INDICATORS ====================
    print("   🚀 Adding Momentum Indicators...")
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.rsi(close, window=14)
    
    # MACD
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Hist'] = macd_indicator.macd_diff()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()
    
    # Rate of Change (ROC)
    df['ROC'] = ta.momentum.roc(close, window=10)
    
    # Williams %R
    df['Williams_R'] = ta.momentum.williams_r(high, low, close, lbp=14)
    
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.trend.cci(high, low, close, window=20)
    
    # ==================== VOLATILITY INDICATORS ====================
    print("   📊 Adding Volatility Indicators...")
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Percent'] = bb.bollinger_pband()
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Keltner Channels
    kelt = ta.volatility.KeltnerChannel(high, low, close, window=20)
    df['Keltner_High'] = kelt.keltner_channel_hband()
    df['Keltner_Low'] = kelt.keltner_channel_lband()
    
    # ==================== VOLUME INDICATORS ====================
    print("   📦 Adding Volume Indicators...")
    
    # On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    
    # Money Flow Index (MFI)
    df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
    
    # Chaikin Money Flow (CMF)
    df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    
    # Force Index
    df['Force_Index'] = ta.volume.force_index(close, volume, window=13)
    
    # VWAP approximation (using typical price * volume cumsum / volume cumsum)
    typical_price = (high + low + close) / 3
    df['VWAP'] = (typical_price * volume).cumsum() / volume.cumsum()
    
    # ==================== DERIVED FEATURES ====================
    print("   🧮 Adding Derived Features...")
    
    # Price relative to moving averages
    df['Price_to_SMA20'] = close / df['SMA_20']
    df['Price_to_SMA50'] = close / df['SMA_50']
    
    # Trend strength
    df['Trend_Strength'] = (df['EMA_12'] - df['EMA_26']) / close * 100
    
    # Volatility normalized
    df['ATR_Percent'] = df['ATR'] / close * 100
    
    # Drop rows with NaN (typically first 200 due to SMA_200)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    print(f"   ✅ Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} indicators")
    print(f"   ✅ Dropped {dropped} rows with incomplete data")
    
    return df


def normalize_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """
    Normalize data using MinMaxScaler.
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to normalize
    
    Returns:
        Tuple of (scaled_data array, scaler object, available_cols)
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if len(available_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"   ⚠️ Missing columns: {missing}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[available_cols].values)
    
    return scaled_data, scaler, available_cols


def denormalize_price(scaled_value: float, scaler: MinMaxScaler, price_idx: int = 0) -> float:
    """Denormalize a scaled price value back to original scale."""
    min_val = scaler.data_min_[price_idx]
    max_val = scaler.data_max_[price_idx]
    return scaled_value * (max_val - min_val) + min_val


def get_feature_importance_proxy(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'Close') -> Dict[str, float]:
    """
    Calculate correlation-based feature importance as a proxy.
    Higher absolute correlation = more predictive potential.
    
    Args:
        df: DataFrame with features
        feature_cols: Columns to analyze
        target_col: Target column for prediction
    
    Returns:
        Dict of feature -> correlation score
    """
    target_returns = df[target_col].pct_change().shift(-1)  # Next day returns
    importance = {}
    
    for col in feature_cols:
        if col in df.columns and col != target_col:
            corr = df[col].corr(target_returns)
            importance[col] = abs(corr) if not np.isnan(corr) else 0
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    from data_fetcher import fetch_data
    
    df = fetch_data()
    df = add_all_indicators(df)
    
    print(f"\n📊 Total Features: {len(df.columns)}")
    print(f"📊 Available Feature Sets: {list(FEATURE_SETS.keys())}")
    
    # Show feature importance
    importance = get_feature_importance_proxy(df, list(df.columns))
    print("\n🎯 Top 10 Correlated Features (to next-day returns):")
    for feat, score in list(importance.items())[:10]:
        print(f"   {feat}: {score:.4f}")
