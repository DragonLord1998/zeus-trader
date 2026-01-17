"""
Zeus Trader Configuration
=========================
Centralized configuration for the trading system.
"""

CONFIG = {
    # Global Market Settings
    "market": "IN",  # India
    "timezone": "Asia/Kolkata",
    
    # Target Assets to Predict
    "targets": [
        {"symbol": "^NSEI", "name": "Nifty 50"},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    ],
    
    # Default target for single-stock mode
    "default_target": "RELIANCE.NS",
    
    # Macro Correlated Assets (The "World Context")
    "macro_assets": {
        "NIFTY": "^NSEI",
        "OIL": "CL=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "COPPER": "HG=F",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
    },
    
    # Features used for model input
    "features": [
        "Close", "Volume", "RSI", "MACD", "SMA_50",
        "NIFTY", "OIL", "GOLD", "SILVER", "COPPER", "US10Y", "DXY"
    ],
    
    # Model Hyperparameters
    "model": {
        "lookback": 60,      # Days of history to look at
        "epochs": 30,        # Training iterations
        "batch_size": 64,    # Batch size for training
        "lstm_units": 512,   # Neurons per LSTM layer
        "lstm_layers": 4,    # Number of LSTM layers (Deep Learning)
        "dropout": 0.2,      # Dropout rate
        "learning_rate": 0.001,
    },
    
    # Backtest Settings
    "backtest": {
        "test_days": 200,           # Days to use for backtesting
        "initial_capital": 100000,  # Starting capital (INR)
        "threshold": 0.005,         # 0.5% movement threshold for signals
    },
    
    # News Sentiment Settings
    "sentiment": {
        "enabled": True,
        "topic": "NIFTY+50+economy+india",
        "lang": "en-IN",
    },
    
    # Grid Search Parameter Grid
    "param_grid": [
        {"units": 512, "layers": 4, "threshold": 0.005},
        {"units": 256, "layers": 4, "threshold": 0.005},
        {"units": 512, "layers": 3, "threshold": 0.005},
        {"units": 512, "layers": 5, "threshold": 0.005},
        {"units": 512, "layers": 4, "threshold": 0.004},
        {"units": 512, "layers": 4, "threshold": 0.006},
        {"units": 1024, "layers": 4, "threshold": 0.004},
    ],
}
