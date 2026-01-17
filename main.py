#!/usr/bin/env python3
"""
Zeus Trader: AI-Powered Quantitative Trading Bot
=================================================
Main entry point for training, backtesting, and prediction.

Usage:
    python main.py              # Run optimizer with grid search
    python main.py --predict    # Generate prediction for tomorrow
    python main.py --backtest   # Run single backtest with default config
"""

import argparse
import sys

from config import CONFIG
from data_fetcher import fetch_data
from feature_engine import add_indicators, normalize_data
from model import ZeusLSTM, create_sequences, train_model, predict, device, get_device_info
from backtest import run_backtest
from optimizer import run_grid_search
from sentiment import get_market_sentiment, get_sentiment_factor


def run_prediction():
    """Generate prediction for the next trading day."""
    print("=" * 60)
    print("⚡ ZEUS TRADER: AI Price Prediction")
    print("=" * 60)
    print(f"🚀 Device: {get_device_info()}")
    
    # Fetch data
    df = fetch_data()
    df = add_indicators(df)
    
    # Prepare features
    feature_cols = [c for c in CONFIG["features"] if c in df.columns]
    scaled_data, scaler, _ = normalize_data(df, feature_cols)
    
    # Train on all data
    lookback = CONFIG["model"]["lookback"]
    X, y = create_sequences(scaled_data, lookback)
    
    print(f"\n🏋️ Training Model...")
    model = ZeusLSTM(
        input_dim=len(feature_cols),
        hidden_dim=CONFIG["model"]["lstm_units"],
        num_layers=CONFIG["model"]["lstm_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)
    
    train_model(model, X, y, verbose=True)
    
    # Predict next day
    print("\n🔮 Generating Prediction...")
    
    last_sequence = scaled_data[-lookback:]
    last_sequence = last_sequence.reshape(1, lookback, -1)
    
    pred_scaled = predict(model, last_sequence)[0]
    
    # Denormalize
    min_price = scaler.data_min_[0]
    max_price = scaler.data_max_[0]
    predicted_price = pred_scaled * (max_price - min_price) + min_price
    
    current_price = df['Close'].iloc[-1]
    
    # Get sentiment adjustment
    sentiment_score = get_market_sentiment()
    sentiment_factor = get_sentiment_factor(sentiment_score)
    adjusted_price = predicted_price * sentiment_factor
    
    # Display results
    print("\n" + "=" * 60)
    print(f"💵 Current Price:     ₹{current_price:,.2f}")
    print(f"🤖 Model Forecast:    ₹{predicted_price:,.2f}")
    print(f"📰 Sentiment Factor:  x{sentiment_factor:.4f}")
    print(f"🔮 FINAL PREDICTION:  ₹{adjusted_price:,.2f}")
    
    change_pct = ((adjusted_price - current_price) / current_price) * 100
    
    if change_pct > 0.5:
        signal = "🚀 AGGRESSIVE BUY"
    elif change_pct > 0:
        signal = "📈 MILD BUY"
    elif change_pct < -0.5:
        signal = "🔻 AGGRESSIVE SELL"
    elif change_pct < 0:
        signal = "📉 MILD SELL"
    else:
        signal = "➖ HOLD"
    
    print(f"📢 ZEUS SIGNAL:       {signal} ({change_pct:+.2f}%)")
    print("=" * 60)


def run_single_backtest():
    """Run a single backtest with default configuration."""
    print("=" * 60)
    print("🔙 ZEUS TRADER: Backtest Simulation")
    print("=" * 60)
    print(f"🚀 Device: {get_device_info()}")
    
    # Fetch data
    df = fetch_data()
    df = add_indicators(df)
    
    # Prepare features
    feature_cols = [c for c in CONFIG["features"] if c in df.columns]
    scaled_data, scaler, _ = normalize_data(df, feature_cols)
    
    # Split
    test_days = CONFIG["backtest"]["test_days"]
    lookback = CONFIG["model"]["lookback"]
    
    split_idx = len(scaled_data) - test_days
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - lookback:]
    
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    real_prices = df['Close'].iloc[split_idx:].values
    
    print(f"\n📊 Data Split: Train {len(X_train)} | Test {len(X_test)}")
    
    # Train
    print(f"\n🏋️ Training Model...")
    model = ZeusLSTM(
        input_dim=len(feature_cols),
        hidden_dim=CONFIG["model"]["lstm_units"],
        num_layers=CONFIG["model"]["lstm_layers"],
        dropout=CONFIG["model"]["dropout"]
    ).to(device)
    
    train_model(model, X_train, y_train, verbose=True)
    
    # Backtest
    print(f"\n🔙 Running Backtest...")
    result = run_backtest(model, X_test, y_test, real_prices, scaler, verbose=True)
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Zeus Trader: AI-Powered Quantitative Trading Bot"
    )
    parser.add_argument(
        "--predict", "-p",
        action="store_true",
        help="Generate prediction for next trading day"
    )
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run single backtest with default config"
    )
    parser.add_argument(
        "--optimize", "-o",
        action="store_true",
        help="Run grid search optimization (default)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "⚡" * 30)
    print("       ZEUS TRADER v2.0 - Python Edition")
    print("⚡" * 30 + "\n")
    
    if args.predict:
        run_prediction()
    elif args.backtest:
        run_single_backtest()
    else:
        # Default: run optimizer
        run_grid_search()


if __name__ == "__main__":
    main()
