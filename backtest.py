"""
Zeus Trader Backtester
======================
Simulates trading strategy using trained model predictions.
"""

import numpy as np
from typing import Dict, List, Any

from config import CONFIG
from model import ZeusLSTM, predict, device
from feature_engine import denormalize_price


def run_backtest(
    model: ZeusLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    real_prices: np.ndarray,
    scaler,
    threshold: float = None,
    initial_capital: float = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run backtest simulation on test data.
    
    Strategy:
        - BUY when predicted price > last close * (1 + threshold)
        - SELL when predicted price < last close * (1 - threshold)
        - HOLD otherwise
    
    Args:
        model: Trained ZeusLSTM model
        X_test: Test features (samples, lookback, features)
        y_test: Test labels (scaled)
        real_prices: Actual close prices (unscaled)
        scaler: Fitted MinMaxScaler
        threshold: Signal threshold (e.g., 0.005 = 0.5%)
        initial_capital: Starting capital
        verbose: Print trade log
    
    Returns:
        Dict with backtest results
    """
    if threshold is None:
        threshold = CONFIG["backtest"]["threshold"]
    if initial_capital is None:
        initial_capital = CONFIG["backtest"]["initial_capital"]
    
    # Get predictions
    predictions_scaled = predict(model, X_test)
    
    # Initialize
    balance = initial_capital
    shares = 0
    trades = 0
    wins = 0
    trade_log = []
    
    for i in range(len(predictions_scaled)):
        if i == 0:
            continue
        
        # Get scaled values
        last_close_scaled = X_test[i][-1][0]  # Last timestep, Close (index 0)
        pred_scaled = predictions_scaled[i]
        current_price = real_prices[i]
        
        action = "HOLD"
        
        # BUY signal: prediction > threshold above last close
        if pred_scaled > last_close_scaled * (1 + threshold):
            if balance > current_price:
                can_buy = int(balance // current_price)
                balance -= can_buy * current_price
                shares += can_buy
                action = "BUY"
                trade_log.append(f"🟢 BUY  @ ₹{current_price:.2f} x{can_buy} shares")
        
        # SELL signal: prediction > threshold below last close
        elif pred_scaled < last_close_scaled * (1 - threshold):
            if shares > 0:
                balance += shares * current_price
                trade_log.append(f"🔴 SELL @ ₹{current_price:.2f} x{shares} shares")
                shares = 0
                action = "SELL"
        
        # Track accuracy
        if action != "HOLD":
            trades += 1
            
            # Check if direction was correct
            actual_move = real_prices[i] > real_prices[i - 1]
            pred_move = pred_scaled > last_close_scaled
            
            if actual_move == pred_move:
                wins += 1
    
    # Final liquidation
    final_value = balance + (shares * real_prices[-1])
    profit = final_value - initial_capital
    roi = (profit / initial_capital) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    # Buy & Hold benchmark
    buy_hold_roi = ((real_prices[-1] - real_prices[0]) / real_prices[0]) * 100
    
    if verbose:
        print("\n📜 Trade Log:")
        for log in trade_log[-10:]:  # Show last 10 trades
            print(f"   {log}")
        
        print(f"\n📊 Backtest Results:")
        print(f"   💰 Initial: ₹{initial_capital:,.0f}")
        print(f"   🏁 Final:   ₹{final_value:,.2f}")
        print(f"   📈 ROI:     {roi:+.2f}%")
        print(f"   🎲 Trades:  {trades}")
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
        print(f"   🐢 Buy&Hold: {buy_hold_roi:+.2f}%")
    
    return {
        "roi": round(roi, 2),
        "profit": round(profit, 2),
        "final_value": round(final_value, 2),
        "trades": trades,
        "win_rate": round(win_rate, 1),
        "buy_hold_roi": round(buy_hold_roi, 2),
        "logs": trade_log,
    }


def compare_strategies(
    model: ZeusLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    real_prices: np.ndarray,
    scaler,
    thresholds: List[float] = [0.003, 0.005, 0.008]
) -> List[Dict]:
    """
    Compare multiple threshold strategies.
    
    Args:
        thresholds: List of threshold values to test
    
    Returns:
        List of results sorted by ROI
    """
    results = []
    
    for thresh in thresholds:
        result = run_backtest(
            model, X_test, y_test, real_prices, scaler,
            threshold=thresh, verbose=False
        )
        result["threshold"] = thresh
        results.append(result)
    
    results.sort(key=lambda x: x["roi"], reverse=True)
    return results


if __name__ == "__main__":
    print("🔙 Backtest module loaded.")
    print("   Run from main.py for full backtest simulation.")
