"""
Zeus Trader Comprehensive Benchmark Runner
==========================================
Multi-model, multi-feature benchmarking framework.
Compares: LSTM, TimesFM, and ensemble approaches.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from data_fetcher import fetch_data
from feature_engine import add_all_indicators, normalize_data, FEATURE_SETS
from model import ZeusLSTM, ZeusTransformer, create_sequences, train_model, predict, device, get_device_info
from backtest import run_backtest


class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(self, model_name: str, feature_set: str, params: dict):
        self.model_name = model_name
        self.feature_set = feature_set
        self.params = params
        self.roi = 0.0
        self.trades = 0
        self.win_rate = 0.0
        self.buy_hold_roi = 0.0
        self.train_time = 0.0
        self.mae = 0.0
        self.rmse = 0.0
        self.direction_accuracy = 0.0
    
    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "features": self.feature_set,
            "params": self.params,
            "roi": round(self.roi, 2),
            "trades": self.trades,
            "win_rate": round(self.win_rate, 1),
            "buy_hold": round(self.buy_hold_roi, 2),
            "alpha": round(self.roi - self.buy_hold_roi, 2),
            "train_time_sec": round(self.train_time, 1),
            "direction_accuracy": round(self.direction_accuracy, 1),
        }


def benchmark_lstm(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: List[str],
    params: dict,
    test_days: int = 200,
    lookback: int = 60
) -> BenchmarkResult:
    """
    Benchmark LSTM model with given features and parameters.
    """
    result = BenchmarkResult("LSTM", feature_set_name, params)
    
    # Get available features
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < 2:
        print(f"   ⚠️ Skipping - insufficient features")
        return result
    
    # Normalize
    scaled_data, scaler, feature_list = normalize_data(df, available_cols)
    
    # Split
    split_idx = len(scaled_data) - test_days
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - lookback:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    if len(X_train) < 10 or len(X_test) < 10:
        print(f"   ⚠️ Skipping - insufficient data")
        return result
    
    # Train
    start_time = time.time()
    
    model_type = params.get('model_type', 'LSTM')
    
    if model_type == 'Transformer':
        model = ZeusTransformer(
            input_dim=len(feature_list),
            d_model=params.get('units', 128),
            nhead=4,
            num_layers=params.get('layers', 2),
            dropout=params.get('dropout', 0.2)
        ).to(device)
    else:
        model = ZeusLSTM(
            input_dim=len(feature_list),
            hidden_dim=params.get('units', 256),
            num_layers=params.get('layers', 3),
            dropout=params.get('dropout', 0.2)
        ).to(device)
    
    train_model(
        model, X_train, y_train,
        epochs=params.get('epochs', 20),
        batch_size=params.get('batch_size', 64),
        verbose=False
    )
    
    result.train_time = time.time() - start_time
    
    # Get predictions for analysis
    predictions = predict(model, X_test)
    
    # Calculate direction accuracy
    actual_directions = np.diff(y_test) > 0
    pred_directions = np.diff(predictions) > 0
    result.direction_accuracy = np.mean(actual_directions == pred_directions) * 100
    
    # MAE and RMSE on scaled data
    result.mae = np.mean(np.abs(predictions - y_test))
    result.rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    
    # Backtest
    real_prices = df['Close'].iloc[split_idx:].values
    bt_result = run_backtest(
        model, X_test, y_test, real_prices, scaler,
        threshold=params.get('threshold', 0.005),
        verbose=False
    )
    
    result.roi = bt_result['roi']
    result.trades = bt_result['trades']
    result.win_rate = bt_result['win_rate']
    result.buy_hold_roi = bt_result['buy_hold_roi']
    
    return result


def try_import_timesfm():
    """Try to import TimesFM, return None if not available."""
    try:
        import timesfm
        return timesfm
    except ImportError:
        return None


def benchmark_timesfm(
    df: pd.DataFrame,
    feature_set_name: str,
    test_days: int = 200,
    horizon: int = 5
) -> Optional[BenchmarkResult]:
    """
    Benchmark TimesFM model (if available).
    TimesFM works with univariate time series - we use Close price.
    """
    timesfm = try_import_timesfm()
    if timesfm is None:
        return None
    
    result = BenchmarkResult("TimesFM", feature_set_name, {"horizon": horizon})
    
    try:
        start_time = time.time()
        
        # Initialize TimesFM
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",  # Use CPU for compatibility
                per_core_batch_size=32,
                horizon_len=horizon,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-200m-pytorch"
            ),
        )
        
        # Prepare data - univariate (Close prices)
        close_prices = df['Close'].values
        split_idx = len(close_prices) - test_days
        
        train_prices = close_prices[:split_idx]
        test_prices = close_prices[split_idx:]
        
        # Forecast
        context = train_prices[-512:]  # Use last 512 points as context
        forecast_input = np.array([context])
        
        point_forecast, _ = tfm.forecast(forecast_input)
        
        result.train_time = time.time() - start_time
        
        # Calculate metrics on first horizon prediction
        if len(point_forecast) > 0 and len(point_forecast[0]) > 0:
            pred = point_forecast[0][:min(horizon, len(test_prices))]
            actual = test_prices[:len(pred)]
            
            result.mae = np.mean(np.abs(pred - actual))
            result.rmse = np.sqrt(np.mean((pred - actual) ** 2))
            
            # Direction accuracy
            if len(pred) > 1:
                pred_dir = np.diff(pred) > 0
                actual_dir = np.diff(actual) > 0
                result.direction_accuracy = np.mean(pred_dir == actual_dir) * 100
        
        # Simple ROI calculation
        if len(point_forecast[0]) > 0:
            first_pred = point_forecast[0][0]
            last_context = context[-1]
            if first_pred > last_context:
                # Would have bought
                result.roi = ((test_prices[-1] - test_prices[0]) / test_prices[0]) * 100
            else:
                result.roi = 0
        
        result.buy_hold_roi = ((test_prices[-1] - test_prices[0]) / test_prices[0]) * 100
        
    except Exception as e:
        print(f"   ⚠️ TimesFM error: {e}")
        return None
    
    return result


def run_comprehensive_benchmark(
    symbol: str = None,
    feature_sets_to_test: List[str] = None,
    save_results: bool = True
) -> List[Dict]:
    """
    Run comprehensive benchmark across models and feature sets.
    """
    print("=" * 70)
    print("🚀 ZEUS TRADER COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"⚡ Device: {get_device_info()}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch and prepare data
    print("\n📡 Fetching and Processing Data...")
    df = fetch_data(symbol)
    df = add_all_indicators(df)
    
    if feature_sets_to_test is None:
        feature_sets_to_test = ["basic", "momentum", "trend", "volatility", 
                                "volume_based", "full_technical", "full_macro", "kitchen_sink"]
    
    # Parameter configurations to test
    param_configs = [
        {"units": 128, "layers": 2, "epochs": 15, "threshold": 0.005, "name": "Small-Fast"},
        {"units": 256, "layers": 3, "epochs": 20, "threshold": 0.005, "name": "Medium"},
        {"units": 512, "layers": 4, "epochs": 25, "threshold": 0.005, "name": "Large"},
        {"units": 256, "layers": 3, "epochs": 20, "threshold": 0.003, "name": "Aggressive"},
        # New Transformer Configs
        {"units": 128, "layers": 2, "epochs": 20, "threshold": 0.005, "name": "Transformer-Base", "model_type": "Transformer"},
        {"units": 256, "layers": 4, "epochs": 25, "threshold": 0.005, "name": "Transformer-Large", "model_type": "Transformer"},
    ]
    
    results = []
    total_tests = len(feature_sets_to_test) * len(param_configs)
    test_count = 0
    
    print(f"\n🧪 Running {total_tests} benchmark configurations...\n")
    
    for feature_set_name in feature_sets_to_test:
        feature_cols = FEATURE_SETS.get(feature_set_name, FEATURE_SETS["basic"])
        
        for params in param_configs:
            test_count += 1
            config_name = params.get('name', 'Custom')
            print(f"[{test_count}/{total_tests}] {feature_set_name} + {config_name}...", end=" ")
            
            try:
                result = benchmark_lstm(df, feature_set_name, feature_cols, params)
                results.append(result.to_dict())
                
                alpha = result.roi - result.buy_hold_roi
                alpha_symbol = "📈" if alpha > 0 else "📉"
                print(f"ROI: {result.roi:+.1f}% | Alpha: {alpha:+.1f}% {alpha_symbol}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
    
    # Try TimesFM if available
    print("\n🔮 Checking TimesFM availability...")
    timesfm_result = benchmark_timesfm(df, "univariate")
    if timesfm_result:
        results.append(timesfm_result.to_dict())
        print(f"   TimesFM: Direction Acc: {timesfm_result.direction_accuracy:.1f}%")
    else:
        print("   TimesFM not installed. Install with: pip install timesfm")
    
    # Sort by alpha (ROI - Buy&Hold)
    results.sort(key=lambda x: x.get('alpha', 0), reverse=True)
    
    # Print top results
    print("\n" + "=" * 70)
    print("🏆 TOP 10 CONFIGURATIONS BY ALPHA (Excess Returns)")
    print("=" * 70)
    
    for i, r in enumerate(results[:10]):
        alpha = r.get('alpha', 0)
        print(f"\n#{i+1}: {r['model']} + {r['features']}")
        print(f"    ROI: {r['roi']:+.2f}% | Buy&Hold: {r['buy_hold']:+.2f}% | Alpha: {alpha:+.2f}%")
        print(f"    Trades: {r['trades']} | Win Rate: {r['win_rate']:.1f}%")
        if 'direction_accuracy' in r:
            print(f"    Direction Accuracy: {r['direction_accuracy']:.1f}%")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "symbol": symbol or CONFIG["default_target"],
                "total_configs": len(results),
                "results": results
            }, f, indent=2)
        print(f"\n✅ Results saved to {filename}")
    
    return results


def generate_benchmark_report(results: List[Dict]) -> str:
    """Generate markdown report from benchmark results."""
    
    report = """# 🏆 Zeus Trader Benchmark Report

## Executive Summary

"""
    
    if not results:
        return report + "No results to display."
    
    # Find best configuration
    best = results[0]
    
    report += f"""### 🥇 Best Configuration
- **Model:** {best['model']}
- **Feature Set:** {best['features']}
- **ROI:** {best['roi']:+.2f}%
- **Alpha (vs Buy&Hold):** {best.get('alpha', 0):+.2f}%
- **Win Rate:** {best['win_rate']:.1f}%
- **Trades:** {best['trades']}

---

## All Results (Sorted by Alpha)

| Rank | Model | Features | ROI | Alpha | Win Rate | Trades |
|------|-------|----------|-----|-------|----------|--------|
"""
    
    for i, r in enumerate(results[:20]):
        report += f"| {i+1} | {r['model']} | {r['features']} | {r['roi']:+.1f}% | {r.get('alpha', 0):+.1f}% | {r['win_rate']:.0f}% | {r['trades']} |\n"
    
    report += """

---

## Insights

### Feature Set Performance
"""
    
    # Group by feature set
    feature_performance = {}
    for r in results:
        fs = r['features']
        if fs not in feature_performance:
            feature_performance[fs] = []
        feature_performance[fs].append(r.get('alpha', 0))
    
    report += "\n| Feature Set | Avg Alpha | Best Alpha |\n|-------------|-----------|------------|\n"
    for fs, alphas in sorted(feature_performance.items(), key=lambda x: np.mean(x[1]), reverse=True):
        report += f"| {fs} | {np.mean(alphas):+.2f}% | {max(alphas):+.2f}% |\n"
    
    return report


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Generate report
    report = generate_benchmark_report(results)
    
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    print("\n📊 Report saved to benchmark_report.md")
