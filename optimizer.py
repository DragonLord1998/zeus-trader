"""
Zeus Trader Optimizer
=====================
Grid search for hyperparameter optimization.
"""

import json
from datetime import datetime
from typing import List, Dict, Any

from config import CONFIG
from data_fetcher import fetch_data
from feature_engine import add_indicators, normalize_data
from model import ZeusLSTM, create_sequences, train_model, device, get_device_info
from backtest import run_backtest


def run_grid_search(
    param_grid: List[Dict] = None,
    save_results: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run grid search over hyperparameter combinations.
    
    Args:
        param_grid: List of parameter dicts with keys: units, layers, threshold
        save_results: Save results to JSON file
        verbose: Print progress
    
    Returns:
        List of results sorted by ROI
    """
    if param_grid is None:
        param_grid = CONFIG["param_grid"]
    
    print(f"🚀 ZEUS OPTIMIZER: Starting Grid Search")
    print(f"   Device: {get_device_info()}")
    print(f"   Configurations to test: {len(param_grid)}")
    
    # Fetch and prepare data once
    print("\n📡 Preparing Data...")
    df = fetch_data()
    df = add_indicators(df)
    
    # Get feature columns
    feature_cols = [c for c in CONFIG["features"] if c in df.columns]
    print(f"   Features: {feature_cols}")
    
    # Normalize
    scaled_data, scaler, _ = normalize_data(df, feature_cols)
    
    # Split train/test
    test_days = CONFIG["backtest"]["test_days"]
    lookback = CONFIG["model"]["lookback"]
    
    split_idx = len(scaled_data) - test_days
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - lookback:]  # Include lookback for continuity
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    # Get real prices for backtest
    real_prices = df['Close'].iloc[split_idx:].values
    
    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Grid search
    results = []
    
    for i, params in enumerate(param_grid):
        units = params.get('units', CONFIG["model"]["lstm_units"])
        layers = params.get('layers', CONFIG["model"]["lstm_layers"])
        threshold = params.get('threshold', CONFIG["backtest"]["threshold"])
        epochs = params.get('epochs', CONFIG["model"]["epochs"])
        
        print(f"\n🧪 [{i+1}/{len(param_grid)}] Testing: Units={units}, Layers={layers}, Threshold={threshold*100:.1f}%")
        
        try:
            # Create model
            model = ZeusLSTM(
                input_dim=len(feature_cols),
                hidden_dim=units,
                num_layers=layers,
                dropout=CONFIG["model"]["dropout"]
            ).to(device)
            
            # Train
            train_model(model, X_train, y_train, epochs=epochs, verbose=False)
            
            # Backtest
            result = run_backtest(
                model, X_test, y_test, real_prices, scaler,
                threshold=threshold, verbose=False
            )
            
            result.update({
                "units": units,
                "layers": layers,
                "threshold": threshold,
                "epochs": epochs,
            })
            
            results.append(result)
            
            if verbose:
                print(f"   👉 ROI: {result['roi']:+.2f}% | Trades: {result['trades']} | Win Rate: {result['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Sort by ROI
    results.sort(key=lambda x: x["roi"], reverse=True)
    
    # Print top results
    print("\n" + "=" * 50)
    print("🏆 TOP CONFIGURATIONS:")
    print("=" * 50)
    
    for i, r in enumerate(results[:5]):
        print(f"\n#{i+1}: ROI {r['roi']:+.2f}% | Win Rate {r['win_rate']:.1f}%")
        print(f"    Units: {r['units']} | Layers: {r['layers']} | Threshold: {r['threshold']*100:.1f}%")
        print(f"    Trades: {r['trades']} | Buy&Hold: {r['buy_hold_roi']:+.2f}%")
    
    # Save results
    if save_results:
        filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            # Remove trade logs for cleaner JSON
            clean_results = [{k: v for k, v in r.items() if k != 'logs'} for r in results]
            json.dump(clean_results, f, indent=2)
        print(f"\n✅ Results saved to {filename}")
    
    return results


if __name__ == "__main__":
    run_grid_search()
