import warnings
import torch
import numpy as np
import random

# Fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

warnings.filterwarnings('ignore')
from config import CONFIG
from data_fetcher import fetch_data
from feature_engine import add_all_indicators, FEATURE_SETS
from benchmark import benchmark_lstm

def run_duel():
    print("="*60)
    print("⚔️  ZEUS MODEL DUEL: LSTM vs TRANSFORMER")
    print("="*60)
    
    # 1. Fetch & Prep Data
    symbol = "RELIANCE.NS"
    print(f"📡 Fetching data for {symbol}...")
    df = fetch_data(symbol)
    df = add_all_indicators(df)
    
    feature_set = "momentum"
    cols = FEATURE_SETS[feature_set]
    
    # 2. Define Contenders
    # Using the best parameters found earlier for LSTM
    lstm_params = {
        "units": 512, "layers": 4, "epochs": 25, 
        "dropout": 0.2, "name": "LSTM-Champion",
        "threshold": 0.005
    }
    tf_params = {
        "units": 128, "layers": 2, "epochs": 25, 
        "dropout": 0.2, "name": "ZeusTransformer", 
        "model_type": "Transformer",
        "threshold": 0.005
    }
    
    # 3. Round 1: LSTM
    print("\n🥊 ROUND 1: LSTM (The Current Champion)")
    res_lstm = benchmark_lstm(df, feature_set, cols, lstm_params)
    alpha_lstm = res_lstm.roi - res_lstm.buy_hold_roi
    print(f"   📊 Results: ROI={res_lstm.roi}% | Alpha={alpha_lstm:.1f}% | WinRate={res_lstm.win_rate}%")
    
    # 4. Round 2: Transformer
    print("\n🥊 ROUND 2: ZeusTransformer (The Challenger)")
    res_tf = benchmark_lstm(df, feature_set, cols, tf_params)
    alpha_tf = res_tf.roi - res_tf.buy_hold_roi
    print(f"   📊 Results: ROI={res_tf.roi}% | Alpha={alpha_tf:.1f}% | WinRate={res_tf.win_rate}%")
    
    # 5. Verdict
    print("\n" + "="*60)
    if res_tf.roi > res_lstm.roi:
        print(f"🏆 WINNER: ZeusTransformer (+{res_tf.roi - res_lstm.roi:.1f}%)")
    else:
        print(f"🏆 WINNER: LSTM (+{res_lstm.roi - res_tf.roi:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    run_duel()
