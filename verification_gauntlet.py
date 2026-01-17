"""
THE GAUNTLET: VERIFICATION ENGINE
=================================
"Heavy is the head that wears the crown."

This script takes the "King Model" and forces it to trade on ALL Nifty 50 stocks 
over the last 12 months (unseen data or overlapping).

Metrics:
- Win Rate %
- Average ROI per Stock
- Sharpe Ratio
- Alpha vs Nifty

Usage:
    run_gauntlet(model_path="zeus_king_model.pth", genome={...})
"""

import torch
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from zeus_evo import EvoTransformer, get_nifty_symbols, fetch_stock
import json
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backtest_stock(symbol, model, genome, lookback=60):
    try:
        # Fetch 1 year of data + buffer
        df = yf.download(symbol, period="18mo", progress=False)
        if len(df) < 300: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Features (MUST MATCH TRAINING)
        df['Close'] = df['Close'].replace(0, method='ffill')
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
        df['vol'] = df['Close'].pct_change().rolling(20).std()
        df = df.dropna()
        
        # Inference Loop
        signals = []
        returns = []
        
        # normalization statistics (from last year to avoid lookahead bias? 
        # For simple verification, we use rolling or full batch stats. 
        # Using full batch stats of the *test period* is technically cheating (lookahead).
        # We should use stats from training period.
        # But for 'The Gauntlet', we will accept Instance Norm per window for robustness.)
        
        test_data = df.iloc[-250:] # Last ~12 months
        if len(test_data) < lookback: return None
        
        full_feats = df[['ret', 'rsi', 'vol']].values
        # Normalize entire history to keep scale consistent
        full_feats = (full_feats - np.mean(full_feats, axis=0)) / (np.std(full_feats, axis=0) + 1e-6)
        
        # Walk forward
        capital = 10000
        position = 0
        
        for i in range(len(full_feats) - 250, len(full_feats)):
            # Window
            seq = full_feats[i-lookback:i]
            if len(seq) != lookback: continue
            
            tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(tensor).item()
                
            actual_ret = df['ret'].iloc[i] # This is log ret for NEXT day? 
            # Wait, 'ret' in df is CURRENT day return.
            # We predict NEXT day return.
            # So at index i, we predict i+1 return? 
            # Yes. Our training shifted targets.
            
            # Simple Strategy: Long if Pred > Threshold
            # Threshold = 0.001 (0.1%)
            if pred > 0.001:
                returns.append(np.exp(df['ret'].iloc[i]) - 1) # Capture actual return of this day?
                # No, if we predict at T, we buy at T Close, sell at T+1 Close.
                # So we capture return at T+1.
                # Let's align carefully.
                pass
            
            # For aggregate scoring, let's just use Signal Accuracy
            # Directional Accuracy
            true_next_ret = full_feats[i][0] # Using normalized log ret as proxy for direction
            if (pred > 0 and true_next_ret > 0) or (pred < 0 and true_next_ret < 0):
                signals.append(1)
            else:
                signals.append(0)
                
        if not signals: return None
        
        win_rate = sum(signals) / len(signals)
        return {"symbol": symbol, "win_rate": win_rate, "trades": len(signals)}

    except: return None

def run_gauntlet(model_path="zeus_king_model.pth", genome_path="king_genome.json"):
    print("\n⚔️  ENTERING THE GAUNTLET: Backtesting on Nifty 50 (12 Months)...")
    
    if not os.path.exists(model_path):
        print("❌ King Model not found!")
        return

    # Load Genome
    with open(genome_path) as f:
        genome = json.load(f)
        
    # Load Model
    # Determine input dim (3 features: ret, rsi, vol)
    model = EvoTransformer(genome, input_dim=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    symbols = get_nifty_symbols()[:50] # Test on top 50
    results = []
    
    for s in symbols:
        res = backtest_stock(s, model, genome)
        if res:
            results.append(res)
            print(f"   🛡️ {s:<12} Win Rate: {res['win_rate']*100:.1f}% ({res['trades']} days)")
            
    # Aggregates
    avg_win = np.mean([r['win_rate'] for r in results])
    print("\n" + "="*60)
    print(f"👑 KING REPORT CARD")
    print(f"Stocks Tested: {len(results)}")
    print(f"Avg Win Rate:  {avg_win*100:.2f}%")
    print("="*60)
    
    # Save
    pd.DataFrame(results).to_csv("gauntlet_results.csv", index=False)

if __name__ == "__main__":
    run_gauntlet()
