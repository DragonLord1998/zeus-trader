"""
Zeus Trader Ultimate Portfolio Scanner
======================================
The "God Mode" scanner that integrates every possible data source:
1. Momentum (LSTM)
2. Trend (SMA Alignment)
3. Valuation (P/E)
4. Growth (PEG)
5. Sentiment (News)
6. Smart Money (Institutional Holdings)
7. Analyst Ratings (Target Upside)
"""

import pandas as pd
import numpy as np
import torch
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

from config import CONFIG
from model import ZeusLSTM, create_sequences, train_model, predict, device
from sentiment import get_stock_sentiment

# Set max fetch limit to avoid timeouts
MAX_STOCKS_TO_SCAN = 20  # User can increase this


def parse_nifty_universe(csv_path: str) -> List[str]:
    """Parse Nifty universe CSV."""
    print(f"📂 Parsing universe from: {csv_path}")
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    symbols = df[0].tolist()
    valid_symbols = []
    for sym in symbols:
        sym = str(sym).strip().replace('"', '').replace('\n', '')
        if sym and not sym.startswith('NIFTY') and (sym.isalpha() or '&' in sym):
            valid_symbols.append(f"{sym}.NS")
    print(f"   ✅ Found {len(valid_symbols)} stocks in universe")
    return valid_symbols


def get_deep_metrics(symbol: str) -> Dict:
    """
    Fetch deep fundamental and institutional metrics.
    Slow operation, so only call for promising candidates.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 1. Valuation
        pe = info.get('trailingPE', 0)
        pb = info.get('priceToBook', 0)
        
        # 2. Growth
        peg = info.get('pegRatio', 0)
        
        # 3. Smart Money
        inst_holding = info.get('heldPercentInstitutions', 0)
        if inst_holding:
            inst_holding = round(inst_holding * 100, 2)
        else:
            inst_holding = 0
            
        # 4. Analyst
        target_price = info.get('targetMeanPrice', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        upside = 0
        if target_price and current_price:
            upside = round(((target_price - current_price) / current_price) * 100, 1)
            
        rec = info.get('recommendationKey', 'none')
        
        return {
            'pe': round(pe, 1) if pe else 0,
            'pb': round(pb, 1) if pb else 0,
            'peg': round(peg, 2) if peg else 0,
            'inst_holding': inst_holding,
            'target_upside': upside,
            'recommendation': rec
        }
    except Exception as e:
        # print(f"   ⚠️ Fundamentals failed for {symbol}: {e}")
        return {
            'pe': 0, 'pb': 0, 'peg': 0, 'inst_holding': 0, 
            'target_upside': 0, 'recommendation': 'none'
        }



    """
    Fast initial scan using Pre-Trained Transformer OR Fallback LSTM.
    """
    try:
        from datetime import datetime, timedelta
        import os
        from model import ZeusTransformer # Import Transformer
        
        # Fetch 1000 days for stability
        start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start_date, progress=False)
        
        if df.empty or len(df) < 300: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        # Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd_diff()
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df = df.dropna()
        if len(df) < lookback + 50: return None

        # --- PREDICT ---
        predicted_change = 0
        model_path = '/Users/philipmathewkavalam/Desktop/Zeus Trader/zeus_core/zeus_best_model.pth' # Local Path
        
        # 1. Check for Pre-Trained Brain 🧠
        if os.path.exists(model_path):
            # Use TRANSFOMER (128, 2, 8) - Winner Config
            # Must replicate Cloud Feature Engineering EXACTLY
            df_inf = df.copy()
            df_inf['ret'] = np.log(df_inf['Close'] / df_inf['Close'].shift(1))
            df_inf['macd_norm'] = (df_inf['MACD'] - df_inf['MACD'].rolling(200).mean()) / (df_inf['MACD'].rolling(200).std() + 1e-6)
            df_inf['vol_roll'] = df_inf['Close'].pct_change().rolling(20).std()
            df_inf['rsi_norm'] = df_inf['RSI'] / 100.0
            
            # Features: ['ret', 'rsi', 'macd', 'vol'] matched to Cloud
            feat_cols = ['ret', 'rsi_norm', 'macd_norm', 'vol_roll']
            feats = df_inf[feat_cols].fillna(0).values
            
            # Normalize (Instance Norm for inference)
            feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1e-6)
            
            X_seq = feats[-lookback:] # Last 60 days
            if len(X_seq) == lookback:
                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(device)
                
                model = ZeusTransformer(input_dim=4, d_model=128, nhead=8, num_layers=2, dropout=0.1).to(device)
                
                # Load weights (relaxed loading)
                state = torch.load(model_path, map_location=device)
                new_state = {}
                for k, v in state.items():
                    # Map 'transformer.layers' -> 'transformer_encoder.layers' if needed
                    # Our ZeusTransformer uses 'transformer_encoder' but Cloud uses 'transformer'
                    if 'transformer.layers' in k:
                        new_key = k.replace('transformer.layers', 'transformer_encoder.layers')
                    else:
                        new_key = k
                    new_state[new_key] = v
                    
                try:
                    model.load_state_dict(new_state, strict=False)
                    model.eval()
                    with torch.no_grad():
                        pred_log_ret = model(X_tensor).item()
                    predicted_change = (np.exp(pred_log_ret) - 1) * 100
                except:
                    # Fallback if load fails
                    pass

        # 2. Fallback to LSTM if no model or load failed
        if predicted_change == 0:
            feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_50']
            available_cols = [c for c in feature_cols if c in df.columns]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[available_cols].values)
            
            train_data = scaled_data[:-5]
            test_data = scaled_data[-lookback-5:]
            
            X_train, y_train = create_sequences(train_data, lookback)
            X_test, y_test = create_sequences(test_data, lookback)
            
            if len(X_train) > 50:
                model = ZeusLSTM(len(available_cols), 128, 2, dropout=0.2).to(device)
                train_model(model, X_train, y_train, epochs=5, verbose=False) # Fast train
                last_pred = predict(model, X_test)[-1]
                last_actual = X_test[-1][-1][0]
                predicted_change = ((last_pred - last_actual) / last_actual) * 100 if last_actual > 0 else 0
        
        # Trend check
        close = df['Close'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        sma200 = df['SMA_200'].iloc[-1]
        trend = "UP" if (close > sma50 > sma200) else "DOWN" if (close < sma50 < sma200) else "SIDEWAYS"
        
        return {
            'symbol': symbol,
            'price': round(float(close), 2),
            'signal_pct': round(float(predicted_change), 2),
            'trend': trend,
            'rsi': round(float(df['RSI'].iloc[-1]), 1)
        }
        
    except Exception as e:
        # print(f"   ❌ Error: {e}")
        return None


def calculate_composite_score(metrics: Dict) -> float:
    """
    Calculate the Zeus Score (0-100) based on all factors.
    """
    score = 0
    
    # 1. Momentum (LSTM) - Max 30 pts
    sig = metrics.get('signal_pct', 0)
    if sig > 1: score += 30
    elif sig > 0: score += 15
    elif sig < -1: score -= 30
    
    # 2. Trend - Max 10 pts
    if metrics.get('trend') == 'UP': score += 10
    
    # 3. Analyst Upside - Max 20 pts
    upside = metrics.get('target_upside', 0)
    if upside > 20: score += 20
    elif upside > 10: score += 10
    
    # 4. Valuation - Max 10 pts
    pe = metrics.get('pe', 0)
    if 0 < pe < 25: score += 10  # Reasonable value
    
    # 5. Smart Money - Max 10 pts
    inst = metrics.get('inst_holding', 0)
    if inst > 30: score += 10
    elif inst > 10: score += 5
    
    # 6. Sentiment - Max 20 pts
    sent = metrics.get('sentiment_score', 0)
    if sent > 1: score += 20
    elif sent > 0: score += 10
    elif sent < -1: score -= 20
    
    return round(max(0, min(100, score)), 1)


def run_ultimate_scan(csv_path: str, limit: int = 15):
    """Run the scanner."""
    universe = parse_nifty_universe(csv_path)
    print("\n" + "=" * 60)
    print(f"🚀 ZEUS ULTIMATE SCANNER (Data Sources: 7)")
    print("=" * 60)
    print(f"Scanning first {min(len(universe), limit)} stocks...")
    
    final_results = []
    
    for i, sym in enumerate(universe[:limit]):
        print(f"\n[{i+1}/{limit}] Scanning {sym}...", end=" ")
        
        # 1. Fast Scan
        initial = scan_stock_initial(sym)
        
        if not initial:
            print("❌ No Data/Error")
            continue
            
        print(f"LSTM: {initial['signal_pct']:+.1f}% | Trend: {initial['trend']}", end=" ")
        
        # 2. Deep Metrics (Only if positive signal OR purely random test)
        # For now, scan all to show user the power
        
        print("➡️  Deep Scan...", end=" ")
        
        # Fetch Fundamentals
        fund = get_deep_metrics(sym)
        
        # Fetch Sentiment
        from sentiment import get_stock_sentiment
        sent = get_stock_sentiment(sym)
        
        # Combine
        full_metrics = {
            **initial,
            **fund,
            'sentiment_score': sent['sentiment'],
            'news_count': sent['news_count']
        }
        
        # Score
        score = calculate_composite_score(full_metrics)
        full_metrics['zeus_score'] = score
        
        final_results.append(full_metrics)
        print(f"✅ Zeus Score: {score}/100")

    # Sort and Report
    final_results.sort(key=lambda x: x['zeus_score'], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"{'SYMBOL':<15} {'PRICE':<10} {'SCORE':<6} {'LSTM':<8} {'TREND':<10} {'SENTIMENT':<10} {'UPSIDE':<8} {'INST%':<6}")
    print("-" * 80)
    
    for r in final_results:
        print(f"{r['symbol'].replace('.NS',''):<15} ₹{r['price']:<9} {r['zeus_score']:<6} {r['signal_pct']:+.1f}%   {r['trend']:<10} {r['sentiment_score']:+.1f}       {r['target_upside']:+.0f}%     {r['inst_holding']}%")
        
    print("=" * 80)
    
    # Save
    with open('ultimate_scan_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    print("✅ Saved to ultimate_scan_results.json")


if __name__ == "__main__":
    csv_path = '/Users/philipmathewkavalam/Desktop/Zeus Trader/Market Data/MW-NIFTY-200-17-Jan-2026.csv'
    run_ultimate_scan(csv_path, limit=15)
