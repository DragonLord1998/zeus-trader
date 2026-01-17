import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import sys
import logging

# --- CONFIGURATION ---
CONFIG = {
    "target": "RELIANCE.NS",
    "macro_assets": {
        "NIFTY": "^NSEI",
        "OIL": "CL=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "COPPER": "HG=F",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB"
    },
    "features": ["Close", "RSI", "MACD", "SMA_50", "Volume", "NIFTY", "OIL", "GOLD", "US10Y", "DXY"],
    "lookback": 60,   # Days to look back
    "forecast_horizon": 1,
    "test_days": 200, # Number of days to backtest
}

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 ZEUS TRADER: Running on {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- 1. DATA MODULE ---
def fetch_data():
    print("📡 Fetching Market Data...")
    
    # Calculate start date (3 years ago)
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    # 1. Fetch Target
    df = yf.download(CONFIG["target"], start=start_date, progress=False)
    if df.empty:
        print("❌ Error: No data for target.")
        sys.exit(1)
    
    # Flatten MultiIndex columns if present (yfinance update behavior)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Fetch Macro
    macro_df = pd.DataFrame(index=df.index)
    
    for name, ticker in CONFIG["macro_assets"].items():
        try:
            m_data = yf.download(ticker, start=start_date, progress=False)
            if isinstance(m_data.columns, pd.MultiIndex):
                m_data.columns = m_data.columns.get_level_values(0)
            
            # Reindex to match target trading days (Forward Fill missing data like US holidays)
            m_data = m_data['Close'].reindex(df.index).ffill()
            macro_df[name] = m_data
        except Exception as e:
            print(f"⚠️ Failed to fetch {name}: {e}")
            macro_df[name] = 0

    # Merge
    full_df = pd.concat([df, macro_df], axis=1)
    return full_df

# --- 2. FEATURE ENGINEERING ---
def add_features(df):
    print("⚙️  Calculating Technical Indicators...")
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACDh_12_26_9'] # Histogram
    
    # SMA
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    
    # Clean NaN (indicators need warmup)
    df.dropna(inplace=True)
    return df

# --- 3. MODEL DEFINITION (PyTorch LSTM) ---
class ZeusLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(ZeusLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# --- 4. TRAINING & OPTIMIZER ---
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0]) # Predict 0th column (Close Price)
    return np.array(X), np.array(y)

def train_and_evaluate(df, config_override={}):
    # Settings
    hidden_dim = config_override.get('units', 128)
    layers = config_override.get('layers', 2)
    epochs = config_override.get('epochs', 20)
    batch_size = config_override.get('batch_size', 64)
    lr = config_override.get('lr', 0.001)
    threshold = config_override.get('threshold', 0.005)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Select features
    feature_cols = [c for c in CONFIG['features'] if c in df.columns]
    
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    # Train/Test Split
    split_idx = len(data_scaled) - CONFIG["test_days"]
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx - CONFIG['lookback']:] # Need lookback buffer
    
    # Create Sequences
    X_train, y_train = create_sequences(train_data, CONFIG['lookback'])
    X_test, y_test = create_sequences(test_data, CONFIG['lookback'])
    
    # To Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # Model Setup
    model = ZeusLSTM(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=layers, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train Loop
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
    # Inference (Backtest)
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_t).cpu().numpy().flatten()
        
    # BACKTEST SIMULATION
    # We need to Inverse Scale the predictions to get real prices?
    # Actually, simpler: Compare Scaled Pred vs Scaled Actual for direction.
    # But for PnL we need Real Prices.
    
    # Get Real "Close" prices for the test period
    real_close_prices = df['Close'].iloc[split_idx:].values
    
    balance = 100000
    shares = 0
    trades = 0
    wins = 0
    
        # Simulation
    
        trade_log = []
    
        
    
        for i in range(len(predictions_scaled)):
    
            if i == 0: continue
    
            
    
            last_close_scaled = X_test[i][-1][0]
    
            pred_scaled = predictions_scaled[i]
    
            current_price_real = real_close_prices[i]
    
            date_idx = split_idx + i 
    
            # We assume df index matches. (Approximate for logging)
    
            
    
            action = "HOLD"
    
            
    
            # Decide
    
            if pred_scaled > last_close_scaled * (1 + threshold):
    
                # Buy
    
                if balance > current_price_real:
    
                    can_buy = balance // current_price_real
    
                    balance -= can_buy * current_price_real
    
                    shares += can_buy
    
                    action = "BUY"
    
                    trade_log.append(f"🟢 BUY  @ {current_price_real:.2f} (Pred: UP)")
    
                    
    
            elif pred_scaled < last_close_scaled * (1 - threshold):
    
                # Sell
    
                if shares > 0:
    
                    balance += shares * current_price_real
    
                    # Calc profit for this trade
    
                    # Simplified: we don't track specific lots here, just total balance change
    
                    action = "SELL"
    
                    trade_log.append(f"🔴 SELL @ {current_price_real:.2f} (Pred: DOWN)")
    
                    shares = 0
    
                    
    
            # Win Rate Check (Direction)
    
            if i > 0:
    
                actual_move = real_close_prices[i] > real_close_prices[i-1]
    
                pred_move = pred_scaled > last_close_scaled
    
                if action != "HOLD" and (actual_move == pred_move):
    
                    wins += 1
    
                if action != "HOLD":
    
                    trades += 1
    
                    
    
        # Final Value
    
        final_value = balance + (shares * real_close_prices[-1])
    
        roi = ((final_value - 100000) / 100000) * 100
    
        
    
        return {
    
            "roi": roi,
    
            "trades": trades,
    
            "win_rate": (wins/trades*100) if trades > 0 else 0,
    
            "logs": trade_log
    
        }
    
    
    
    # --- MAIN EXECUTION ---
    
    if __name__ == "__main__":
    
        df = fetch_data()
    
        df = add_features(df)
    
        
    
        print("\n🔍 STARTING GRID SEARCH OPTIMIZATION...")
    
        
    
        # REFINED Grid Search Options based on previous success
    
        param_grid = [
    
            # The Baseline Winner
    
            {"units": 512, "layers": 4, "threshold": 0.005},
    
            
    
            # Variations around the winner
    
            {"units": 256, "layers": 4, "threshold": 0.005}, # Lighter model
    
            {"units": 512, "layers": 3, "threshold": 0.005}, # Shallower
    
            {"units": 512, "layers": 5, "threshold": 0.005}, # Deeper
    
            
    
            # Threshold Tuning
    
            {"units": 512, "layers": 4, "threshold": 0.004}, # More aggressive
    
            {"units": 512, "layers": 4, "threshold": 0.006}, # More conservative
    
            
    
            # The "Tank" v2 (Better threshold)
    
            {"units": 1024, "layers": 4, "threshold": 0.004} 
    
        ]
    
        
    
        results = []
    
        
    
        for params in param_grid:
    
            print(f"\n🧪 Testing Config: {params}")
    
            try:
    
                res = train_and_evaluate(df, params)
    
                print(f"   👉 ROI: {res['roi']:.2f}% | Trades: {res['trades']} | WR: {res['win_rate']:.1f}%")
    
                if res['trades'] > 0:
    
                    print(f"      Last 3 Trades: {res['logs'][-3:]}")
    
                results.append({**params, **res})
    
            except Exception as e:
    
                print(f"   ❌ Failed: {e}")
    
                # import traceback
    
                # traceback.print_exc()
    
    
    
        # Sort best
    
        results.sort(key=lambda x: x['roi'], reverse=True)
    
        print("\n🏆 BEST CONFIGURATION:")
    
        best = results[0]
    
        print(best)
    
        print("\n📜 Full Trade Log for Best Run:")
    
        for log in best['logs']:
    
            print(log)
    
    
