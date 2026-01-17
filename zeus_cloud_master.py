"""
ZEUS TRADER: CLOUD GRID SEARCH MASTER (NIFTY 200)
=================================================
HPC Script to find the "Winning Architecture" via exhaustive Grid Search.
Trains multiple Transformer configurations on the Global Nifty 200 dataset.

Features:
- Global Dataset Construction (200 stocks).
- Automated Grid Search (Architecture & Hyperparams).
- Mixed Precision Training.
- Best Model Checkpointing.

Usage:
    python3 zeus_cloud_master.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import itertools
import warnings
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 ZEUS CLOUD GRID SEARCH initializing on {DEVICE}")

# --- GRID SEARCH SPACE ---
# Define the universe of models to test
GRID_SEARCH_SPACE = {
    "d_model": [128, 256, 512],
    "num_layers": [2, 4, 8],
    "nhead": [4, 8],
    "dropout": [0.1, 0.2],
    "lr": [1e-4] 
}

# --- DATA PIPELINE (Same as before) ---
# ... (Reusing robust fetching logic)

def get_nifty_200_symbols():
    csv_path = "Market Data/MW-NIFTY-200-17-Jan-2026.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, skiprows=1)
            symbols = [str(x).strip().replace('"','') + ".NS" for x in df[0].tolist()]
            return [s for s in symbols if "NIFTY" not in s]
        except: pass
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

def fetch_and_process_stock(symbol):
    try:
        df = yf.download(symbol, period="5y", progress=False)
        if len(df) < 500: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Features
        df['Close'] = df['Close'].replace(0, method='ffill')
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff()
        df['macd'] = (df['macd'] - df['macd'].rolling(200).mean()) / (df['macd'].rolling(200).std() + 1e-6)
        df['vol'] = df['Close'].pct_change().rolling(20).std()
        df = df.dropna()
        
        # Select Features
        features = df[['ret', 'rsi', 'macd', 'vol']].values
        
        # Normalize
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0) + 1e-6
        features = (features - means) / stds
        
        # Sequences
        X, y = [], []
        lookback = 60
        targets = df['ret'].shift(-1).dropna().values
        features = features[:-1]
        
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(targets[i+lookback])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    except: return None

def build_global_dataset():
    symbols = get_nifty_200_symbols()
    print(f"🌍 Fetching data for {len(symbols)} stocks...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(fetch_and_process_stock, symbols))
    
    all_X, all_y = [], []
    for res in results:
        if res:
            all_X.append(res[0])
            all_y.append(res[1])
            
    if not all_X: raise ValueError("No data!")
    return np.concatenate(all_X), np.concatenate(all_y)

# --- MODEL --- 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class ZeusGlobalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :])

# --- GRID SEARCH ENGINE ---

def run_grid_search():
    # 1. Prepare Data
    X, y = build_global_dataset()
    input_dim = X.shape[2]
    print(f"📚 Global Dataset: {X.shape[0]} samples")
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    # 2. Generate Params
    keys, values = zip(*GRID_SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🧪 Starting Grid Search: {len(combinations)} candidates")
    
    best_loss = float('inf')
    best_config = None
    results_log = []
    
    for i, config in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {config}")
        
        model = ZeusGlobalTransformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        
        # Short training for grid search (10-20 epochs)
        epochs = 15 
        model_best_val = float('inf')
        
        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = criterion(model(bx).squeeze(), by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    val_loss += criterion(model(bx).squeeze(), by).item()
            val_loss /= len(val_loader)
            model_best_val = min(model_best_val, val_loss)
            
        print(f"   🏁 Result: Val Loss = {model_best_val:.6f}")
        
        results_log.append({**config, "val_loss": model_best_val})
        
        if model_best_val < best_loss:
            best_loss = model_best_val
            best_config = config
            print("   🌟 New Best Model! Saving...")
            torch.save(model.state_dict(), "zeus_best_model.pth")
            
    # Save Results
    with open("grid_search_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
        
    print("\n" + "="*60)
    print(f"🏆 GRID SEARCH COMPLETE")
    print(f"Best Loss: {best_loss}")
    print(f"Best Config: {best_config}")
    print("Model saved to zeus_best_model.pth")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
