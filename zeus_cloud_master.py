"""
ZEUS TRADER: CLOUD MASTER (NIFTY 200 EDITION)
=============================================
This script is designed for High-Performance Computing (HPC) environments (RunPod/Lambda).
It trains a "Global Market Model" using data from ALL Nifty 200 stocks.

Architecture:
1. Data Ingestion: Parallel download of Nifty 200 universe.
2. Preprocessing: Stock-specific Z-Score normalization (critical for global training).
3. Model: Deep ZeusTransformer (12 Layers, 8 Heads, 512 Embed).
4. Training: Mixed Precision (AMP), Large Batch Size, AdamW.

Usage:
    python3 zeus_cloud_master.py [--epochs 50] [--batch 1024]
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
import warnings
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

# Suppress junk
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONFIG = {
    "lookback": 60,
    "forecast": 1,
    "features": ["pro_return", "rsi", "macd", "volatility", "volume_change"],
    "model": {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,  # Deep model for cloud
        "dropout": 0.1
    },
    "train": {
        "epochs": 100,
        "batch_size": 2048,  # Huge batch for GPU
        "lr": 1e-4,
        "patience": 10
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 ZEUS CLOUD MASTER initializing on {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- DATA PIPELINE ---

def get_nifty_200_symbols():
    # Fallback to hardcoded top 200 if CSV parsing fails in cloud
    # In production, we upload the CSV. Here we assume generic list or parse provided CSV.
    # For robust cloud start, we'll try to read CSV, else fetch standard list.
    csv_path = "Market Data/MW-NIFTY-200-17-Jan-2026.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, skiprows=1)
            symbols = [str(x).strip().replace('"','') + ".NS" for x in df[0].tolist()]
            return [s for s in symbols if "NIFTY" not in s and "&" not in s]
        except:
            pass
    
    # Fallback list (truncated for script size, ideally dynamic)
    print("⚠️ CSV not found, fetching Nifty 50 tickers from web...")
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[1]
        return [f"{s}.NS" for s in payload.Symbol.tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"] # Minimal fallback

def fetch_and_process_stock(symbol):
    try:
        # Fetch 5 years of data for deep learning
        df = yf.download(symbol, period="5y", progress=False)
        if len(df) < 500: return None
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # --- FEATURE ENGINEERING (Vectorized) ---
        # 1. Log Returns (Stationary)
        df['Close'] = df['Close'].replace(0, method='ffill')
        df['pro_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0 # Scale 0-1
        
        # 3. MACD Normalised
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff() 
        # Robust scale MACD
        df['macd'] = (df['macd'] - df['macd'].rolling(200).mean()) / (df['macd'].rolling(200).std() + 1e-6)
        
        # 4. Volatility (ATR-like)
        df['volatility'] = df['Close'].pct_change().rolling(20).std()
        
        # 5. Volume Change
        df['volume_change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0)
        df['volume_change'] = np.clip(df['volume_change'], -1, 1) # Clip outliers
        
        df = df.dropna()
        
        # Final cleanup
        features = df[CONFIG['features']].values
        
        # Z-Score Normalization (Global vs Local? Local is safer for multi-stock)
        # Normalize PER STOCK to mean 0, std 1
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0) + 1e-6
        features = (features - means) / stds
        
        # Create sequences
        X, y = [], []
        lookback = CONFIG['lookback']
        # Target: Next day return (Classify or Regress?) 
        # We will Regress on next day Return
        targets = df['pro_return'].shift(-1).dropna().values
        features = features[:-1] # Align
        
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(targets[i+lookback])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    except Exception as e:
        return None

def build_global_dataset(max_workers=8):
    symbols = get_nifty_200_symbols()
    print(f"🌍 Building Global Dataset from {len(symbols)} stocks...")
    
    all_X = []
    all_y = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_and_process_stock, symbols))
        
    for res in results:
        if res:
            X, y = res
            all_X.append(X)
            all_y.append(y)
            
    if not all_X:
        raise ValueError("No data collected!")
        
    # Stack 'em high
    Global_X = np.concatenate(all_X)
    Global_y = np.concatenate(all_y)
    
    print(f"📚 Dataset Compiled: {Global_X.shape[0]} samples")
    print(f"   Shape: {Global_X.shape}")
    return Global_X, Global_y

# --- MODEL ARCHITECTURE ---

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

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ZeusGlobalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :]) # Predict from last token

# --- TRAINING LOOP ---

def train_global_model():
    # 1. Data
    X, y = build_global_dataset()
    
    # Train/Val Split (Time based? No, Random is okay for Global model physics learning, 
    # but Time-based holdout is stricter. We'll use Random 90/10 for now given diverse tickers)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['train']['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['train']['batch_size'], shuffle=False)
    
    # 2. Model
    model = ZeusGlobalTransformer(
        input_dim=X.shape[2],
        d_model=CONFIG['model']['d_model'],
        nhead=CONFIG['model']['nhead'],
        num_layers=CONFIG['model']['num_layers'],
        dropout=CONFIG['model']['dropout']
    ).to(DEVICE)
    
    # 3. Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['train']['lr'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision Support
    
    print(f"🚀 Starting Training (Epochs: {CONFIG['train']['epochs']})")
    
    best_loss = float('inf')
    
    for epoch in range(CONFIG['train']['epochs']):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.cuda.amp.autocast():
                preds = model(batch_X).squeeze()
                loss = criterion(preds, batch_y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_X).squeeze()
                val_loss += criterion(preds, batch_y).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"   Epoch {epoch+1}: Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "zeus_global_model.pth")
            print("   💾 Best Model Saved")

if __name__ == "__main__":
    train_global_model()
