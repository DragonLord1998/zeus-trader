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
    "lookback": 60,   
    "test_days": 200, 
}

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 ZEUS TRADER: Running on {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# --- 1. DATA MODULE ---
def fetch_data():
    print("📡 Fetching Market Data...")
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    df = yf.download(CONFIG["target"], start=start_date, progress=False)
    if df.empty:
        print("❌ Error: No data for target.")
        sys.exit(1)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    macro_df = pd.DataFrame(index=df.index)
    for name, ticker in CONFIG["macro_assets"].items():
        try:
            m_data = yf.download(ticker, start=start_date, progress=False)
            if isinstance(m_data.columns, pd.MultiIndex):
                m_data.columns = m_data.columns.get_level_values(0)
            m_data = m_data['Close'].reindex(df.index).ffill()
            macro_df[name] = m_data
        except Exception as e:
            macro_df[name] = 0

    return pd.concat([df, macro_df], axis=1)

# --- 2. FEATURES ---
def add_features(df):
    print("⚙️  Calculating Indicators...")
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACDh_12_26_9']
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df.dropna(inplace=True)
    return df

# --- 3. MODEL ---
class ZeusLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(ZeusLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 4. TRAIN & BACKTEST ---
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)

def train_and_evaluate(df, config_override={}):
    hidden_dim = config_override.get('units', 128)
    layers = config_override.get('layers', 2)
    epochs = config_override.get('epochs', 20)
    batch_size = config_override.get('batch_size', 64)
    threshold = config_override.get('threshold', 0.005)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = [c for c in CONFIG['features'] if c in df.columns]
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    split_idx = len(data_scaled) - CONFIG["test_days"]
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx - CONFIG['lookback']:]
    
    X_train, y_train = create_sequences(train_data, CONFIG['lookback'])
    X_test, y_test = create_sequences(test_data, CONFIG['lookback'])
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    model = ZeusLSTM(len(feature_cols), hidden_dim, layers, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
            
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_t).cpu().numpy().flatten()
        
    real_close_prices = df['Close'].iloc[split_idx:].values
    balance = 100000
    shares = 0
    trades = 0
    wins = 0
    trade_log = []
    
    for i in range(len(predictions_scaled)):
        if i == 0: continue
        
        last_close_scaled = X_test[i][-1][0]
        pred_scaled = predictions_scaled[i]
        current_price_real = real_close_prices[i]
        
        action = "HOLD"
        
        if pred_scaled > last_close_scaled * (1 + threshold):
            if balance > current_price_real:
                can_buy = balance // current_price_real
                balance -= can_buy * current_price_real
                shares += can_buy
                action = "BUY"
                trade_log.append(f"🟢 BUY  @ {current_price_real:.2f} (Pred: UP)")
                
        elif pred_scaled < last_close_scaled * (1 - threshold):
            if shares > 0:
                balance += shares * current_price_real
                action = "SELL"
                trade_log.append(f"🔴 SELL @ {current_price_real:.2f} (Pred: DOWN)")
                shares = 0
                
        if i > 0:
            actual_move = real_close_prices[i] > real_close_prices[i-1]
            pred_move = pred_scaled > last_close_scaled
            if action != "HOLD" and (actual_move == pred_move):
                wins += 1
            if action != "HOLD":
                trades += 1
                
    final_value = balance + (shares * real_close_prices[-1])
    roi = ((final_value - 100000) / 100000) * 100
    
    return {
        "roi": roi,
        "trades": trades,
        "win_rate": (wins/trades*100) if trades > 0 else 0,
        "logs": trade_log
    }

if __name__ == "__main__":
    df = fetch_data()
    df = add_features(df)
    print("\n🔍 STARTING OPTIMIZATION...")
    
    param_grid = [
        {"units": 512, "layers": 4, "threshold": 0.005},
        {"units": 256, "layers": 4, "threshold": 0.005},
        {"units": 512, "layers": 3, "threshold": 0.005},
        {"units": 512, "layers": 5, "threshold": 0.005},
        {"units": 512, "layers": 4, "threshold": 0.004},
        {"units": 512, "layers": 4, "threshold": 0.006},
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

    results.sort(key=lambda x: x['roi'], reverse=True)
    print("\n🏆 BEST CONFIGURATION:")
    best = results[0]
    print(best)
    print("\n📜 Trade Log:")
    for log in best['logs']:
        print(log)