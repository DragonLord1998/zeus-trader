import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import random
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
GENERATIONS = 15     # God Mode: 15 Generations
POPULATION_SIZE = 20 # God Mode: 20 Models
SURVIVORS = 5        # Top 5 survive
MUTATION_RATE = 0.2  # Stability

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧬 ZEUS EVO Initializing on {DEVICE}")

# --- GENOME DEFINITION ---
GENE_SPACE = {
    "d_model": [64, 128, 256, 512],
    "num_layers": [2, 3, 4, 6, 8],
    "nhead": [4, 8, 16],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-3, 5e-4, 1e-4, 5e-5],
    "lookback": [30, 60, 90]
}

def random_genome():
    return {k: random.choice(v) for k, v in GENE_SPACE.items()}

def mutate(genome):
    new_genome = genome.copy()
    gene_to_change = random.choice(list(GENE_SPACE.keys()))
    new_genome[gene_to_change] = random.choice(GENE_SPACE[gene_to_change])
    return new_genome

def crossover(parent1, parent2):
    child = {}
    for k in GENE_SPACE.keys():
        child[k] = parent1[k] if random.random() > 0.5 else parent2[k]
    return child

# --- DATA PIPELINE ---
def get_nifty_symbols():
    csv_path = "Market Data/MW-NIFTY-200-17-Jan-2026.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, skiprows=1)
            raw = [str(x).strip().replace('"','') for x in df[0].tolist()]
            return [f"{s}.NS" for s in raw if "NIFTY" not in s]
        except: pass
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')
        return [f"{s}.NS" for s in tables[1]['Symbol'].tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

def fetch_stock(symbol, lookback=60):
    try:
        df = yf.download(symbol, period="3y", progress=False)
        if len(df) < 300: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Close'] = df['Close'].replace(0, method='ffill')
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
        df['vol'] = df['Close'].pct_change().rolling(20).std()
        
        df = df.dropna()
        feats = df[['ret', 'rsi', 'vol']].values
        
        # Instance Norm
        feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1e-6)
        
        X, y = [], []
        target = df['ret'].shift(-1).dropna().values
        feats = feats[:-1]
        
        if len(feats) < lookback: return None
        
        for i in range(len(feats) - lookback):
            X.append(feats[i:i+lookback])
            y.append(target[i+lookback])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    except: return None

# Global Data Loading
LOOKBACK_FIXED = 60
print("🌍 Building Global Data (Lookback=60)...", flush=True)
symbols = get_nifty_symbols()
all_X, all_y = [], []
with ThreadPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(lambda s: fetch_stock(s, LOOKBACK_FIXED), symbols))

for res in results:
    if res:
        all_X.append(res[0])
        all_y.append(res[1])
        
GLOBAL_X = np.concatenate(all_X)
GLOBAL_Y = np.concatenate(all_y)
print(f"📚 Dataset: {GLOBAL_X.shape[0]} samples", flush=True)

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

class EvoTransformer(nn.Module):
    def __init__(self, gene, input_dim):
        super().__init__()
        d_model = gene['d_model']
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, gene['nhead'], dim_feedforward=d_model*4, 
                                         dropout=gene['dropout'], batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, gene['num_layers'])
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :])

# --- EVALUATION ---
def evaluate_genome(genome, X_train, y_train, X_val, y_val):
    try:
        model = EvoTransformer(genome, input_dim=X_train.shape[2]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=genome['lr'])
        criterion = nn.MSELoss()
        
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        # Large batch size for speed
        loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, persistent_workers=True)
        
        scaler = torch.cuda.amp.GradScaler()
        
        model.train()
        for epoch in range(5):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = criterion(model(bx).squeeze(), by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=4096)
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx).squeeze()
                val_loss += criterion(pred, by).item()
        
        score = val_loss / len(val_loader)
        return score, model.state_dict()
    except Exception as e:
        print(f"💀 Genome Died: {e}", flush=True)
        return float('inf'), None

def run_evolution():
    print(f"\n🧬 Starting Evolution: {GENERATIONS} Generations, {POPULATION_SIZE} Population", flush=True)
    
    population = [random_genome() for _ in range(POPULATION_SIZE)]
    
    split = int(0.9 * len(GLOBAL_X))
    X_train, X_val = GLOBAL_X[:split], GLOBAL_X[split:]
    y_train, y_val = GLOBAL_Y[:split], GLOBAL_Y[split:]
    
    best_ever_score = float('inf')
    best_ever_genome = None
    
    print("🐛 Debug: Entering Generation Loop...", flush=True)

    for gen in range(GENERATIONS):
        print(f"\n⏳ Generation {gen+1}/{GENERATIONS}", flush=True)
        
        scores = []
        MAX_PARALLEL = 4
        print(f"   [System] Spawning {MAX_PARALLEL} parallel training threads...", flush=True)
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            future_to_genome = {
                executor.submit(evaluate_genome, genome, X_train, y_train, X_val, y_val): genome 
                for genome in population
            }
            
            for i, future in enumerate(future_to_genome):
                genome = future_to_genome[future]
                try:
                    score, weights = future.result()
                    scores.append((score, genome, weights))
                    print(f"   ✅ Finished Genome: Loss {score:.6f}", flush=True)
                except Exception as exc:
                    print(f"   💀 Crashed: {exc}", flush=True)
            
        scores.sort(key=lambda x: x[0])
        
        best_gen_score, best_gen_genome, best_weights = scores[0]
        if best_gen_score < best_ever_score:
            best_ever_score = best_gen_score
            best_ever_genome = best_gen_genome
            print(f"   👑 NEW KING FOUND! Loss: {best_ever_score:.6f}", flush=True)
            torch.save(best_weights, "zeus_king_model.pth")
            with open("king_genome.json", "w") as f:
                import json
                json.dump(best_ever_genome, f)
        
        # Survival & Evolution
        survivors = [s[1] for s in scores[:SURVIVORS]]
        new_pop = survivors[:]
        
        while len(new_pop) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_pop.append(child)
            
        population = new_pop
        print(f"   ➡️ Generation {gen+1} Complete. Best: {best_gen_score:.6f}", flush=True)

    print("\n" + "="*60)
    print("🏆 EVOLUTION COMPLETE")
    print(f"The King: {best_ever_genome}")
    print(f"Loss: {best_ever_score}")
    print("="*60)

    from verification_gauntlet import run_gauntlet
    run_gauntlet()

if __name__ == "__main__":
    run_evolution()
