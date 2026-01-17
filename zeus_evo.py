"""
ZEUS EVO: THE GENETIC ALGORITHM ENGINE
======================================
"Survival of the Fittest" for Financial AI.

This script runs an Evolutionary Search to find the optimal Transformer architecture.
Instead of checking every grid point, it:
1. Spawns 20 random models (Population).
2. Trains them.
3. Selects the Top 5 ("The Kings").
4. "Breeds" and "Mutates" them to create the next generation.
5. Repeats for N generations.

Usage:
    python3 zeus_evo.py
"""

import os
import time
import json
import random
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
GENERATIONS = 2    # Smoke Test
POPULATION_SIZE = 5 # Smoke Test
SURVIVORS = 2
MUTATION_RATE = 0.3
MUTATION_RATE = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧬 ZEUS EVO Initializing on {DEVICE}")

# --- GENOME DEFINITION ---
# A "Genome" is a dictionary of hyperparameters
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
    # Mutate one gene
    new_genome = genome.copy()
    gene_to_change = random.choice(list(GENE_SPACE.keys()))
    new_genome[gene_to_change] = random.choice(GENE_SPACE[gene_to_change])
    return new_genome

def crossover(parent1, parent2):
    # Mix genes
    child = {}
    for k in GENE_SPACE.keys():
        child[k] = parent1[k] if random.random() > 0.5 else parent2[k]
    return child

# --- DATA PIPELINE (Global Nifty 200) ---
# Reusing robust fetcher logic
def get_nifty_symbols():
    csv_path = "Market Data/MW-NIFTY-200-17-Jan-2026.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, header=None, skiprows=1)
            raw = [str(x).strip().replace('"','') for x in df[0].tolist()]
            return [f"{s}.NS" for s in raw if "NIFTY" not in s]
        except: pass
    
    # Wiki Fallback
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
        
        # Fast Features
        df['Close'] = df['Close'].replace(0, method='ffill')
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
        
        # Volatility
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

def get_data(lookback):
    # Cache data based on lookback to avoid re-fetching?
    # For simplicity in EVO (where lookback varies), we might standardize lookback=60 for all,
    # OR we re-process data for each individual.
    # To maximize GPU usage, we fix lookback=60 for V1 and evolve only model params.
    # Evolving Data Params (Lookback) is expensive I/O.
    # DECISION: Fix Lookback=60 for V1.
    pass

# Simplified global data loader (Fixed Lookback 60)
LOOKBACK_FIXED = 60
print("🌍 Building Global Data (Lookback=60)...")
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

# --- EVOLUTION ENGINE ---

def evaluate_genome(genome, X_train, y_train, X_val, y_val):
    # Train ONE genome
    # In V2 we can parallelize THIS function across GPUs/Processes
    try:
        model = EvoTransformer(genome, input_dim=X_train.shape[2]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=genome['lr'])
        criterion = nn.MSELoss()
        
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        
        scaler = torch.cuda.amp.GradScaler()
        
        # Short life: 5 epochs
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
                
        # Evaluate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # Evaluation on full val set (batched)
            val_preds = []
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=4096)
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx).squeeze()
                val_loss += criterion(pred, by).item()
        
        score = val_loss / len(val_loader)
        return score, model.state_dict()
    except Exception as e:
        print(f"💀 Genome Died: {e}")
        return float('inf'), None

def run_evolution():
    print(f"\n🧬 Starting Evolution: {GENERATIONS} Generations, {POPULATION_SIZE} Population")
    
    # Init Population
    population = [random_genome() for _ in range(POPULATION_SIZE)]
    
    # Data Split
    split = int(0.9 * len(GLOBAL_X))
    X_train, X_val = GLOBAL_X[:split], GLOBAL_X[split:]
    y_train, y_val = GLOBAL_Y[:split], GLOBAL_Y[split:]
    
    best_ever_score = float('inf')
    best_ever_genome = None
    
    print("🐛 Debug: Entering Generation Loop...", flush=True)

    for gen in range(GENERATIONS):
        print(f"\n⏳ Generation {gen+1}/{GENERATIONS}", flush=True)
        
        # Evaluate Population
        scores = []
        for i, genome in enumerate(population):
            print(f"   [Debug] Evaluating Genome {i+1}...", end="", flush=True)
            score, weights = evaluate_genome(genome, X_train, y_train, X_val, y_val)
            scores.append((score, genome, weights))
            print(f"   Input {i+1}: Loss {score:.6f} | Gene: {genome}")
            
        # Sort by Fitness (Lowest Loss)
        scores.sort(key=lambda x: x[0])
        
        # Save Best
        best_gen_score, best_gen_genome, best_weights = scores[0]
        if best_gen_score < best_ever_score:
            best_ever_score = best_gen_score
            best_ever_genome = best_gen_genome
            print(f"   👑 NEW KING FOUND! Loss: {best_ever_score:.6f}")
            torch.save(best_weights, "zeus_king_model.pth")
            with open("king_genome.json", "w") as f:
                json.dump(best_ever_genome, f, indent=2)
                
        # Selection (Top Survivors)
        survivors = [s[1] for s in scores[:SURVIVORS]]
        
        # Breeding
        new_pop = survivors[:] # Keep elites
        while len(new_pop) < POPULATION_SIZE:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_pop.append(child)
            
        population = new_pop
        print(f"   ➡️ Generation {gen+1} Complete. Best: {best_gen_score:.6f}")

    print("\n" + "="*60)
    print("🏆 EVOLUTION COMPLETE")
    print(f"The King: {best_ever_genome}")
    print(f"Loss: {best_ever_score}")
    print("="*60)
    

    # Verification using The Gauntlet
    from verification_gauntlet import run_gauntlet
    run_gauntlet()

if __name__ == "__main__":
    run_evolution()

