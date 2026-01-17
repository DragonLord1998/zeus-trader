# ⚡ Zeus Trader: AI-Powered Quantitative Trading Bot

**Zeus Trader** is a deep-learning-based algorithmic trading system designed to predict stock market movements by analyzing a "Multi-Modal" dataset comprising price history, technical indicators, and global macroeconomic correlations.

Unlike simple trading bots that only look at a single stock's chart, Zeus Trader understands the "Market Context" by monitoring Oil, Gold, Silver, Copper, US Bond Yields, and Currency Indices simultaneously.

---

## 🧠 Core Architecture

### 1. The Model: Deep LSTM
At the heart of Zeus is a **Long Short-Term Memory (LSTM)** Neural Network, built with **PyTorch**.
- **Depth:** 4 Layers (Deep Learning) to capture hierarchical patterns (Daily noise -> Weekly trends -> Macro cycles).
- **Width:** 512 Neurons per layer (High capacity for feature extraction).
- **Dropout:** 20% (Prevents overfitting/memorization).

### 2. Multi-Asset "Global Vision"
Zeus doesn't just watch the stock price. It watches the world.

**Input Features:**
1.  **Target Asset:** Price (Close), Volume.
2.  **Technical Indicators:**
    -   **RSI (14):** Relative Strength Index (Overbought/Oversold).
    -   **MACD:** Moving Average Convergence Divergence (Momentum).
    -   **SMA (50):** Trend direction.
3.  **Macroeconomic Correlators:**
    -   **🛢️ Oil (CL=F):** Crucial for Energy/Industrial stocks (and India's import bill).
    -   **🥇 Gold (GC=F):** Market fear/safe-haven index.
    -   **🥈 Silver (SI=F):** Industrial demand indicator.
    -   **🔶 Copper (HG=F):** Economic health indicator.
    -   **💵 US 10Y Yield (^TNX):** The global "Risk-Free Rate" (affects FII flows).
    -   **💲 DXY (Dollar Index):** Strength of USD vs INR.
    -   **📈 Nifty 50 (^NSEI):** Broader market sentiment.

### 3. News Sentiment Analysis
Zeus analyzes Google News headlines for market sentiment using NLP, adjusting predictions based on bullish/bearish news flow.

### 4. The "Grid Search" Optimizer
Zeus doesn't guess parameters. It **evolves**.
The optimizer runs a brute-force simulation over different strategies:
- **Model Sizes:** 256 vs 512 vs 1024 units.
- **Depths:** 3 vs 4 vs 5 layers.
- **Thresholds:** Conservative (0.6%) vs Aggressive (0.3%).

---

## 📂 Project Structure

```
zeus_core/
├── config.py          # Centralized configuration
├── data_fetcher.py    # Yahoo Finance data download
├── feature_engine.py  # Technical indicators & normalization
├── model.py           # PyTorch LSTM model
├── backtest.py        # Trading simulation
├── optimizer.py       # Grid search hyperparameter tuning
├── sentiment.py       # News sentiment analysis
├── main.py            # Entry point (CLI)
└── requirements.txt   # Python dependencies
```

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- NVIDIA GPU (Recommended) with CUDA.

### Setup
```bash
# 1. Clone Repository
git clone https://github.com/DragonLord1998/zeus-trader.git
cd zeus-trader/zeus_core

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Download TextBlob data (for sentiment)
python -m textblob.download_corpora
```

### Running Zeus Trader

```bash
# Run Grid Search Optimizer (default)
python main.py

# Generate Tomorrow's Prediction
python main.py --predict

# Run Single Backtest
python main.py --backtest
```

---

## 📊 Performance Benchmark (Case Study: Reliance Industries)

**Best Configuration Found:**
- **Model:** 512 Units x 4 Layers
- **Strategy:** Swing Trading (Threshold 0.5%)
- **Result:** **+6.37% ROI** in 200 days (vs -1.2% Market Benchmark).
- **Behavior:** High Patience. The bot ignored market noise and only executed 2 high-quality trades, capturing a +74 point move.

---

## 🔮 Roadmap: Scaling to Nifty 100

The current engine is optimized for a single ticker. The next phase involves:
1.  **Scanner Mode:** Iterate through all **Nifty 100** tickers.
2.  **Portfolio Manager:** Allocating capital dynamically to the top 5 predicted stocks.
3.  **Paper Trading:** Connecting to a broker API (Zerodha/Angel) for live alerts.

---

*Disclaimer: This project is for educational and research purposes only. Algorithmic trading involves significant risk.*