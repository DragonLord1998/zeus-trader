# Zeus Trader Core

An AI-powered Market Trading Bot using LSTM (TensorFlow.js) and Multi-Asset Analysis (Stocks, Commodities, Macro).

## 🚀 Setup on Cloud GPU (RunPod / Lambda Labs / AWS)

This project is optimized for Node.js.

### 1. Prerequisites
Ensure you have Node.js (v18+) and CUDA drivers installed on your server.
(Most RunPod TensorFlow templates already have CUDA).

### 2. Installation
```bash
# Install dependencies
npm install

# ⚡ IMPORTANT FOR GPU SPEED:
# Remove the CPU/WASM versions and install the GPU binding
npm uninstall @tensorflow/tfjs-node
npm install @tensorflow/tfjs-node-gpu
```

### 3. Configuration
Edit `config.js` to change:
- Target Stock (Default: Reliance)
- Lookback periods
- Correlated assets

### 4. Running the Optimizer
To run the Grid Search algorithm that finds the best strategy:

```bash
node optimizer.js
```

This will output `optimization-results.json` with the best parameters.

### 5. Running the Backtest (Single Run)
```bash
node backtest.js
```
