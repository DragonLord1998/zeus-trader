#!/bin/bash
echo "🚀 ZEUS TRADER CLOUD LAUNCHER"
echo "============================="

# 1. System Updates
apt-get update && apt-get install -y git python3-pip python3-venv

# 2. Virtual Env (Optional if root, but good practice)
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies (Optimized for Cloud)
echo "📦 Installing Dependencies..."
pip install --upgrade pip
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas yfinance ta scikit-learn lxml html5lib

# 4. Run Master Script
echo "🔥 Starting Global Model Training..."
python3 zeus_cloud_master.py
