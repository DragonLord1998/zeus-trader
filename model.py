"""
Zeus Trader LSTM Model
======================
PyTorch LSTM neural network for price prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeusLSTM(nn.Module):
    """
    Deep LSTM neural network for stock price prediction.
    
    Architecture:
        - Multiple stacked LSTM layers with dropout
        - Final fully connected layer for regression output
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 512, 
        num_layers: int = 4, 
        output_dim: int = 1, 
        dropout: float = 0.2
    ):
        super(ZeusLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take only the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        prediction = self.fc(last_output)
        
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ZeusTransformer(nn.Module):
    """
    Transformer-based model for time series prediction (TimesFM alternative).
    Uses Multi-Head Self-Attention to capture long-range dependencies.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=1, dropout=0.2):
        super(ZeusTransformer, self).__init__()
        
        # Feature projection
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.decoder = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Take the last time step
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.decoder(x)


def create_sequences(data: np.ndarray, lookback: int) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Scaled data array of shape (samples, features)
        lookback: Number of time steps to look back
    
    Returns:
        Tuple of (X, y) where X is (samples, lookback, features) and y is (samples,)
    """
    X, y = [], []
    
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, 0])  # Predict Close price (index 0)
    
    return np.array(X), np.array(y)


def train_model(
    model: ZeusLSTM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    verbose: bool = True
) -> list:
    """
    Train the LSTM model.
    
    Args:
        model: ZeusLSTM model instance
        X_train: Training features (samples, lookback, features)
        y_train: Training labels (samples,)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        verbose: Print training progress
    
    Returns:
        List of loss values per epoch
    """
    # Use config defaults if not specified
    if epochs is None:
        epochs = CONFIG["model"]["epochs"]
    if batch_size is None:
        batch_size = CONFIG["model"]["batch_size"]
    if learning_rate is None:
        learning_rate = CONFIG["model"]["learning_rate"]
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
    
    return losses


def predict(model: ZeusLSTM, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with the model.
    
    Args:
        model: Trained ZeusLSTM model
        X: Input features (samples, lookback, features)
    
    Returns:
        Predictions array
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    return predictions


def get_device_info() -> str:
    """Get information about the compute device."""
    if device.type == 'cuda':
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


if __name__ == "__main__":
    # Test the model
    print(f"🚀 Device: {get_device_info()}")
    
    # Create a dummy model
    model = ZeusLSTM(input_dim=12, hidden_dim=128, num_layers=2).to(device)
    print(f"✅ Model created: {model}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 60, 12).to(device)
    output = model(dummy_input)
    print(f"✅ Forward pass: input {dummy_input.shape} -> output {output.shape}")
