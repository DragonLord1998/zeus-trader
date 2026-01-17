"""
Zeus Trader - Backtrader Verification
======================================
Verify our LSTM momentum strategy using the professional backtrader framework.
"""

import backtrader as bt
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from data_fetcher import fetch_data
from feature_engine import add_all_indicators, normalize_data, FEATURE_SETS
from model import ZeusLSTM, create_sequences, train_model, predict, device


class ZeusLSTMSignalData(bt.feeds.PandasData):
    """Custom data feed that includes our LSTM predictions."""
    lines = ('signal',)
    params = (('signal', -1),)


class ZeusLSTMStrategy(bt.Strategy):
    """
    Zeus LSTM Momentum Strategy for Backtrader.
    Uses pre-computed LSTM signals to make buy/sell decisions.
    """
    params = (
        ('threshold', 0.005),  # 0.5% threshold
        ('stake_percent', 0.95),  # 95% of portfolio per trade
    )
    
    def __init__(self):
        self.signal = self.datas[0].signal
        self.order = None
        self.trade_count = 0
        self.wins = 0
        self.entry_price = 0
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
            self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            if trade.pnl > 0:
                self.wins += 1
    
    def next(self):
        if self.order:
            return
        
        signal = self.signal[0]
        
        if not self.position:
            # Not in market - check for BUY signal
            if signal > self.params.threshold:
                stake = int(self.broker.getcash() * self.params.stake_percent / self.datas[0].close[0])
                if stake > 0:
                    self.order = self.buy(size=stake)
        else:
            # In market - check for SELL signal
            if signal < -self.params.threshold:
                self.order = self.sell(size=self.position.size)
    
    def stop(self):
        win_rate = (self.wins / self.trade_count * 100) if self.trade_count > 0 else 0
        print(f'\n📊 Strategy Stats: Trades={self.trade_count}, Win Rate={win_rate:.1f}%')


def prepare_data_with_predictions(symbol=None, test_days=200, lookback=60):
    """
    Fetch data, train LSTM, and generate predictions.
    Returns DataFrame with 'signal' column for backtrader.
    """
    print("📡 Fetching and processing data...")
    
    # Fetch data
    df = fetch_data(symbol)
    df = add_all_indicators(df)
    
    # Use momentum features (our winning set)
    feature_cols = FEATURE_SETS["momentum"]
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # Normalize
    scaled_data, scaler, _ = normalize_data(df, available_cols)
    
    # Split
    split_idx = len(scaled_data) - test_days
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - lookback:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    print(f"🏋️ Training LSTM model (512 units, 4 layers)...")
    
    # Train model
    model = ZeusLSTM(
        input_dim=len(available_cols),
        hidden_dim=512,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    train_model(model, X_train, y_train, epochs=25, verbose=False)
    
    # Generate predictions
    print("🔮 Generating predictions...")
    predictions = predict(model, X_test)
    
    # Calculate signals: (predicted - actual_last) / actual_last
    signals = []
    for i in range(len(predictions)):
        if i == 0:
            signals.append(0)
        else:
            last_close_scaled = X_test[i][-1][0]  # Last timestep, Close (index 0)
            pred = predictions[i]
            signal = (pred - last_close_scaled) / last_close_scaled if last_close_scaled != 0 else 0
            signals.append(signal)
    
    # Prepare backtrader dataframe
    test_df = df.iloc[split_idx:].copy()
    test_df = test_df.iloc[:len(signals)]  # Align lengths
    test_df['signal'] = signals
    
    # Rename columns for backtrader
    test_df = test_df.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Keep only required columns
    bt_df = test_df[['open', 'high', 'low', 'close', 'volume', 'signal']].copy()
    bt_df.index = pd.to_datetime(bt_df.index)
    
    return bt_df


def run_backtrader_verification(symbol=None, initial_cash=100000):
    """
    Run backtrader verification of our LSTM strategy.
    """
    print("=" * 60)
    print("🔬 BACKTRADER VERIFICATION")
    print("=" * 60)
    
    # Prepare data with LSTM predictions
    bt_df = prepare_data_with_predictions(symbol)
    
    print(f"\n📈 Running Backtrader simulation...")
    print(f"   Period: {bt_df.index[0].date()} to {bt_df.index[-1].date()}")
    print(f"   Initial Cash: ₹{initial_cash:,}")
    
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add data feed
    data = bt.feeds.PandasData(
        dataname=bt_df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # Add custom signal line
    class SignalData(bt.feeds.PandasData):
        lines = ('signal',)
        params = (('signal', 5),)  # Column index 5 = signal
    
    signal_data = SignalData(dataname=bt_df)
    cerebro.adddata(signal_data)
    
    # Add strategy
    cerebro.addstrategy(ZeusLSTMStrategy, threshold=0.005)
    
    # Set initial cash
    cerebro.broker.setcash(initial_cash)
    
    # Set commission (like Zerodha: 0.03%)
    cerebro.broker.setcommission(commission=0.0003)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.06)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run
    results = cerebro.run()
    strat = results[0]
    
    # Get results
    final_value = cerebro.broker.getvalue()
    roi = ((final_value - initial_cash) / initial_cash) * 100
    
    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    
    # Calculate Buy & Hold
    buy_hold_roi = ((bt_df['close'].iloc[-1] - bt_df['close'].iloc[0]) / bt_df['close'].iloc[0]) * 100
    
    print("\n" + "=" * 60)
    print("📊 BACKTRADER RESULTS")
    print("=" * 60)
    print(f"\n💰 Portfolio Performance:")
    print(f"   Initial Value:  ₹{initial_cash:,.2f}")
    print(f"   Final Value:    ₹{final_value:,.2f}")
    print(f"   ROI:            {roi:+.2f}%")
    print(f"   Buy & Hold:     {buy_hold_roi:+.2f}%")
    print(f"   Alpha:          {roi - buy_hold_roi:+.2f}%")
    
    print(f"\n📈 Risk Metrics:")
    sharpe_ratio = sharpe.get('sharperatio', 'N/A')
    print(f"   Sharpe Ratio:   {sharpe_ratio}")
    print(f"   Max Drawdown:   {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)
    
    return {
        'roi': roi,
        'buy_hold_roi': buy_hold_roi,
        'alpha': roi - buy_hold_roi,
        'final_value': final_value,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
    }


if __name__ == "__main__":
    results = run_backtrader_verification()
