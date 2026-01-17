const config = require('./config');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');
const { getHistoricalData } = require('./data-fetcher');
const { calculateIndicators, normalize } = require('./feature-engine');
const { createModel, trainModel, createSequences } = require('./model');

async function runBacktest() {
  await tf.setBackend('wasm');
  console.log('🔙 STARTING BACKTEST SIMULATION...');

  // 1. Fetch ALL Data
  const target = config.targets[1]; // RELIANCE.NS
  const rawData = await getHistoricalData(target.symbol, 1000); // Get 3 years
  
  if (rawData.length === 0) return;

  // 2. Process
  const processedData = calculateIndicators(rawData);
  
  // 3. Split: Train (80%), Test (20%)
  const splitIndex = Math.floor(processedData.length * 0.8);
  const trainDataRaw = processedData.slice(0, splitIndex);
  const testDataRaw = processedData.slice(splitIndex);

  console.log(`📊 Data Split: Training on ${trainDataRaw.length} days, Testing on ${testDataRaw.length} days.`);

  // 4. Normalize based on TRAINING data only (to avoid look-ahead bias)
  // We need to export the min/max from train set to scale test set
  const { scaledData: trainScaled, min, max } = normalize(trainDataRaw);
  
  // Helper to scale test data using Train Min/Max
  const scaleRow = (row) => {
    const scaled = {};
    Object.keys(row).forEach(key => {
        if (min[key] !== undefined) {
            scaled[key] = (row[key] - min[key]) / (max[key] - min[key] || 1);
        } else {
            scaled[key] = row[key]; // date, etc.
        }
    });
    return scaled;
  };
  
  const testScaled = testDataRaw.map(scaleRow);

  // 5. Train Model
  const { inputs, labels } = createSequences(trainScaled);
  const inputShape = [inputs[0].length, inputs[0][0].length];
  
  const model = createModel(inputShape);
  await trainModel(model, { inputs, labels }, 20); // Faster training for backtest

  // 6. Simulate Trading
  let balance = 100000; // Starting Capital (INR)
  let shares = 0;
  let initialBalance = balance;
  let wins = 0;
  let totalTrades = 0;

  console.log('\n📅 Running Simulation...');
  
  // We need the sliding window from the end of Training into Testing
  // We start predicting the first day of Test Data.
  // Context needed: The last 60 days of Training Data.
  
  let historyWindow = [...trainScaled.slice(-config.model.lookBackPeriod)];
  
  for (let i = 0; i < testScaled.length; i++) {
     const todayTrueData = testScaled[i];
     const truePrice = testDataRaw[i].close;
     const trueDate = testDataRaw[i].date;

     // Explicit feature list to ensure consistency
     const features = ['close', 'rsi', 'macd', 'sma', 'volume', 'nifty', 'oil', 'gold', 'silver', 'copper', 'us10y', 'dxy'];
     
     const cleanHistory = historyWindow.map(row => {
         return features.map(f => {
             const val = row[f];
             if (typeof val !== 'number' || isNaN(val)) return 0;
             return val;
         });
     });

     // Check shape consistency
     const featuresCount = features.length;

     if (cleanHistory.some(row => row.length !== featuresCount)) {
         console.warn(`Skipping day ${i}: Inconsistent feature length`);
         continue; 
     }

     const inputTensor = tf.tensor3d([cleanHistory]);
     const predTensor = model.predict(inputTensor);
     const predValue = predTensor.dataSync()[0];
     
     const predictedPrice = predValue * (max.close - min.close) + min.close;
     
     // Current Price (Yesterday's Close for decision making, technically)
     // But for simplicity, let's say we make decision at Open based on prediction vs Open
     // Or Close-to-Close.
     // Strategy: If Predicted Close > Current Close (last known), BUY.
     
     const lastKnownClose = (historyWindow[historyWindow.length-1].close * (max.close - min.close)) + min.close;
     
     let action = 'HOLD';
     if (predictedPrice > lastKnownClose * 1.005) { // >0.5% upside
         if (balance > truePrice) {
             const canBuy = Math.floor(balance / truePrice);
             balance -= canBuy * truePrice;
             shares += canBuy;
             action = 'BUY';
         }
     } else if (predictedPrice < lastKnownClose * 0.995) { // >0.5% downside
         if (shares > 0) {
             balance += shares * truePrice;
             shares = 0;
             action = 'SELL';
         }
     }

     // Evaluate correctness (Direction)
     const actualMove = truePrice > lastKnownClose ? 'UP' : 'DOWN';
     const predMove = predictedPrice > lastKnownClose ? 'UP' : 'DOWN';
     if (action !== 'HOLD' && actualMove === predMove) wins++;
     if (action !== 'HOLD') totalTrades++;

     // Update Window: Remove oldest, add today's ACTUAL data (so next prediction has correct history)
     historyWindow.shift();
     historyWindow.push(todayTrueData);
     
     inputTensor.dispose();
     predTensor.dispose();
     
     // console.log(`${trueDate.toISOString().split('T')[0]}: ${action} @ ${truePrice} | Pred: ${predictedPrice.toFixed(1)}`);
  }

  // Final Liquidation
  const finalValue = balance + (shares * testDataRaw[testDataRaw.length-1].close);
  const profit = finalValue - initialBalance;
  const roi = (profit / initialBalance) * 100;

  console.log('\n=======================================\n');
  console.log(`💰 Initial Balance: ₹${initialBalance}`);
  console.log(`🏁 Final Balance:   ₹${finalValue.toFixed(2)}`);
  console.log(`📈 Net Profit:      ₹${profit.toFixed(2)} (${roi.toFixed(2)}%)`);
  console.log(`🎲 Trades: ${totalTrades} | Win Rate: ${totalTrades > 0 ? ((wins/totalTrades)*100).toFixed(1) : 0}%`);
  
  // Benchmark: Buy and Hold
  const startPrice = testDataRaw[0].close;
  const endPrice = testDataRaw[testDataRaw.length-1].close;
  const buyHoldRoi = ((endPrice - startPrice) / startPrice) * 100;
  console.log(`🐢 Buy & Hold ROI:  ${buyHoldRoi.toFixed(2)}%`);
  console.log('=======================================\n');

}

runBacktest();
