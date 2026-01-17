const config = require('./config');
const tf = require('@tensorflow/tfjs');
const { getHistoricalData } = require('./data-fetcher');
const { calculateIndicators, normalize } = require('./feature-engine');
const { createModel, trainModel, createSequences } = require('./model');

// Allow params to be overridden
async function runBacktest(overrides = {}) {
  // Determine backend (WASM for Mac, Node/GPU for Linux if available)
  try {
      if (tf.getBackend() !== 'node' && tf.getBackend() !== 'tensorflow') {
          await tf.setBackend('wasm'); 
      }
  } catch(e) {}

  const settings = { ...config.model, ...overrides };
  const features = overrides.features || ['close', 'rsi', 'macd', 'sma', 'volume', 'nifty', 'oil', 'gold', 'silver', 'copper', 'us10y', 'dxy'];

  // 1. Fetch Data (Cache this outside if possible for speed, but okay for now)
  const target = config.targets[1]; 
  const rawData = await getHistoricalData(target.symbol, 1000); 
  
  if (rawData.length === 0) return { profit: -99999, winRate: 0, trades: 0 };

  // 2. Process
  const processedData = calculateIndicators(rawData);
  
  // 3. Split
  const splitIndex = Math.floor(processedData.length * 0.8);
  const trainDataRaw = processedData.slice(0, splitIndex);
  const testDataRaw = processedData.slice(splitIndex);

  // 4. Normalize
  const { scaledData: trainScaled, min, max } = normalize(trainDataRaw);
  
  const scaleRow = (row) => {
    const scaled = {};
    Object.keys(row).forEach(key => {
        if (min[key] !== undefined) {
            scaled[key] = (row[key] - min[key]) / (max[key] - min[key] || 1);
        } else {
            scaled[key] = row[key];
        }
    });
    return scaled;
  };
  const testScaled = testDataRaw.map(scaleRow);

  // Custom Sequence Creator for dynamic features
  const createSequencesCustom = (data, lookBack, feats) => {
      const rawValues = data.map(d => {
        return feats.map(f => {
             const val = d[f];
             if (typeof val !== 'number' || isNaN(val)) return 0;
             return val;
        });
      });
    
      const inputs = [];
      const labels = [];
      for (let i = lookBack; i < rawValues.length; i++) {
        inputs.push(rawValues.slice(i - lookBack, i)); 
        labels.push([rawValues[i][0]]); 
      }
      return { inputs, labels };
  };

  // 5. Train
  const { inputs, labels } = createSequencesCustom(trainScaled, settings.lookBackPeriod, features);
  const inputShape = [inputs[0].length, inputs[0][0].length];
  
  const model = createModel(inputShape);
  await trainModel(model, { inputs, labels }, settings.epochs); // Use override epochs

  // 6. Simulate
  let balance = 100000;
  let shares = 0;
  let initialBalance = balance;
  let wins = 0;
  let totalTrades = 0;
  
  let historyWindow = [...trainScaled.slice(-settings.lookBackPeriod)];
  
  for (let i = 0; i < testScaled.length; i++) {
     const todayTrueData = testScaled[i];
     const truePrice = testDataRaw[i].close;

     // Prepare Input
     const cleanHistory = historyWindow.map(row => {
         return features.map(f => {
             const val = row[f];
             if (typeof val !== 'number' || isNaN(val)) return 0;
             return val;
         });
     });

     // Check shape consistency
     if (cleanHistory.length !== settings.lookBackPeriod || cleanHistory[0].length !== features.length) {
         historyWindow.shift();
         historyWindow.push(todayTrueData);
         continue;
     }

     const inputTensor = tf.tensor3d([cleanHistory]);
     const predTensor = model.predict(inputTensor);
     const predValue = predTensor.dataSync()[0];
     
     const predictedPrice = predValue * (max.close - min.close) + min.close;
     const lastKnownClose = (historyWindow[historyWindow.length-1].close * (max.close - min.close)) + min.close;
     
     // Trading Logic with Dynamic Threshold
     let action = 'HOLD';
     const threshold = overrides.threshold || 0.005;

     if (predictedPrice > lastKnownClose * (1 + threshold)) { 
         if (balance > truePrice) {
             const canBuy = Math.floor(balance / truePrice);
             balance -= canBuy * truePrice;
             shares += canBuy;
             action = 'BUY';
         }
     } else if (predictedPrice < lastKnownClose * (1 - threshold)) {
         if (shares > 0) {
             balance += shares * truePrice;
             shares = 0;
             action = 'SELL';
         }
     }

     const actualMove = truePrice > lastKnownClose ? 'UP' : 'DOWN';
     const predMove = predictedPrice > lastKnownClose ? 'UP' : 'DOWN';
     if (action !== 'HOLD' && actualMove === predMove) wins++;
     if (action !== 'HOLD') totalTrades++;

     historyWindow.shift();
     historyWindow.push(todayTrueData);
     
     inputTensor.dispose();
     predTensor.dispose();
  }

  const finalValue = balance + (shares * testDataRaw[testDataRaw.length-1].close);
  const profit = finalValue - initialBalance;
  const roi = (profit / initialBalance) * 100;
  const winRate = totalTrades > 0 ? ((wins/totalTrades)*100) : 0;

  // Cleanup
  model.dispose();

  return {
      settings: { ...settings, threshold: overrides.threshold, features },
      roi: parseFloat(roi.toFixed(2)),
      trades: totalTrades,
      winRate: parseFloat(winRate.toFixed(1))
  };
}

module.exports = { runBacktest };
