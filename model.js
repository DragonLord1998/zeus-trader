const tf = require('@tensorflow/tfjs');
const config = require('./config');

function createModel(inputShape) {
  const model = tf.sequential();

  model.add(tf.layers.lstm({
    units: 50,
    returnSequences: true, // Pass sequences to the next LSTM layer if we had one, or False if next is Dense
    inputShape: inputShape
  }));
  
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.lstm({
    units: 50,
    returnSequences: false
  }));

  model.add(tf.layers.dense({ units: 1 })); // Predict 1 value (Price)

  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
  });

  return model;
}

async function trainModel(model, data, epochs = config.model.epochs) {
  // Data must be formatted as [samples, timeSteps, features]
  const { inputs, labels } = data;
  
  const xs = tf.tensor3d(inputs);
  const ys = tf.tensor2d(labels);

  console.log('Training model...');
  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: config.model.batchSize,
    shuffle: true,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            if (epoch % 5 === 0) console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(5)}`);
        }
    }
  });
  
  xs.dispose();
  ys.dispose();
}

function createSequences(data, lookBack = config.model.lookBackPeriod) {
    // data is array of objects {close, rsi...}
    // Explicit feature list to match backtest.js
    const features = ['close', 'rsi', 'macd', 'sma', 'volume', 'nifty', 'oil', 'gold', 'silver', 'copper', 'us10y', 'dxy'];

    const rawValues = data.map(d => {
        return features.map(f => {
             const val = d[f];
             // Simple sanitization for training data too
             if (typeof val !== 'number' || isNaN(val)) return 0;
             return val;
        });
    });
    
    const inputs = [];
    const labels = [];
    
    for (let i = lookBack; i < rawValues.length; i++) {
        inputs.push(rawValues.slice(i - lookBack, i)); // Past 60 days
        // Predict the NEXT 'close'. 'close' is at index 0 in our features array ['close', 'rsi'...]
        labels.push([rawValues[i][0]]); 
    }
    
    return { inputs, labels };
}

module.exports = {
  createModel,
  trainModel,
  createSequences
};
