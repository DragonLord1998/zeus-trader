const config = require('./config');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm'); // Import WASM backend

const { getHistoricalData, getNewsSentiment } = require('./data-fetcher');
const { calculateIndicators, normalize } = require('./feature-engine');
const { createModel, trainModel, createSequences } = require('./model');

async function main() {
  // Set backend to WASM
  await tf.setBackend('wasm');
  console.log(`⚡ ZEUS TRADER: Initializing... (Backend: ${tf.getBackend()})`);
  console.log(`🎯 Target Market: ${config.market} | Timezone: ${config.timezone}`);

  // 1. Pick a target (For MVP, let's just do the first one: Nifty 50 or Reliance)
  const target = config.targets[1]; // RELIANCE.NS
  console.log(`📈 Analyzing: ${target.name} (${target.symbol})`);

  // 2. Fetch Data
  console.log('📡 Fetching Market Data...');
  const rawData = await getHistoricalData(target.symbol);
  
  if (rawData.length === 0) {
      console.error('❌ No data found. Check your internet connection or ticker symbol.');
      return;
  }
  console.log(`✅ Fetched ${rawData.length} days of data.`);

  // 3. Fetch Sentiment
  console.log('📰 Analyzing News Sentiment...');
  const sentimentScore = await getNewsSentiment();
  console.log(`🧠 Market Sentiment Score: ${sentimentScore.toFixed(2)} (Range: -5 to +5)`);

  // 4. Feature Engineering
  console.log('⚙️  Calculating Technical Indicators (RSI, MACD, SMA)...');
  const processedData = calculateIndicators(rawData);
  console.log(`✅ Processed ${processedData.length} data points with indicators.`);

  // 5. Normalize
  const { scaledData, min, max } = normalize(processedData);

  // 6. Prepare Training Data
  const { inputs, labels } = createSequences(scaledData);
  const inputShape = [inputs[0].length, inputs[0][0].length]; // [60, features]

  // 7. Train Model
  console.log('🏋️  Training LSTM Neural Network...');
  const model = createModel(inputShape);
  await trainModel(model, { inputs, labels });

  // 8. Predict Tomorrow
  console.log('🔮 Generating Prediction...');
  
  // Get the LAST sequence from our data to predict the NEXT step
  const lastSequence = scaledData.slice(-config.model.lookBackPeriod);
  
  // Need to wrap it as a batch of 1: [1, 60, features]
  // tf is already imported at top level
  const inputTensor = tf.tensor3d([lastSequence.map(d => Object.values(d))]);
  
  const predictionTensor = model.predict(inputTensor);
  const predictionValue = predictionTensor.dataSync()[0]; // Scaled 0-1
  
  // De-normalize (Reverse the scaling for Price)
  const currentPrice = rawData[rawData.length - 1].close;
  const predictedPrice = predictionValue * (max.close - min.close) + min.close;
  
  // Incorporate Sentiment (Heuristic Adjustment)
  // If sentiment is super positive (>2), boost predict slightly. If negative, drag it down.
  // This is a naive "Fusion" for the MVP.
  const sentimentFactor = 1 + (sentimentScore * 0.005); // 0.5% shift per sentiment point
  const finalPrediction = predictedPrice * sentimentFactor;

  console.log('\n=======================================');
  console.log(`💵 Current Price:   ₹${currentPrice.toFixed(2)}`);
  console.log(`🤖 Model Forecast:  ₹${predictedPrice.toFixed(2)}`);
  console.log(`📰 Sentiment Adj:   x${sentimentFactor.toFixed(4)}`);
  console.log(`🔮 FINAL PREDICTION: ₹${finalPrediction.toFixed(2)}`);
  
  const movement = finalPrediction > currentPrice ? 'AGGRESSIVE BUY 🚀' : 'SELL / CAUTION 🔻';
  console.log(`📢 ZEUS SIGNAL:     ${movement}`);
  console.log('=======================================\n');

}

main();
