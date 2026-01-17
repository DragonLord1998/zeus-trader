const technicalIndicators = require('technicalindicators');

// Helper to extract a specific field (e.g., 'close') from the data array
const getField = (data, field) => data.map(d => d[field]);

function calculateIndicators(data) {
  const closes = getField(data, 'close');

  // RSI
  const rsiInput = {
    values: closes,
    period: 14
  };
  const rsi = technicalIndicators.RSI.calculate(rsiInput);

  // MACD
  const macdInput = {
    values: closes,
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false
  };
  const macd = technicalIndicators.MACD.calculate(macdInput);

  // SMA (50 day)
  const smaInput = {
    period: 50,
    values: closes
  };
  const sma = technicalIndicators.SMA.calculate(smaInput);

  // Align arrays
  const minLength = Math.min(rsi.length, macd.length, sma.length);
  const offset = data.length - minLength;

  const processedData = [];

  for (let i = 0; i < minLength; i++) {
    const dataIndex = offset + i; 
    
    const rsiVal = rsi[rsi.length - minLength + i];
    const macdVal = macd[macd.length - minLength + i];
    const smaVal = sma[sma.length - minLength + i];
    
    const row = data[dataIndex];

    processedData.push({
      date: row.date,
      close: row.close,
      rsi: rsiVal,
      macd: macdVal.histogram,
      sma: smaVal,
      volume: row.volume,
      // Macro Data (already forward-filled in fetcher)
      nifty: row.nifty || 0,
      oil: row.oil || 0,
      gold: row.gold || 0,
      silver: row.silver || 0,
      copper: row.copper || 0,
      us10y: row.us10y || 0,
      dxy: row.dxy || 0
    });
  }

  return processedData;
}

function normalize(data) {
  if (data.length === 0) return { scaledData: [], min: {}, max: {} };
  
  const fields = Object.keys(data[0]).filter(k => k !== 'date'); // Exclude date

  const max = {};
  const min = {};

  // Init
  fields.forEach(f => {
    max[f] = -Infinity;
    min[f] = Infinity;
  });

  // Find Min/Max
  data.forEach(d => {
    fields.forEach(f => {
      if (d[f] > max[f]) max[f] = d[f];
      if (d[f] < min[f]) min[f] = d[f];
    });
  });

  // Scale
  const scaledData = data.map(d => {
    const row = {};
    fields.forEach(f => {
      const range = max[f] - min[f];
      row[f] = range === 0 ? 0.5 : (d[f] - min[f]) / range;
    });
    return row;
  });

  return { scaledData, min, max };
}

module.exports = {
  calculateIndicators,
  normalize
};