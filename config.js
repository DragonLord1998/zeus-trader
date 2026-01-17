module.exports = {
  // Global Market Settings
  market: 'IN', // India
  timezone: 'Asia/Kolkata',

  // Target Assets to Predict
  targets: [
    { symbol: '^NSEI', name: 'Nifty 50' },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services' },
    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank' }
  ],

  // Correlated Assets (The "World Context")
  correlations: [
    { symbol: 'INR=X', name: 'USD/INR' },      // Currency Strength
    { symbol: 'BZ=F', name: 'Brent Crude Oil' }, // Oil Price (Crucial for India)
    { symbol: 'GC=F', name: 'Gold' },          // Safe Haven
    { symbol: '^GSPC', name: 'S&P 500' },      // Global Sentiment
    { symbol: '^TNX', name: 'US 10Y Bond Yield' } // Global Rate Sentiment
  ],

  // News Configuration
  news: {
    lang: 'en-IN',
    topic: 'NIFTY+50+economy+india'
  },

  // Model Settings
  model: {
    lookBackPeriod: 60, // Days of history to look at
    epochs: 30,         // Training iterations
    batchSize: 32
  }
};
