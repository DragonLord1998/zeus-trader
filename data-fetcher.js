const YahooFinance = require('yahoo-finance2').default;
const yahooFinance = new YahooFinance();
const Parser = require('rss-parser');
const Sentiment = require('sentiment');
const config = require('./config');

const parser = new Parser();
const sentiment = new Sentiment();

async function getHistoricalData(symbol, days = 730) {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  const queryOptions = { 
    period1: startDate.toISOString().split('T')[0],
    period2: endDate.toISOString().split('T')[0],
    interval: '1d'
  };

  const assets = [
      { symbol: symbol, key: 'target' }, // RELIANCE.NS
      { symbol: '^NSEI', key: 'nifty' },
      { symbol: 'CL=F', key: 'oil' },
      { symbol: 'GC=F', key: 'gold' },
      { symbol: 'SI=F', key: 'silver' },
      { symbol: 'HG=F', key: 'copper' },
      { symbol: '^TNX', key: 'us10y' },
      { symbol: 'DX-Y.NYB', key: 'dxy' }
  ];

  try {
      // Fetch all in parallel
      const promises = assets.map(a => yahooFinance.chart(a.symbol, queryOptions).catch(e => null));
      const results = await Promise.all(promises);

      // Create a map of Date -> Object
      // Primary driver is the Target Symbol's dates (we only care when India is open)
      const masterData = new Map();
      
      const targetResult = results[0];
      if (!targetResult || !targetResult.quotes) return [];

      // Initialize with Target Data
      targetResult.quotes.forEach(q => {
          if (!q.close) return;
          const dateStr = q.date.toISOString().split('T')[0];
          masterData.set(dateStr, {
              date: q.date,
              open: q.open,
              high: q.high,
              low: q.low,
              close: q.close,
              volume: q.volume
          });
      });

      // Merge Context Data
      for (let i = 1; i < assets.length; i++) {
          const res = results[i];
          const key = assets[i].key;
          
          if (!res || !res.quotes) continue;

          res.quotes.forEach(q => {
              const dateStr = q.date.toISOString().split('T')[0];
              if (masterData.has(dateStr)) {
                  const entry = masterData.get(dateStr);
                  // specific handling: some assets might be closed when India is open
                  // We store the close price
                  entry[key] = q.close; 
              }
          });
      }

      // Convert back to Array and Forward Fill missing values
      // (e.g. India open, US closed -> Gold price is missing -> use yesterday's Gold)
      let sortedData = Array.from(masterData.values()).sort((a,b) => a.date - b.date);
      
      // Keys to fill
      const contextKeys = assets.slice(1).map(a => a.key);
      
      let lastValues = {};
      
      sortedData = sortedData.map(row => {
          contextKeys.forEach(k => {
              if (row[k] !== undefined && row[k] !== null) {
                  lastValues[k] = row[k];
              } else {
                  // Fill with last known value (or 0 if start)
                  row[k] = lastValues[k] || 0; 
              }
          });
          return row;
      });

      return sortedData;

  } catch (error) {
    console.error(`Error fetching history:`, error.message);
    return [];
  }
}

async function getQuote(symbol) {
  try {
    return await yahooFinance.quote(symbol);
  } catch (error) {
    console.error(`Error fetching quote for ${symbol}:`, error.message);
    return null;
  }
}

async function getNewsSentiment() {
  const feedUrl = `https://news.google.com/rss/search?q=${config.news.topic}&gl=IN&ceid=IN:en`;
  try {
    const feed = await parser.parseURL(feedUrl);
    let totalScore = 0;
    let count = 0;

    feed.items.forEach(item => {
      const textToAnalyze = item.title + ' ' + (item.contentSnippet || '');
      const result = sentiment.analyze(textToAnalyze);
      totalScore += result.score;
      count++;
    });

    return count > 0 ? (totalScore / count) : 0;
  } catch (error) {
    console.error('Error fetching news:', error.message);
    return 0;
  }
}

module.exports = {
  getHistoricalData,
  getQuote,
  getNewsSentiment
};
