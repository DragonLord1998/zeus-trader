"""
Zeus Trader Sentiment Analyzer
==============================
News sentiment analysis using RSS feeds and TextBlob.
"""

import feedparser
from textblob import TextBlob
from typing import List, Dict

from config import CONFIG


def fetch_news(topic: str = None, max_items: int = 20) -> List[Dict]:
    """
    Fetch news articles from Google News RSS.
    
    Args:
        topic: Search topic (default: from CONFIG)
        max_items: Maximum number of articles to fetch
    
    Returns:
        List of news items with title and content
    """
    import urllib.parse
    
    if topic is None:
        topic = CONFIG["sentiment"]["topic"]
    
    encoded_topic = urllib.parse.quote(topic)
    url = f"https://news.google.com/rss/search?q={encoded_topic}&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(url)
        
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        
        return items
        
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Text to analyze
    
    Returns:
        Sentiment polarity (-1 to +1)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_market_sentiment(topic: str = None, verbose: bool = False) -> float:
    """
    Get aggregate market sentiment from news.
    
    Args:
        topic: Search topic
        verbose: Print individual article sentiments
    
    Returns:
        Aggregate sentiment score scaled to -5 to +5
    """
    if not CONFIG["sentiment"]["enabled"]:
        return 0.0
    
    print("📰 Analyzing News Sentiment...")
    
    # Fetch news
    news = fetch_news(topic)
    
    if not news:
        print("   ⚠️ No news found, returning neutral sentiment")
        return 0.0
    
    # Analyze each article
    scores = []
    
    for item in news:
        text = f"{item['title']} {item['summary']}"
        score = analyze_sentiment(text)
        scores.append(score)
        
        if verbose:
            sentiment_label = "📈" if score > 0 else "📉" if score < 0 else "➖"
            print(f"   {sentiment_label} [{score:+.2f}] {item['title'][:60]}...")
    
    # Calculate aggregate score
    if not scores:
        return 0.0
    
    avg_score = sum(scores) / len(scores)
    
    # Scale from (-1, +1) to (-5, +5)
    scaled_score = avg_score * 5
    
    sentiment_label = "BULLISH 🐂" if scaled_score > 1 else "BEARISH 🐻" if scaled_score < -1 else "NEUTRAL ➖"
    print(f"   🧠 Market Sentiment: {scaled_score:+.2f} ({sentiment_label})")
    print(f"   📊 Based on {len(scores)} articles")
    
    return scaled_score


def get_sentiment_factor(sentiment_score: float) -> float:
    """
    Convert sentiment score to a price adjustment factor.
    
    Args:
        sentiment_score: Score from -5 to +5
    
    Returns:
        Multiplier factor (e.g., 1.025 for +5 sentiment)
    """
    # 0.5% adjustment per sentiment point
    return 1 + (sentiment_score * 0.005)


    return 1 + (sentiment_score * 0.005)


def get_stock_sentiment(symbol: str) -> Dict:
    """
    Get sentiment for a specific stock.
    
    Args:
        symbol: Stock symbol (e.g. 'RELIANCE.NS')
        
    Returns:
        Dict with score and article count
    """
    # Clean symbol
    clean_sym = symbol.replace('.NS', '').replace('.BO', '')
    topic = f"{clean_sym} share price india news"
    
    news = fetch_news(topic, max_items=10)
    
    if not news:
        return {'sentiment': 0.0, 'news_count': 0}
    
    scores = []
    for item in news:
        text = f"{item['title']} {item['summary']}"
        scores.append(analyze_sentiment(text))
        
    avg_score = sum(scores) / len(scores) if scores else 0
    scaled_score = avg_score * 5  # Scale to -5 to +5 range
    
    return {
        'sentiment': round(scaled_score, 2),
        'news_count': len(scores)
    }


if __name__ == "__main__":
    # Test sentiment analysis
    score = get_market_sentiment(verbose=True)
    factor = get_sentiment_factor(score)
    print(f"\n📊 Sentiment Factor: {factor:.4f}")
    
    # Test stock sentiment
    print("\n📰 Testing Stock Sentiment for RELIANCE...")
    stock_sent = get_stock_sentiment("RELIANCE.NS")
    print(f"   Score: {stock_sent['sentiment']} (from {stock_sent['news_count']} articles)")
