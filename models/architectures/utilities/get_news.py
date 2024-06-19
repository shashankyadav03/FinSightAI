import requests
import json
import os
import hashlib
import pandas as pd
from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)

def get_cache_filename(query):
    """Generate a filename for the cache based on the query."""
    hash_object = hashlib.md5(query.encode())
    cache_filename = f"cache_{hash_object.hexdigest()}.json"
    return os.path.join('utilities/cache_news', cache_filename)

def get_api_key():
    load_dotenv()  # Load environment variables from the .env file
    return os.getenv('news_api_key')

def fetch_news(query, use_cache=True):
    """Fetch news articles related to a specific query using NewsAPI."""
    api_key = get_api_key()
    cache_filename = get_cache_filename(query)

    # Check if cache exists and is valid
    if use_cache and os.path.exists(cache_filename):
        with open(cache_filename, 'r') as cache_file:
            cached_data = json.load(cache_file)
            print(f"Using cached data for query: {query}")
            return cached_data

    # Fetch from NewsAPI if no cache or cache is invalid
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200 or 'articles' not in data:
        raise Exception(f"Failed to fetch news: {data.get('message', 'Unknown error')}")
    print(f"Fetched {len(data['articles'])} articles for query: {query}.")

    # Save fetched data to cache
    with open(cache_filename, 'w') as cache_file:
        json.dump(data, cache_file)

    return data

def save_news_to_file(news_data, filename):
    """Save news data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(news_data, f)
    print(f"Saved news data to {filename}.")

def load_news_from_file(filename):
    """Load news data from a JSON file and return as a pandas DataFrame."""
    with open(filename) as f:
        data = json.load(f)
        articles = data.get('articles', [])
        df = pd.DataFrame(articles)
    return df

def process_news_data(query, use_cache=True):
    """Fetch, save, and process news data for given queries."""
    news_data = fetch_news(query, use_cache)
    filename = os.path.join('utilities/data', f"{query}_news.json")
    save_news_to_file(news_data, filename)
    df = load_news_from_file(filename)
    df['query'] = query
    return df

def run_news_api(queries):
    """Run the news data processing pipeline."""
    # Ensure directories exist
    os.makedirs('utilities/cache_news', exist_ok=True)
    os.makedirs('utilities/data', exist_ok=True)
    
    news_df = process_news_data(queries)
    log.info(f"Processed news data for {len(news_df)} articles.")
    return news_df

# run_news_api('bitcoin')

    
