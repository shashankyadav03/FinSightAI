import requests
import json
import os
import hashlib
import pandas as pd
import logging
from services.dotenv_loader import load_environment

# Initialize the logger
log = logging.getLogger(__name__)

def get_cache_filename(query):
    """
    Generate a filename for caching the API response based on the query.

    Args:
        query (str): The search query for which the cache filename is generated.

    Returns:
        str: The path to the cache file based on the hashed query.
    """
    hash_object = hashlib.md5(query.encode())
    cache_filename = f"cache_{hash_object.hexdigest()}.json"
    return os.path.join('utilities/cache_news', cache_filename)

def get_api_key():
    """
    Retrieve the NewsAPI key from environment variables.

    Returns:
        str: The API key for NewsAPI.
    """
    load_environment()  # Ensure environment variables are loaded
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise EnvironmentError("NEWS_API_KEY is not set in the environment variables.")
    return api_key

def fetch_news(query, use_cache=True):
    """
    Fetch news articles related to a specific query using the NewsAPI.

    Args:
        query (str): The search query to fetch news articles for.
        use_cache (bool): Whether to use cached data if available.

    Returns:
        dict: A dictionary containing news articles fetched from the API.

    Raises:
        Exception: If the news API request fails.
    """
    api_key = get_api_key()
    cache_filename = get_cache_filename(query)

    # Check if cache exists and is valid
    if use_cache and os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'r') as cache_file:
                cached_data = json.load(cache_file)
                log.info(f"Using cached data for query: {query}")
                return cached_data
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load cache for query '{query}': {e}")

    # Fetch from NewsAPI if no cache or cache is invalid
    url = (
        f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=5'
        f'&apiKey={api_key}&sources=bbc-news,bloomberg,financial-times,fortune,reuters,the-wall-street-journal,google-news'
    )

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200 or 'articles' not in data:
            raise Exception(f"Failed to fetch news: {data.get('message', 'Unknown error')}")
        log.info(f"Fetched {len(data['articles'])} articles for query: {query}.")

        # Save fetched data to cache
        with open(cache_filename, 'w') as cache_file:
            json.dump(data, cache_file)
            log.info(f"Cached data for query: {query} at {cache_filename}")
        return data
    except requests.RequestException as e:
        log.error(f"Network error occurred while fetching news: {e}")
        raise Exception(f"Failed to fetch news due to a network error: {e}")
    except Exception as e:
        log.error(f"Failed to fetch news: {e}")
        raise Exception(f"Failed to fetch news: {e}")

def save_news_to_file(news_data, filename):
    """
    Save the fetched news data to a JSON file.

    Args:
        news_data (dict): The news data to save.
        filename (str): The path to the file where the data will be saved.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(news_data, f)
        log.info(f"Saved news data to {filename}.")
    except IOError as e:
        log.error(f"Failed to save news data to file {filename}: {e}")
        raise

def load_news_from_file(filename):
    """
    Load news data from a JSON file and return it as a pandas DataFrame.

    Args:
        filename (str): The path to the file from which the data will be loaded.

    Returns:
        pd.DataFrame: DataFrame containing the loaded news articles.
    """
    try:
        with open(filename) as f:
            data = json.load(f)
            articles = data.get('articles', [])
            df = pd.DataFrame(articles)
        log.info(f"Loaded news data from {filename}.")
        return df
    except (json.JSONDecodeError, IOError) as e:
        log.error(f"Failed to load news data from file {filename}: {e}")
        raise

def process_news_data(query, use_cache=True):
    """
    Fetch, save, and process news data for a given query.

    Args:
        query (str): The search query to fetch and process news for.
        use_cache (bool): Whether to use cached data if available.

    Returns:
        pd.DataFrame: DataFrame containing the processed news data.
    """
    news_data = fetch_news(query, use_cache)
    filename = os.path.join('utilities/data', f"{query}_news.json")
    save_news_to_file(news_data, filename)
    df = load_news_from_file(filename)
    df['query'] = query
    return df

def run_news_api(queries):
    """
    Run the news data processing pipeline.

    Args:
        queries (str): The search query to process news for.

    Returns:
        pd.DataFrame: DataFrame containing the processed news data.
    """
    try:
        # Ensure directories exist
        os.makedirs('utilities/cache_news', exist_ok=True)
        os.makedirs('utilities/data', exist_ok=True)
        
        news_df = process_news_data(queries)
        log.info(f"Processed news data for {len(news_df)} articles.")
        return news_df
    except Exception as e:
        log.error(f"Failed to run news API processing: {e}")
        raise
