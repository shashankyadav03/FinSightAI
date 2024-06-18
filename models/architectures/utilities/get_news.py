import requests
import json
import pandas as pd

# Replace with your actual NewsAPI key
API_KEY = '25757eb42ae7439cb712fb91c1c57398'
industries = ['finance']

def fetch_news(industry, api_key):
    url = (f'https://newsapi.org/v2/everything?q={industry}&'
           'language=en&sortBy=publishedAt&apiKey=' + api_key)
    response = requests.get(url)
    data = response.json()
    print(f"Fetched {len(data['articles'])} articles for {industry}.")
    return data

# Fetch and store news data for each industry
for industry in industries:
    news_data = fetch_news(industry, API_KEY)
    with open(f'{industry}_news.json', 'w') as f:
        json.dump(news_data, f)
    print(f"Saved {industry} news data.")

# Load data into a pandas DataFrame for processing
df_list = []
for industry in industries:
    with open(f'{industry}_news.json') as f:
        data = json.load(f)
        articles = data.get('articles', [])
        df = pd.DataFrame(articles)
        df['industry'] = industry
        df_list.append(df)

news_df = pd.concat(df_list, ignore_index=True)

# Display the first few rows of the DataFrame
print(news_df.head())
