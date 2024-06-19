from flask import request, jsonify
import os
from utilities.get_news import run_news_api
from utilities.wrapper import run_openai_api

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

log = logging.getLogger(__name__)


def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts."""
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

def verify():
    try:
        news = request.get_data(as_text=True) 
        if not news:
            return jsonify({"error": "No news provided"}), 400
        log.info(f"Received news: {news}")
        # Fetch news articles related to the title, using cache
        title = run_openai_api(news,"Give me keywords for this news article.")
        print(title)
        news_df = run_news_api(title)
        top_5_news_articles = news_df.head(5)
        news_articles = top_5_news_articles.to_dict(orient='records')
        print(news_articles)
        # Get OpenAI response
        openai_response = news
        # Calculate similarity for each article title with the OpenAI response
        
        similarities = []
        for article in news_articles:  # Consider top 5 articles
            if isinstance(article['title'], str):  # Ensure the title is a string
                similarity = calculate_similarity(openai_response, article['title'])
                similarities.append({
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'url': article['url'],
                    'similarity': similarity
                })

        # Calculate average similarity
        avg_similarity = sum([sim['similarity'] for sim in similarities]) / len(similarities) if similarities else 0

        verification_results = {
            "openai_response": openai_response,
            "related_articles": similarities,
            "average_similarity": avg_similarity
        }
        
        return jsonify({"results": verification_results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

