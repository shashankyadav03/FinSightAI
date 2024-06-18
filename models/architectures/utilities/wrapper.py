import json
import requests
import os
from dotenv import load_dotenv
import hashlib

def call_openai_api(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'messages': prompt,
        'max_tokens': 100,
        'model': 'gpt-3.5-turbo-0125',
        'temperature': 0
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        return f"Error: Rate limit exceeded"
    elif response.status_code == 401:
        return f"Error: Unauthorized"
    elif response.status_code == 403:
        return f"Error: Forbidden"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Get API key from .env file from openai tag
def get_api_key():
    load_dotenv()  # Load environment variables from the .env file
    return os.getenv('openai_api_key')

def get_prompt(prompt_content):
    prompt = [
        {'role': 'user', 'content': prompt_content},
        {'role': 'system', 'content': 'Reply in one line'}
    ]
    return prompt

def write_response_to_cache(prompt_hash, response):
    cache_dir = 'models/utilities/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{prompt_hash}.json')
    with open(cache_path, 'w') as outfile:
        json.dump(response, outfile)

def read_response_from_cache(prompt_hash):
    cache_path = os.path.join('models/utilities/cache', f'{prompt_hash}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as json_file:
            return json.load(json_file)
    return None

def hash_prompt(prompt_content):
    return hashlib.md5(prompt_content.encode()).hexdigest()

def get_message_content(data):
    return data['choices'][0]['message']['content']

# Test OpenAI API
def openai_api(prompt_content):
    api_key = get_api_key()
    prompt = get_prompt(prompt_content)
    prompt_hash = hash_prompt(prompt_content)
    
    # Check cache first
    cached_response = read_response_from_cache(prompt_hash)
    if cached_response:
        print("Cache hit. Returning cached response.")
        data = cached_response
    else:
        print("Cache miss. Calling API.")
        result = call_openai_api(prompt, api_key)
        write_response_to_cache(prompt_hash, result)
        data = result

    message_content = get_message_content(data)
    print(message_content)
    return message_content

def run_openai_api(prompt_content):
    message_content = openai_api(prompt_content)
    return message_content

# Example usage
# run_openai_api("How weather effects stocks?")
