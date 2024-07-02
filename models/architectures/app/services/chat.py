# app/chat.py

from flask import request, jsonify
# import fin_gpt_model  # hypothetical module for your FinGPT model
    
from flask import request, jsonify
from utilities.wrapper import run_openai_api
import logging
log = logging.getLogger(__name__)

def chat_with_model():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Interact with the OpenAI API
        response= run_openai_api(user_message,"Give news in 1 line")
        log.info(response)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

