# app/chat.py

from flask import request, jsonify
# import fin_gpt_model  # hypothetical module for your FinGPT model

def chat_with_model():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Interact with the model
        # model = fin_gpt_model.load_model()
        # response, id = model.chat(user_message)  # assuming model.chat returns a response and an ID
        response, id = "Hello! How can I help you?", 123  # placeholder values
        
        return jsonify({"response": response, "id": id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
