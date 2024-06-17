# app/finetune.py

from flask import request, jsonify
import pandas as pd
FROM_REMOTE=True

base_model = 'llama2'
peft_model = 'FinGPT/fingpt-mt_llama2-7b_lora' if FROM_REMOTE else 'finetuned_models/MT-llama2-linear_202309210126'

# model, tokenizer = load_model(base_model, peft_model, FROM_REMOTE)

def fine_tune():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        data = pd.read_csv(file)
        # Perform data preprocessing if needed
        # data = preprocess(data)
        
        # Fine-tune the model
        # model.fine_tune(data)
        return jsonify({"message": "Model fine-tuned successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
