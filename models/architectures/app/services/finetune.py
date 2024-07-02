import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import logging
from flask import Flask, request, jsonify
from io import StringIO

logging.basicConfig(level=logging.INFO)

def load_tokenizer_and_model(model_name="gpt2"):
    """
    Load the tokenizer and model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Ensure the tokenizer will use the end-of-sequence token as the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model


def preprocess_data(tokenizer, data_path):
    """
    Load and preprocess the data from a CSV file.
    """
    data = pd.read_csv(data_path)
    texts = data['text'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    return encodings

class TextDataset(torch.utils.data.Dataset):
    """
    A custom torch dataset class for tokenized texts. Includes labels for training.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Set labels to be the same as input_ids
        return item


def fine_tune():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in the request")
        return jsonify({"error": "No selected file"}), 400

    file.stream.seek(0)  # Reset file stream position
    data = pd.read_csv(StringIO(file.read().decode('utf-8')))
    logging.info(f"File {file.filename} read successfully")

    tokenizer, model = load_tokenizer_and_model("gpt2")
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, return_tensors="pt")

    dataset = TextDataset(encodings)
    training_args = TrainingArguments(
        output_dir='./finetuned_model',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    logging.info("Training started")

    trainer.save_model('./finetuned_model')
    logging.info("Model fine-tuned and saved successfully")

    return jsonify({"message": "Model fine-tuned successfully"}), 200


def load_finetuned_model(model_path='./finetuned_model'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat2():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    tokenizer, model = load_finetuned_model()
    response = generate_response(model, tokenizer, user_message)
    return jsonify({"response": response}), 200



# from flask import Flask, request, jsonify
# import pandas as pd
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from peft import get_peft_model, LoraConfig

# # Constants
# FROM_REMOTE = True
# TOKEN = "YOUR_HF_TOKEN"  # Only needed if using a private model
# BASE_MODEL = 'FinGPT/fingpt-mt_llama2-7b_lora'  # Update this based on actual model name
# OUTPUT_DIR = 'finetuned_models'

# def load_model(from_remote):
#     model_identifier = BASE_MODEL if not from_remote else BASE_MODEL + "-remote"
#     model = AutoModelForCausalLM.from_pretrained(model_identifier, use_auth_token=(TOKEN if from_remote else None))
#     tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_auth_token=(TOKEN if from_remote else None))
#     return model, tokenizer

# def fine_tune():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     try:
#         data = pd.read_csv(file)
#         model_output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(file.filename)[0])

#         if os.path.exists(model_output_dir):
#             return jsonify({"message": "Model already fine-tuned and cached"}), 200

#         model, tokenizer = load_model(FROM_REMOTE)
#         train_encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
#         train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'])

#         training_args = TrainingArguments(
#             output_dir=model_output_dir,
#             overwrite_output_dir=True,
#             num_train_epochs=3,  # Increased epochs for better fine-tuning
#             per_device_train_batch_size=4,
#             save_steps=500,
#             save_total_limit=1,
#             logging_dir='./logs'
#         )

#         trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
#         trainer.train()
#         trainer.save_model(model_output_dir)

#         return jsonify({"message": "Model fine-tuned successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def generate_text():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({"error": "No text provided"}), 400

#     text = data['text']
#     try:
#         model, tokenizer = load_model(FROM_REMOTE)
#         input_ids = tokenizer.encode(text, return_tensors='pt')
#         output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#         generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#         return jsonify({"generated_text": generated_text}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
