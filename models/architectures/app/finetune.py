from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

def finetune_model(data):
    # Example logic for finetuning
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load and preprocess data here
    # ...

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset="train_dataset",
        eval_dataset="eval_dataset",
    )

    trainer.train()
    return {"status": "Model fine-tuned successfully"}

