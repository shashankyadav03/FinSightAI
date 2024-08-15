from datasets import load_dataset, Dataset
import pandas as pd
import json
from huggingface_hub import HfApi, HfFolder

# Step 1: Load the financial-qa-10K dataset
financial_qa_10K = load_dataset("virattt/financial-qa-10K")

# Step 2: Load your custom dataset from a JSON file
with open('custom_dataset.json', 'r') as f:
    custom_data = json.load(f)

# Convert your custom data to a Pandas DataFrame
df_custom = pd.DataFrame(custom_data)

# Step 3: Convert the custom dataset to a Hugging Face Dataset object
hf_custom_dataset = Dataset.from_pandas(df_custom)

# Step 4: Combine the two datasets
combined_dataset = financial_qa_10K['train'].concatenate(hf_custom_dataset)

# Step 5: Save the combined dataset locally (optional)
combined_dataset.save_to_disk('./combined_financial_dataset')

# Step 6: Upload the combined dataset to Hugging Face

# Define your dataset repository name on Hugging Face
# dataset_repo_name = "shashankyadav03/combined-financial-dataset"

# # Push the combined dataset to Hugging Face Hub
# combined_dataset.push_to_hub(dataset_repo_name)

print("Dataset uploaded and deployed successfully!")

# Step 7: Verify the Dataset on Hugging Face
print(f"Dataset can be accessed at: https://huggingface.co/datasets/{dataset_repo_name}")
