import csv
import json

# Path to the input CSV file
csv_file_path = '/Users/admin/Downloads/FinSightAI/FinDecide/Test/updated_file.csv'

# Path to the output JSON file
json_file_path = 'output.json'

# Initialize an empty list to store the data
data = []

# Read the CSV file and convert it to a list of dictionaries
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# Convert the list of dictionaries to JSON format
with open(json_file_path, mode='w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"CSV data has been successfully converted to JSON and saved to {json_file_path}.")
