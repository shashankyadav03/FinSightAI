#!/bin/bash

# Create the main project directory and subdirectories
mkdir -p {FinGPT/{finbert,scripts},data/{raw,processed,external},notebooks,src/{data_processing,model/{custom,finGPT},evaluation,utilities},models,docs}

# Create a requirements.txt file
touch requirements.txt

echo "Project directory structure created successfully."
