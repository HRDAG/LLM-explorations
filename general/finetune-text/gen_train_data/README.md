# Prep training data

This script prepares training data for fine-tuning LLMs on document summarization tasks. It generates two types of examples:

1. Valid document examples from input CSV data:
   - Filters for documents between 2,000 and 5,000 tokens

2. Empty input examples:
   - Generates examples with no input text
   - Trains model to respond with "not enough information" 

## Input/Output

- Input: CSV file with 'ocr' and 'summary' columns
- Output: JSONL files containing:
  - training_data.jsonl (80% of examples)
  - validation_data.jsonl (20% of examples)

Each example includes system prompt, user input, and assistant response in the standard chat format.