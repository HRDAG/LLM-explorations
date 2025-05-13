# Long Context Evaluation for Criminal Justice Documents

## Overview
This repository is dedicated to evaluating the performance of long context windows on criminal justice documents. The project aims to analyze how well different models and techniques handle context in legal text processing, specifically focusing on the extraction of entities from criminal justice documents.

## Requirements

- Python 3.x
- Required Python packages (can be installed using `pip install -r requirements.txt`)

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/ayyubibrahimi/long-context-cj-eval.git
    cd long-context-cj-eval
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the input data:
    - Place the transcript PDFs in the `ocr/data/input/transcripts` directory.
    - Place the report PDFs in the `ocr/data/input/reports` directory.
    - Place the groundtruth CSVS in the `evaluation/data/input` directory.

4. Configure the models:
    - Open the `main.py` file.
    - Adjust the model lists for each processing type as needed:
      ```python
      allContextModels = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
      allPagesModels = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
      nerModels = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
      ```

5. Run the script:
    ```bash
    python3 main.py
    ```

The script will process the input files using all specified models and processing types. It will generate an output csv, evaluate the performance, and save the evaluation results for each model and processing type.

## Processing Types and Models

The script supports four processing types:

1. **allContext**: Stuffs the entire document into the context window. 
2. **allPages**: Processes each page of the document individually.
3. **ner**: The script uses Named Entity Recognition (NER) to calculate the number of entities on each page as a preprocessing step. It then processes the document in three separate iterations. In each iteration, it focuses on different top fractions of pages containing the most entities: first the top 1/4th, then the top 1/2, and finally the top 3/4ths of the document. For each fraction, the script identifies the pages with the highest number of entities and then iterates over each of those pages, similar to the allPages script, to extract the entities.
4. **vision**: Processes documents using Vision models. This is currently commented out. 

## Directory Structure

The script uses the following directory structure for outputs:

- LLM output: `model/{processing_type}/data/output/{model_name}/`
- Evaluation results: `evaluation/data/output/{processing_type}/{model_name}/`

Where:
- `{processing_type}` is one of: `allContext`, `allPages`, `ner`, or `vision`
- `{model_name}` is the name of the model being used

## Results

The evaluation results will be saved in the following locations:

1. Individual model results: 
   `evaluation/data/output/{processing_type}/{model_name}/results_{model_name}.csv`

## Extending the Script

To add new processing types or models:

1. Create a new `process_query` function in an appropriate module.
2. Import the new function in `main.py`.
3. Add a new model list for the processing type.
4. Add a call to `process_models()` with the new model list, processing type, and query function.


