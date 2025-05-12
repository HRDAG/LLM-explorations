import os
import logging
import pandas as pd
from model.allContext.src.src import process_query as process_query_allContext
from model.allPages.src.src import process_query as process_query_allPages
from model.ner.src.src import process_query as process_query_ner_original
from model.Vision.src.src import process_query as process_query_vision
from ocr.src.ocr import ocr_process
from evaluation.src.evaluation import (
    preprocess_data,
    compute_jaro_winkler_metrics,
    title_patterns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_query_ner(input_path_transcripts, input_path_reports, output_path, model, fraction=1.0):
    return process_query_ner_original(input_path_transcripts, input_path_reports, output_path, model, fraction)

def ocr_files_exist(output_path):
    return os.path.exists(output_path) and len([f for f in os.listdir(output_path) if f.endswith('.json')]) > 0

def main(
    input_path_transcripts,
    input_path_reports,
    output_path_transcripts,
    output_path_reports,
    llm_output_directory,
    evaluation_output_directory,
    groundtruth_directory,
    model,
    process_function,
    process_type
):
    if process_type != "vision":
        # Check if OCR files already exist
        if ocr_files_exist(output_path_transcripts) and ocr_files_exist(output_path_reports):
            logging.info("OCR files already exist. Skipping OCR process.")
        else:
            # Process the PDF files using OCR
            logging.info("Starting OCR process...")
            ocr_process(input_path_transcripts, input_path_reports, output_path_transcripts, output_path_reports)

        # Process the OCR output files and generate LLM output
        logging.info(f"Starting LLM processing with model: {model}...")
        process_function(
            output_path_transcripts,
            output_path_reports,
            llm_output_directory,
            model
        )
    else:
        # For Vision model, process the original PDF files
        logging.info(f"Starting Vision processing with model: {model}...")
        process_function(
            input_path_transcripts,
            input_path_reports,
            llm_output_directory,
            model
        )

    # Evaluate the LLM output
    logging.info("Starting evaluation...")
    results = []
    for file_name in os.listdir(llm_output_directory):
        if file_name.endswith(".csv"):
            llm_output_path = os.path.join(llm_output_directory, file_name)
            groundtruth_file_name = f"{os.path.splitext(file_name)[0]}-groundtruth.csv"
            groundtruth_path = os.path.join(
                groundtruth_directory,
                groundtruth_file_name
            )
            if os.path.exists(groundtruth_path):
                logging.info(f"Processing file: {file_name}")
                llm_output_df, groundtruth_df = preprocess_data(
                    llm_output_path,
                    groundtruth_path,
                    title_patterns,
                )
                file_results = compute_jaro_winkler_metrics(
                    llm_output_df,
                    groundtruth_df
                )
                file_results["file_name"] = file_name
                file_results["model"] = model
                results.append(file_results)
            else:
                logging.warning(f"Groundtruth file not found for {file_name}")

    logging.info("Concatenating results...")
    results_df = pd.concat(results, ignore_index=True)

    # Save individual model results
    individual_output_path = os.path.join(evaluation_output_directory, f"results_{model}.csv")
    os.makedirs(os.path.dirname(individual_output_path), exist_ok=True)
    results_df.to_csv(individual_output_path, index=False)
    logging.info(f"Results for model {model} saved to: {individual_output_path}")

    return results_df

def process_models(models, process_type, process_function, fractions=None):
    all_results = []
    for model in models:
        logging.info(f"Processing with model: {model}")
        
        if process_type == "ner" and fractions:
            for fraction in fractions:
                fraction_str = f"{int(fraction * 100)}percent"
                llm_output_directory = os.path.join(f"model/{process_type}/data/output/{model}_{fraction_str}")
                evaluation_output_directory = os.path.join(f"evaluation/data/output/{process_type}/{model}_{fraction_str}")
                os.makedirs(llm_output_directory, exist_ok=True)
                os.makedirs(evaluation_output_directory, exist_ok=True)

                model_results = main(
                    input_path_transcripts,
                    input_path_reports,
                    output_path_transcripts,
                    output_path_reports,
                    llm_output_directory,
                    evaluation_output_directory,
                    groundtruth_directory,
                    model,
                    lambda *args: process_function(*args, fraction=fraction),
                    process_type
                )
                model_results['fraction'] = fraction
                all_results.append(model_results)
        else:
            llm_output_directory = os.path.join(f"model/{process_type}/data/output/{model}")
            evaluation_output_directory = os.path.join(f"evaluation/data/output/{process_type}/{model}")
            os.makedirs(llm_output_directory, exist_ok=True)
            os.makedirs(evaluation_output_directory, exist_ok=True)

            model_results = main(
                input_path_transcripts,
                input_path_reports,
                output_path_transcripts,
                output_path_reports,
                llm_output_directory,
                evaluation_output_directory,
                groundtruth_directory,
                model,
                process_function,
                process_type
            )
            all_results.append(model_results)

    # Combine results from all models
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_results_directory = f"evaluation/data/output/{process_type}"
    os.makedirs(combined_results_directory, exist_ok=True)
    combined_output_path = os.path.join(combined_results_directory, "combined_results.csv")
    combined_results.to_csv(combined_output_path, index=False)
    logging.info(f"Combined results from all models saved to: {combined_output_path}")

if __name__ == "__main__":
    input_path_transcripts = r"ocr/data/input/transcripts"
    input_path_reports = r"ocr/data/input/reports"
    output_path_transcripts = r"ocr/data/output/transcripts"
    output_path_reports = r"ocr/data/output/reports"
    
    groundtruth_directory = "evaluation/data/input"

    # Define models and processing types
    allContextModels = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229", "open-mistral-nemo"]
    # allPagesModels = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1", "open-mistral-nemo"]
    allPagesModels = ["mistralai/Mixtral-8x7B-Instruct-v0.1"]

    nerModels = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620",  "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1", "open-mistral-nemo"]
    visionModels = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]

    # # Process allContext models
    # process_models(allContextModels, "allContext", process_query_allContext)

    # Process allPages models
    process_models(allPagesModels, "allPages", process_query_allPages)

    # # Process ner models with different fractions
    # ner_fractions = [0.25, 0.5, 0.75]
    # process_models(nerModels, "ner", process_query_ner, fractions=ner_fractions)

    # # Process vision models
    # process_models(visionModels, "vision", process_query_vision)