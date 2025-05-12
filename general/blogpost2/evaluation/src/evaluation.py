import pandas as pd
import os
import re
import numpy as np
import logging
from jellyfish import jaro_winkler_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Consolidate titles to handle variations with and without periods
title_patterns = [
    r"\bLt\.?\b",
    r"\bLes\.?\b",
    r"\bDet\.?\b",
    r"\bSat\.?\b",
    r"\bCet\.?\b",
    r"\bDetective\b",
    r"\bDetectives\b",
    r"\bOfficer\b",
    r"\bOfficers\b",
    r"\bSgt\.?\b",
    r"\bLieutenant\b",
    r"\bSergeant\b",
    r"\bCaptain\b",
    r"\bCorporal\b",
    r"\bDeputy\b",
    r"\bOfc\.?\b",
    r"\b\(?Technician\)?\b",
    r"\b\(?Criminalist\)?\b",
    r"\b\(?Photographer\)?\b",
    r"\bPolice Officer\b",
    r"\bCrime Lab Technician\b",
    r"\bUNIT \d+\b",
    r"\bMOUNTED OFFICERS\b",
    r"Officer Context:",
    r"\bP/O\b",
    r"\bP\.O\.?\b",
    r"\bPolice\b",
    r"\bDr\.?\b",
    r"\bInvestigator\b",
    r"\bCoroners\b",
    r"\bCrime Lab\b",
    r"\bEmergency Medical Technicians\b",
    r"#(\w+)\b",
]

# Compile regex patterns for efficiency
compiled_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in title_patterns]

def split_consolidated_names(name):
    if isinstance(name, str):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    else:
        return ""

def clean_llm_output_officer_names(df, compiled_patterns):
    logging.info("Cleaning LLM output officer names...")
    
    # Ensure all values in "Officer Name" column are strings
    df["Officer Name"] = df["Officer Name"].astype(str).fillna("").str.lower()
    
    for pattern in compiled_patterns:
        df["Officer Name"] = df["Officer Name"].apply(lambda x: pattern.sub("", x).strip())
    
    # Apply split_consolidated_names and additional cleaning for periods and whitespace
    df["Officer Name"] = df["Officer Name"].apply(split_consolidated_names)
    df["Officer Name"] = df["Officer Name"].str.replace(r'^[\s.]+|[\s.]+$', '', regex=True)
    
    return df

def clean_groundtruth_officer_names(df, compiled_patterns):
    logging.info("Cleaning groundtruth officer names...")
    df["Officer Names to Match"] = df["Officer Names to Match"].astype(str).fillna("").str.lower()
    
    for pattern in compiled_patterns:
        df["Officer Names to Match"] = df["Officer Names to Match"].apply(lambda x: pattern.sub("", x).strip())
    
    # Apply split_consolidated_names and additional cleaning for periods and whitespace
    df["Officer Names to Match"] = df["Officer Names to Match"].apply(split_consolidated_names)
    df["Officer Names to Match"] = df["Officer Names to Match"].str.replace(r'^[\s.]+|[\s.]+$', '', regex=True)
    
    return df

def preprocess_data(llm_output_path, groundtruth_path, title_patterns):
    logging.info(
        f"Preprocessing data for LLM output: {llm_output_path} and groundtruth: {groundtruth_path}"
    )
    try:
        llm_output_df = pd.read_csv(llm_output_path)
        if llm_output_df.empty:
            logging.warning(f"The LLM output file {llm_output_path} is empty. Creating default DataFrame.")
            llm_output_df = pd.DataFrame(columns=["Officer Name"])
    except pd.errors.EmptyDataError:
        logging.warning(f"The LLM output file {llm_output_path} is empty. Creating default DataFrame.")
        llm_output_df = pd.DataFrame(columns=["Officer Name"])

    try:
        groundtruth_df = pd.read_csv(groundtruth_path)
        if groundtruth_df.empty:
            logging.warning(f"The groundtruth file {groundtruth_path} is empty. Creating default DataFrame.")
            groundtruth_df = pd.DataFrame(columns=["Officer Names to Match"])
    except pd.errors.EmptyDataError:
        logging.warning(f"The groundtruth file {groundtruth_path} is empty. Creating default DataFrame.")
        groundtruth_df = pd.DataFrame(columns=["Officer Names to Match"])

    llm_output_df = clean_llm_output_officer_names(llm_output_df, compiled_patterns)
    groundtruth_df = clean_groundtruth_officer_names(groundtruth_df, compiled_patterns)


    return llm_output_df, groundtruth_df

def compute_jaro_winkler_metrics(llm_output_df, groundtruth_df, threshold=0.8):
    logging.info("Computing Jaro-Winkler metrics...")
    results = []
    matched_names = set()
    false_positive_names = set()

    # Get unique officer names from LLM output and groundtruth dataframes
    llm_output_names = llm_output_df["Officer Name"].unique()
    groundtruth_names = groundtruth_df["Officer Names to Match"].unique()

    # Count the number of unique entities in the groundtruth dataframe
    unique_entity_count = len(groundtruth_df)

    # Iterate over each officer name in the LLM output
    for name in llm_output_names:
        # Calculate Jaro-Winkler similarity between the current name and each groundtruth name
        similarities = [
            (gt_name, jaro_winkler_similarity(name, gt_name))
            for gt_name in groundtruth_names
        ]
        # Find the best match based on the highest similarity score
        best_match = max(similarities, key=lambda x: x[1])

        # If the best match similarity is above the threshold, consider it a match
        if best_match[1] > threshold:
            matched_names.add(best_match[0])
        else:
            false_positive_names.add(name)

    # Calculate true positives (matched names that exist in groundtruth)
    true_positives = len(matched_names.intersection(set(groundtruth_names.tolist())))
    # Calculate false positives (names in LLM output that don't exist in groundtruth)
    false_positives = len(false_positive_names)

    # Calculate precision (true positives / (true positives + false positives))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    # Calculate recall (true positives / total groundtruth names)
    recall = (
        true_positives / len(groundtruth_names) if len(groundtruth_names) > 0 else 0
    )
    # Calculate F1 score (harmonic mean of precision and recall)
    f1_score = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    # Set beta value for F-beta score calculation
    beta = 2
    # Calculate F-beta score (weighted average of precision and recall)
    f_beta_score = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )

    # Get additional metadata from the LLM output dataframe
    token_count = (
        llm_output_df["token_count"].iloc[0]
        if not llm_output_df["token_count"].empty
        else None
    )
    filename = llm_output_df["fn"].iloc[0] if not llm_output_df["fn"].empty else None
    filetype = (
        llm_output_df["file_type"].iloc[0]
        if not llm_output_df["file_type"].empty
        else None
    )
    model = llm_output_df["model"].iloc[0] if not llm_output_df["model"].empty else None

    # Calculate unmatched names (names in groundtruth but not matched)
    unmatched_names = set(groundtruth_names.tolist()) - matched_names

    # Prepare the results dictionary
    results.append(
        {
            "matched_count": len(matched_names),
            "total_ground_truth": len(groundtruth_names),
            "percentage_matched": len(matched_names) / len(groundtruth_names) * 100,
            "matched_names": matched_names,
            "unmatched_names": unmatched_names,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f_beta_score": f_beta_score,
            "token_count": token_count,
            "filename": filename,
            "filetype": filetype,
            "model": model,
            "unique_entity_count": unique_entity_count,
        }
    )

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df
