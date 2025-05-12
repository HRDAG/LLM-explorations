import os
import logging
import pandas as pd
from langchain_community.document_loaders.json_loader import JSONLoader
from .helper import PROMPT_TEMPLATE_HYDE, extract_officer_data
import spacy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import tiktoken
from multiprocessing import Pool, cpu_count
from langchain_together import ChatTogether
from langchain_mistralai import ChatMistralAI
import re
from collections import namedtuple
import json 
from sentence_transformers import SentenceTransformer
import numpy as np


Doc = namedtuple("Doc", ["page_content", "metadata"])

load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration parameters
TEMPERATURE = 1
ITERATION_TIMES = 1
MAX_RETRIES = 10

REQUIRED_COLUMNS = [
    "Officer Name", "Officer Role", "Officer Context", "page_number", "fn", "Query", 
    "Prompt Template for Hyde", "Prompt Template for Model", 
    "Temperature", "token_count", "file_type", "model"
]

DEFAULT_VALUES = {
    "Officer Name": "",
    "Officer Role": "",
    "Officer Context": "",
    "page_number": [],
    "fn": "",
    "Query": "",
    "Prompt Template for Hyde": "",
    "Prompt Template for Model": "",
    "Temperature": 0.0,
    "token_count": 0,
    "file_type": "",
    "model": ""
}



def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def format_content(content):
    # Remove extra whitespace and empty lines
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = re.sub(r' +', ' ', content)
    
    # Split content into lines
    lines = content.split('\n')
    
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Check if the line is a header (e.g., all caps, or ends with a colon)
            if line.isupper() or line.endswith(':'):
                formatted_lines.append(f"\n{line}\n")
            else:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def word_count(text):
    return len(text.split())

def load_and_split(file_path):
    logger.info(f"Processing document: {file_path}")

    with open(file_path, "r") as file:
        file_content = file.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            data = {}

    logger.info(f"Keys in parsed JSON data: {data.keys()}")

    if "messages" in data and data["messages"]:
        docs = []
        original_page_number = 1
        for message in data["messages"]:
            page_content = message.get("page_content", "")
            if word_count(page_content) >= 50:
                formatted_content = format_content(page_content)
                doc = Doc(
                    page_content=formatted_content,
                    metadata={"seq_num": original_page_number},
                )
            else:
                doc = Doc(
                    page_content="No data to be processed on this page",
                    metadata={"seq_num": original_page_number},
                )
            docs.append(doc)
            original_page_number += 1

        logger.info(f"Data loaded and formatted from document: {file_path}")
        return docs
    else:
        logger.warning(f"No valid data found in document: {file_path}")
        return []
    

QUERY = [
    "In the transcript, identify individuals by their names along with their specific law enforcement titles, such as officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., and coroner. Alongside each name and title, note the context of their mention. This includes the roles they played in key events, decisions they made, actions they took, their interactions with others, responsibilities in the case, and any significant outcomes or incidents they were involved in."
]



template = """
     As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners.
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, please make this clear in your response.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst
       Please prefix this information with "Officer Role: "
       For example, "Officer Role: Lead Detective"

    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: Officer X
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: Person Y
    Officer Context: Was on scene
    Officer Role: Witness

    - Do not include any prefixes
    - Only derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
    """

validation_template = """
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.

    Your instructions will be provided to you after you read these documents: {docs}

    Query: {question}

    # Instructions

    The response will contain:

    1) The name of a unique law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners.
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, please make this clear in your response.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst
       Please prefix this information with "Officer Role: "
       For example, "Officer Role: Lead Detective"

    
    Your findings must be based on informtion contained within the document provided to you. The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: Officer X
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: Person Y
    Officer Context: Was on scene
    Officer Role: Witness

    - Do not include any prefixes
    - Only derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
 
 # Warning
    If there are no identified law enforcement officers named in the documents, return "No law enforcement officers identified".
"""

def regex_parse_officers(text):
    officer_pattern = r'\b(?:P\.O\.|Police Officer|Officer|Sergeant|Sgt\.|Lieutenant|Lt\.|Captain|Cpt\.|Detective|Det\.|Inspector|Insp\.|Chief|Sheriff|Deputy|Dep\.|Trooper|Constable)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)'
    matches = re.finditer(officer_pattern, text, re.IGNORECASE)
    
    officers = []
    for match in matches:
        name = match.group(1)  # Get the officer's name
        officers.append({
            "Officer Name": name,
            "Officer Role": "",
            "Officer Context": ""
        })
    
    return officers

def embedding_parse_officers(text, model='all-MiniLM-L6-v2'):
    # Load the embedding model
    embed_model = SentenceTransformer(model)
    
    # Define officer titles
    officer_titles = [
        "Police Officer", "Officer", "Sergeant", "Lieutenant", "Captain",
        "Detective", "Inspector", "Chief", "Sheriff", "Deputy", "Trooper", "Constable",
        "P.O.", "Sgt.", "Lt.", "Cpt.", "Det.", "Insp.", "Dep.", "Ofc."
    ]
    
    # Create embeddings for officer titles
    title_embeddings = embed_model.encode(officer_titles)
    
    # Split the text into sentences
    sentences = text.split('.')
    
    officers = []
    for sentence in sentences:
        # Create embedding for the sentence
        sentence_embedding = embed_model.encode([sentence])[0]
        
        # Compare sentence embedding with officer title embeddings
        similarities = np.dot(title_embeddings, sentence_embedding)
        
        # If there's a high similarity, extract potential officer name
        if np.max(similarities) > 0.5:  # Threshold can be adjusted
            words = sentence.split()
            for i, word in enumerate(words):
                if word[0].isupper() and i+1 < len(words) and words[i+1][0].isupper():
                    officers.append({
                        "Officer Name": f"{word} {words[i+1]}",
                        "Officer Role": "",
                        "Officer Context": ""
                    })
                elif word[0].isupper() and len(word) > 1:
                    officers.append({
                        "Officer Name": word,
                        "Officer Role": "",
                        "Officer Context": ""
                    })
    
    return officers

def format_officer_results(officers, title):
    formatted = f"{title}:\n"
    for officer in officers:
        formatted += f"Officer Name: {officer['Officer Name']}\n"
        formatted += f"Officer Role: {officer['Officer Role']}\n"
        formatted += f"Officer Context: {officer['Officer Context']}\n\n"
    return formatted.strip()


def get_response_from_query(db, query, temperature, model, pages_per_chunk=2):
    # 2 gets us 89% recall
    logger.info("Performing query...")
    if db is None:
        return "", []
    docs = db

    if model == "claude-3-haiku-20240307":
        llm = ChatAnthropic(model_name=model, temperature=temperature)
    elif model == "claude-3-5-sonnet-20240620":
        llm = ChatAnthropic(model_name=model, temperature=temperature)
    elif model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
        llm = ChatTogether(model_name=model, temperature=temperature)
    elif model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = ChatTogether(model_name=model, temperature=temperature)
    elif model == "meta-llama/Llama-3-8b-chat-hf":
        llm = ChatTogether(model_name=model, temperature=temperature)
    elif model == "meta-llama/Llama-3-70b-chat-hf":
        llm = ChatTogether(model_name=model, temperature=temperature)
    elif model == "open-mistral-nemo":
        llm = ChatMistralAI(model_name=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Initial extraction chain
    initial_prompt = ChatPromptTemplate.from_template(template)
    initial_chain = initial_prompt | llm | StrOutputParser()

    # Validation chain
    validation_prompt = ChatPromptTemplate.from_template(validation_template)
    validation_chain = validation_prompt | llm | StrOutputParser()

    responses = []
    page_numbers = []
    
    # Process documents in chunks
    for i in range(0, len(docs), pages_per_chunk):
        chunk = docs[i:i+pages_per_chunk]
        combined_content = ""
        chunk_page_numbers = []
        
        for doc in chunk:
            
            page_content = doc.page_content
            print(page_content)
            page_number = doc.metadata.get("page_number")
            combined_content += f"Page {page_number}:\n{page_content}\n\n"
            chunk_page_numbers.append(str(page_number))
        
        combined_content = combined_content.strip()
        combined_page_numbers = "-".join(chunk_page_numbers)

        if combined_content:
            regex_officers = regex_parse_officers(combined_content)
            regex_results = format_officer_results(regex_officers, "Regex-identified Officers")
            print(f"REGEX OFFICERS {regex_results}")
            responses.append(regex_results)

            embedding_officers = embedding_parse_officers(combined_content)
            embedding_results = format_officer_results(embedding_officers, "Embedding-identified Officers")
            print(f"EMBEDDING OFFICERS {embedding_results}")
            responses.append(embedding_results)

            # First model call: Initial extraction
            initial_extraction = initial_chain.invoke({"question": query, "docs": combined_content})
            
            # Append initial extraction
            responses.append(initial_extraction)
            
            # Second model call: Validation
            validated_extraction = validation_chain.invoke({
                "question": query, 
                "docs": combined_content,
            })
            
            # Append validated extraction
            responses.append(validated_extraction)
        else:
            responses.append("")
        
        page_numbers.append(combined_page_numbers)

    concatenated_responses = "\n\n".join(responses)
    print(concatenated_responses)
    return concatenated_responses, page_numbers, model


def process_file(args):
    file_name, input_path, output_path, file_type, model = args
    csv_output_path = os.path.join(output_path, f"{file_name}.csv")
    if os.path.exists(csv_output_path):
        logger.info(f"CSV output for {file_name} already exists. Skipping...")
        return

    file_path = os.path.join(input_path, file_name)
    output_data = []

    db = load_and_split(file_path)
    officer_data_string, page_numbers, model = get_response_from_query(
        db, QUERY, TEMPERATURE, model, pages_per_chunk=2
    )
    officer_data = extract_officer_data(officer_data_string)

    
    if not officer_data:
        # If no officers found, create a row with default values
        default_row = {column: DEFAULT_VALUES[column] for column in REQUIRED_COLUMNS}
        default_row["page_number"] = page_numbers
        default_row["fn"] = file_name
        default_row["Query"] = QUERY
        default_row["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
        default_row["Prompt Template for Model"] = template
        default_row["Temperature"] = TEMPERATURE
        default_row["file_type"] = file_type
        default_row["model"] = model
        output_data.append(default_row)
    else:
        for item in officer_data:
            item["page_number"] = page_numbers
            item["fn"] = file_name
            item["Query"] = QUERY
            item["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
            item["Prompt Template for Model"] = template
            item["Temperature"] = TEMPERATURE
            item["file_type"] = file_type
            item["model"] = model
        output_data.extend(officer_data)

    output_df = pd.DataFrame(output_data, columns=REQUIRED_COLUMNS)
    output_df.to_csv(csv_output_path, index=False)


def process_files(input_path, output_path, file_type, model):
    file_list = [f for f in os.listdir(input_path) if f.endswith(".json")]
    args_list = [(file_name, input_path, output_path, file_type, model) for file_name in file_list]
    
    # Use half of the available CPU cores, but at least 1
    num_processes = max(1, cpu_count() // 4)
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, args_list)

def process_query(input_path_transcripts, input_path_reports, output_path, model):
    process_files(input_path_transcripts, output_path, "transcript", model)
    process_files(input_path_reports, output_path, "report", model)
