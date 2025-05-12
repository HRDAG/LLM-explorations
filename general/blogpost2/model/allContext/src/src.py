import os
import logging
import pandas as pd
from langchain_community.document_loaders.json_loader import JSONLoader
from .helper import PROMPT_TEMPLATE_HYDE, extract_officer_data
import spacy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import tiktoken
from multiprocessing import Pool
from langchain_together import ChatTogether
from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration parameters
TEMPERATURE = 0.7
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

    Officer Name: John Smith 
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: 
    Officer Context:
    Officer Role: 

    - Do not include any prefixes
    - Only derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
    """

QUERY = [
    "In the transcript, identify individuals by their names along with their specific law enforcement titles, such as officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., and coroner. Alongside each name and title, note the context of their mention. This includes the roles they played in key events, decisions they made, actions they took, their interactions with others, responsibilities in the case, and any significant outcomes or incidents they were involved in."
]


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def preprocess_document(file_path):
    logger.info(f"Processing document: {file_path}")

    loader = JSONLoader(
        file_path,
        jq_schema=".messages[]",
        content_key="page_content",
        metadata_func=metadata_func,
    )
    data = loader.load()
    logger.info(f"Data loaded from document: {file_path}")

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Calculate token count for the document
    token_count = sum(len(enc.encode(doc.page_content)) for doc in data)

    return data, token_count


def get_response_from_query(db, query, temperature, model):
    logger.info("Performing query...")
    if db is None:
        return "", [], model

    if model == "claude-3-opus-20240229":
        llm = ChatAnthropic(model_name=model, temperature=temperature)
    elif model == "claude-3-5-sonnet-20240620":
        llm = ChatAnthropic(model_name=model, temperature=temperature)
    elif model == "claude-3-haiku-20240307":
        llm = ChatAnthropic(model_name=model, temperature=temperature)
    elif model == "open-mistral-nemo":
        llm = ChatMistralAI(model_name=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model: {model}")

    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()

    # Concatenate all documents to leverage the full context window
    concatenated_docs = " ".join(doc.page_content.replace("\n", " ") for doc in db)
    page_numbers = [doc.metadata.get("page_number") for doc in db]

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60))
    def try_get_response(docs):
        try:
            return response_chain.invoke({"question": query, "docs": docs})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "client_body_buffer_size" in str(e):
                # Reduce input size by approximately 50 tokens
                tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(docs)
                if len(tokens) > 50:
                    reduced_tokens = tokens[:-50]
                    reduced_docs = tiktoken.encoding_for_model("gpt-3.5-turbo").decode(reduced_tokens)
                    logger.info(f"Reducing input size by 50 tokens. New size: {len(reduced_tokens)}")
                    return try_get_response(reduced_docs)
                else:
                    raise ValueError("Input is too small to be reduced further")
            else:
                raise

    try:
        response = try_get_response(concatenated_docs)
        logger.info(f"Response received: {response}")
        return response, page_numbers, model
    except Exception as e:
        logger.error(f"Error in get_response_from_query: {e}")
        return str(e), page_numbers, model


def process_file(args):
    file_name, input_path, output_path, file_type, model = args
    csv_output_path = os.path.join(output_path, f"{file_name}.csv")
    if os.path.exists(csv_output_path):
        logger.info(f"CSV output for {file_name} already exists. Skipping...")
        return

    file_path = os.path.join(input_path, file_name)
    output_data = []

    db, token_count = preprocess_document(file_path)
    officer_data_string, page_numbers, model = get_response_from_query(
        db, QUERY, TEMPERATURE, model
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
        default_row["token_count"] = token_count
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
            item["token_count"] = token_count
            item["file_type"] = file_type
            item["model"] = model
        output_data.extend(officer_data)

    output_df = pd.DataFrame(output_data, columns=REQUIRED_COLUMNS)
    output_df.to_csv(csv_output_path, index=False)


# def process_files(input_path, output_path, file_type, model):
#     file_args = [
#         (file_name, input_path, output_path, file_type, model)
#         for file_name in os.listdir(input_path)
#         if file_name.endswith(".json")
#     ]

#     with Pool() as pool:
#         pool.map(process_file, file_args)


# def process_query(input_path_transcripts, input_path_reports, output_path, model):
#     process_files(input_path_transcripts, output_path, "transcript", model)
#     process_files(input_path_reports, output_path, "report", model)

def process_files(input_path, output_path, file_type, model):
    for file_name in os.listdir(input_path):
        if file_name.endswith(".json"):
            process_file((file_name, input_path, output_path, file_type, model))

def process_query(input_path_transcripts, input_path_reports, output_path, model):
    process_files(input_path_transcripts, output_path, "transcript", model)
    process_files(input_path_reports, output_path, "report", model)