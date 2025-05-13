import os
import logging
import pandas as pd
import spacy
from langchain_community.document_loaders.json_loader import JSONLoader
from .helper import (
    PROMPT_TEMPLATE_HYDE,
    extract_officer_data,
    generate_hypothetical_embeddings,
    sort_retrived_documents,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores.faiss import FAISS
import tiktoken
from langchain_together import ChatTogether
import math 
from multiprocessing import Pool, cpu_count
from langchain_mistralai import ChatMistralAI

load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration parameters
TEMPERATURE = 0.7
MAX_RETRIES = 10
K = 20
MAX_PAGES = 20


REQUIRED_COLUMNS = [
    "Officer Name", "page_number", "fn", "Query", 
    "Prompt Template for Hyde", "Prompt Template for Model", 
    "Temperature", "token_count", "file_type", "model"
]
DEFAULT_VALUES = {
    "Officer Name": "",
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

law_enforcement_titles = [
    "officer",
    "sergeant",
    "lieutenant",
    "captain",
    "commander",
    "sheriff",
    "deputy",
    "detective",
    "inspector",
    "technician",
    "analyst",
    "coroner",
    "chief",
    "marshal",
    "agent",
    "superintendent",
    "commissioner",
    "trooper",
    "constable",
    "special agent",
    "patrol officer",
    "field agent",
    "investigator",
    "forensic specialist",
    "crime scene investigator",
    "public safety officer",
    "security officer",
    "patrolman",
    "watch commander",
    "undercover officer",
    "intelligence officer",
    "tactical officer",
    "bomb technician",
    "K9 handler",
    "SWAT team member",
    "emergency dispatcher",
    "corrections officer",
    "probation officer",
    "parole officer",
    "bailiff",
    "court officer",
    "wildlife officer",
    "park ranger",
    "border patrol agent",
    "immigration officer",
    "customs officer",
    "air marshal",
    "naval investigator",
    "military police",
    "forensic scientist",
    "forensic analyst",
    "crime lab technician",
    "forensic technician",
    "laboratory analyst",
    "DNA analyst",
    "toxicologist",
    "serologist",
    "ballistics expert",
    "fingerprint analyst",
    "forensic chemist",
    "forensic biologist",
    "trace evidence analyst",
    "forensic pathologist",
    "forensic odontologist",
    "forensic entomologist",
    "forensic anthropologist",
    "digital forensic analyst",
    "forensic engineer",
    "crime scene examiner",
    "evidence technician",
    "latent print examiner",
    "forensic psychologist",
    "forensic document examiner",
    "forensic photography specialist",
    "det.",
    "sgt.",
    "cpt.",
    "lt.",
    "p.o.",
    "dpty.",
    "ofc.",
]


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def preprocess_document(file_path, fraction=1.0):
    logger.info(f"Processing document: {file_path}")
    loader = JSONLoader(
        file_path,
        jq_schema=".messages[]",
        content_key="page_content",
        metadata_func=metadata_func,
    )
    text = loader.load()
    logger.info(f"Text loaded from document: {file_path}")

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    page_entities = []
    token_count = 0

    for page in text:
        doc = nlp(page.page_content)
        entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and any(title in page.page_content[max(0, ent.start_char - 100) : ent.end_char + 100].lower() for title in law_enforcement_titles)]
        page_entities.append((page, len(entities)))
        token_count += len(enc.encode(page.page_content))

    # Sort pages by number of entities, from most to least
    page_entities.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate MAX_PAGES based on fraction
    total_pages = len(page_entities)
    MAX_PAGES = math.ceil(total_pages * fraction)
    
    # Select the top MAX_PAGES with the most entities
    selected_pages = [page for page, _ in page_entities[:MAX_PAGES]]

    return selected_pages, token_count, MAX_PAGES

def get_response_from_query(docs, query, temperature, model):
    logger.info("Performing query...")
    if not docs:
        return "", []
    
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
    
    
    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    responses = []
    page_numbers = []

    for doc in docs:
        page_content = doc.page_content.replace("\n", " ")
        page_number = doc.metadata.get("page_number")
        if page_content:
            response = response_chain.invoke({"question": query, "docs": page_content})
            print(response)
            responses.append(response)
        else:
            responses.append("")
        if page_number is not None:
            page_numbers.append(page_number)
    concatenated_responses = "\n\n".join(responses)
    return concatenated_responses, page_numbers, model


def process_file(args):
    file_name, input_path, output_path, file_type, model, fraction = args
    csv_output_path = os.path.join(output_path, f"{file_name}.csv")
    if os.path.exists(csv_output_path):
        logger.info(f"CSV output for {file_name} already exists. Skipping...")
        return

    file_path = os.path.join(input_path, file_name)
    output_data = []

    docs, token_count, MAX_PAGES = preprocess_document(file_path, fraction)
    
    for query in QUERY:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                officer_data_string, page_numbers, _ = get_response_from_query(
                    docs, query, TEMPERATURE, model
                )
                break
            except ValueError as e:
                retries += 1
                logger.warning(f"Retry {retries} for query {query} due to error: {e}")
                if retries == MAX_RETRIES:
                    logger.error(f"Max retries reached for query {query}. Skipping...")
                    return

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

def process_files(input_path, output_path, file_type, model, fraction=1.0):
    file_list = [f for f in os.listdir(input_path) if f.endswith(".json")]
    args_list = [(file_name, input_path, output_path, file_type, model, fraction) for file_name in file_list]
    
    # Use half of the available CPU cores, but at least 1
    num_processes = max(1, cpu_count() // 4)
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, args_list)

def process_query(input_path_transcripts, input_path_reports, output_path, model, fraction=1.0):
    process_files(input_path_transcripts, output_path, "transcript", model, fraction)
    process_files(input_path_reports, output_path, "report", model, fraction)
