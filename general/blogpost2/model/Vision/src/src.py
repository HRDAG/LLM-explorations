import os
import logging
import pandas as pd
import base64
from io import BytesIO
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage
import concurrent.futures

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Officer Name", "Officer Context", "Officer Role", "page_number", "fn", 
    "Temperature", "token_count", "file_type", "model"
]

DEFAULT_VALUES = {
    "Officer Name": "",
    "Officer Context": "",
    "Officer Role": "",
    "page_number": 0,
    "fn": "",
    "Temperature": 0.7,
    "token_count": 0,
    "file_type": "pdf",
    "model": "claude-3-haiku-20240307"
}

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def classify_page(chat, base64_image, page_number):
    prompt = """
    As an AI assistant, your task is to identify and provide information about law enforcement personnel mentioned in the image. Please follow these guidelines:

    1. Identify individuals by their names and specific law enforcement titles (e.g., officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., coroner).

    2. For each identified individual, provide:
       - Their name (prefixed with "Officer Name: ")

    3. Format the response as follows:

    Officer Name: [Name]

    Important notes:
    - Do not use numbered or lettered prefixes in your response.
    - Only include information directly stated in the police report image.
    - If an individual's context is unclear, note this fact.
    - Do not extract or include information about victims or witnesses.
    - If no law enforcement personnel are identified in the image, state this fact clearly.

    Please analyze the provided image and respond according to these guidelines.
    """

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ]
                )
            ]
        )
        return msg.content.strip(), page_number
    except Exception as e:
        return f"Error classifying page: {str(e)}", page_number

# def classify_page(chat, base64_image, page_number):
#     prompt = """
#     As an AI assistant, your task is to identify and provide information about law enforcement personnel mentioned in the image. Please follow these guidelines:

#     1. Identify individuals by their names and specific law enforcement titles (e.g., officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., coroner).

#     2. For each identified individual, provide:
#        - Their name (prefixed with "Officer Name: ")
#        - The context of their mention (prefixed with "Officer Context: ")
#        - Their role, if discernible (prefixed with "Officer Role: ")

#     3. Format the response as follows:

#     Officer Name: [Name]
#     Officer Context: [Detailed description of their mention, including key events, decisions, actions, interactions, responsibilities, and any significant outcomes or incidents they were involved in]
#     Officer Role: [Specific role, e.g., Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst]

#     Important notes:
#     - Do not use numbered or lettered prefixes in your response.
#     - Only include information directly stated in the police report image.
#     - If an individual's context is unclear, note this fact.
#     - Do not extract or include information about victims or witnesses.
#     - If no law enforcement personnel are identified in the image, state this fact clearly.

#     Please analyze the provided image and respond according to these guidelines.
#     """

#     try:
#         msg = chat.invoke(
#             [
#                 HumanMessage(
#                     content=[
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/png;base64,{base64_image}"},
#                         },
#                     ]
#                 )
#             ]
#         )
#         return msg.content.strip(), page_number
#     except Exception as e:
#         return f"Error classifying page: {str(e)}", page_number

def extract_officer_data(response):
    officers = []
    current_officer = {}
    for line in response.split('\n'):
        if line.startswith("Officer Name:"):
            if current_officer:
                officers.append(current_officer)
                current_officer = {}
            current_officer["Officer Name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Officer Context:"):
            current_officer["Officer Context"] = line.split(":", 1)[1].strip()
        elif line.startswith("Officer Role:"):
            current_officer["Officer Role"] = line.split(":", 1)[1].strip()
    if current_officer:
        officers.append(current_officer)
    return officers

def process_page(args):
    chat, pdf_path, page_number = args
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    image = images[0]
    base64_image = encode_image(image)
    return classify_page(chat, base64_image, page_number)

def process_pdf(pdf_path, output_csv):
    chat = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
    
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_page = {executor.submit(process_page, (chat, pdf_path, i+1)): i+1 for i in range(page_count)}
            
            all_officer_data = []
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_number = future_to_page[future]
                try:
                    classification, _ = future.result()
                    logger.info(f"Processed page {page_number} of {pdf_path}")
                    
                    officer_data = extract_officer_data(classification)
                    for officer in officer_data:
                        officer["page_number"] = page_number
                        officer["fn"] = os.path.basename(pdf_path)
                        officer["Temperature"] = 0.7
                        officer["token_count"] = len(classification.split())
                        officer["file_type"] = "pdf"
                        officer["model"] = "claude-3-haiku-20240307"
                        all_officer_data.append(officer)
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_number} of {pdf_path}: {str(e)}")
    
    if not all_officer_data:
        default_row = DEFAULT_VALUES.copy()
        default_row["fn"] = os.path.basename(pdf_path)
        all_officer_data.append(default_row)
    
    df = pd.DataFrame(all_officer_data)
    df = df.reindex(columns=REQUIRED_COLUMNS)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV output to {output_csv}")

def process_query(input_path_transcripts, input_path_reports, output_path, model):
    if model != "claude-3-haiku-20240307":
        logger.warning(f"Model {model} is not supported for vision processing. Using claude-3-haiku-20240307.")
    
    for input_path in [input_path_transcripts, input_path_reports]:
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_path, filename)
                output_csv = os.path.join(output_path, os.path.splitext(filename)[0] + '.json.csv')
                
                if os.path.exists(output_csv):
                    logger.info(f"Output file {output_csv} already exists. Skipping processing of {filename}")
                    continue
                
                process_pdf(pdf_path, output_csv)
                logger.info(f"Processed {filename} and saved results to CSV")

    logger.info("All PDFs processed successfully.")