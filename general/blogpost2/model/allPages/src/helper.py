import re
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder


PROMPT_TEMPLATE_HYDE = PromptTemplate(
    input_variables=["question"],
    template="""
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, coroners, investigators, patrolman, and criminalists, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

    Question: {question}

    Roles and Responses:""",
)


def generate_hypothetical_embeddings():
    llm = OpenAI()
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings()

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings

def clean_name(officer_name):
    return re.sub(
        r"^(Detective|Det\.?|Officer|[Dd]et\.|[Ss]gt\.|[Ll]t\.|[Cc]pt\.|[Oo]fc\.|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator|^-|\d{1}\)|\w{1}\.)\.?\s+",
        "",
        officer_name,
    )

def extract_officer_data(text):
    officer_data = []

    # Normalize the text by removing any leading/trailing whitespace and extra newlines
    normalized_text = re.sub(r'\s+', ' ', text.strip())

    # Split the text into officer sections
    officer_sections = re.split(r'(?=Officer Name:)', normalized_text)

    for section in officer_sections:
        if not section.strip():
            continue

        officer_dict = {}

        # Extract Officer Name
        name_match = re.search(r'Officer Name:\s*(.*?)(?=\s*Officer (Context|Role):|$)', section, re.DOTALL | re.IGNORECASE)
        if name_match:
            officer_dict['Officer Name'] = clean_name(name_match.group(1).strip())

        # Extract Officer Context
        context_match = re.search(r'Officer Context:\s*(.*?)(?=\s*Officer Role:|$)', section, re.DOTALL | re.IGNORECASE)
        if context_match:
            officer_dict['Officer Context'] = context_match.group(1).strip()

        # Extract Officer Role
        role_match = re.search(r'Officer Role:\s*(.*?)(?=\s*Officer Name:|$)', section, re.DOTALL | re.IGNORECASE)
        if role_match:
            officer_dict['Officer Role'] = role_match.group(1).strip()

        if officer_dict:
            officer_data.append(officer_dict)

    return officer_data


def sort_retrived_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs
