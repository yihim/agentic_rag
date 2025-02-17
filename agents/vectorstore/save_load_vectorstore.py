from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from agents.constants.models import (
    EMBEDDING_MODEL,
    FALCON3_10B_INSTRUCT,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
    TABLE_ORGANIZER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from agents.utils.models import load_llm_and_tokenizer, set_random_seed
from typing import Tuple
import re
import requests
from tqdm import tqdm
import torch
import shutil
import string
from time import perf_counter
from llm_preprocess_data import process_data
from dotenv import load_dotenv
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
set_random_seed(42)


def clean_and_organize_external_data(
    llm_and_tokenizer: Tuple[
        AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained
    ],
    file_name_without_ext: str,
):

    md_dir = "../../data_extraction/tmp/md/"

    def clean_text_to_json(text):
        # Split the text into lines
        lines = text.split("\n")

        # Initialize variables
        json_output = []
        current_header = None
        current_content = []
        non_header_content = []

        def clean_header(header):
            # Remove the # symbol and any leading/trailing whitespace
            header = header.replace("#", "").strip()
            # Remove any numbers and special characters, keeping only letters and spaces
            header = re.sub(r"[^a-zA-Z\s]", "", header)
            # Remove extra spaces and strip
            header = " ".join(header.split())
            return header

        def process_current_group():
            if current_header:
                # Check if content ends with punctuation or contains .jpg
                has_valid_content = any(
                    line.strip()[-1] in string.punctuation or ".jpg" in line
                    for line in current_content
                )

                # Only add the group if there's content and it meets our criteria
                if current_content and has_valid_content:
                    json_output.append(
                        {
                            "header": clean_header(current_header),
                            "content": "\n".join(current_content),
                        }
                    )
                elif not current_content:
                    # If header has no content, add it with empty content
                    json_output.append(
                        {"header": clean_header(current_header), "content": ""}
                    )

        # Process each line
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Check if line is a header
            if line.startswith("#"):
                # Process previous group before starting new one
                process_current_group()
                # Start new group
                current_header = line
                current_content = []
            else:
                if current_header:
                    # Add line to current header's content
                    current_content.append(line)
                else:
                    # Add line to non-header content
                    non_header_content.append(line)

        # Process the last group
        process_current_group()

        # Add non-header content if it exists and has valid content
        if non_header_content and any(
            line.strip()[-1] in string.punctuation or ".jpg" in line
            for line in non_header_content
        ):
            json_output.append(
                {"header": "None", "content": "\n".join(non_header_content)}
            )

        # Convert to JSON with ensure_ascii=False to preserve Unicode characters
        return json_output

    def clean_references(text):
        # Pattern to match:
        # 1. Period followed by numbers and commas
        # 2. Comma followed by numbers
        # Both patterns should be followed by a space or end of string
        pattern = r"[.,][0-9]+(?:,\s*[0-9]+)*(?=\s|$)"

        # Function to process each match
        def replace_match(match):
            # If match starts with period, return period
            if match.group().startswith("."):
                return "."
            # If match starts with comma, return comma
            elif match.group().startswith(","):
                return ","

        # Replace the pattern
        cleaned_text = re.sub(pattern, replace_match, text)

        return cleaned_text.strip()

    with open(
        os.path.join(md_dir, file_name_without_ext + ".md"), "r", encoding="utf-8"
    ) as f:
        extracted_data = f.read()

    print("Cleaning extracted data...")

    cleaned_references = clean_references(extracted_data)

    organized_data = clean_text_to_json(cleaned_references)

    print("Cleaning completed.")

    text_data = []
    table_data = []
    organized_all_data = []

    # Organize data
    for index, data in enumerate(organized_data):
        header = data["header"]
        content = data["content"].split("\n")

        text = []

        for item in content:
            if item:
                if item.startswith("<html><body><table>"):
                    table_data.append({"header": header, "table": item})
                elif item.startswith("![]"):
                    pass
                else:
                    text.append(item)

        if text:
            text_data.append({"header": header, "text": text})

    # Process table data
    if table_data:
        table_data = process_data(
            data=table_data,
            data_type="table",
            llm_and_tokenizer=llm_and_tokenizer,
            system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
            max_tokens=TABLE_ORGANIZER_LLM_MAX_TOKENS,
            batch_size=8,
        )

        # print(table_data)

        organized_all_data.extend(table_data)

    # Process text data
    if text_data:
        text_data = process_data(
            data=text_data,
            data_type="text",
            llm_and_tokenizer=llm_and_tokenizer,
            system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
            max_tokens=AGENTIC_CHUNKER_LLM_MAX_TOKENS,
            batch_size=12,
        )

        # print(text_data)

        organized_all_data.extend(text_data)

    return organized_all_data


def extract_data_from_source(data_path: str):
    file_name = os.path.basename(data_path)
    file_name_without_ext = file_name.split(".")[0]
    mounted_dir = "../../data_extraction/data/"
    shutil.copy(data_path, mounted_dir)
    print("Extracting data from source...")
    response = requests.post(
        "http://localhost:8000/extract", json={"file_path": f"./data/{file_name}"}
    )
    return response.status_code, file_name_without_ext


def save_data_to_vectorstore(
    llm_and_tokenizer: Tuple[
        AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained
    ],
    embedding_model: HuggingFaceEmbeddings,
    vectordb_path: str,
    data_path: str,
):
    response_status, file_name_without_ext = extract_data_from_source(data_path)
    if response_status == 200:
        print("Data extraction completed successfully.")
        organized_all_data = clean_and_organize_external_data(
            llm_and_tokenizer, file_name_without_ext
        )
        print("All data processed successfully.")

        docs = []

        with tqdm(
            total=len(organized_all_data),
            desc="Creating documents for vector store.",
            unit="Data",
        ) as pbar:
            for item in organized_all_data:
                header = (
                    f"Header: {item['header'].capitalize()}\n"
                    if item["header"] != "None"
                    else ""
                )

                if "table" in item.keys():
                    page_content = (
                        f"{header}Table: {item['table']}\nText: {item['text']}"
                    )
                    docs.append(Document(page_content=page_content))

                else:
                    for text in item["text"]:
                        page_content = f"{header}Text: {text}"
                        docs.append(Document(page_content=page_content))

                pbar.update(1)

        print(f"Created {len(docs)} documents for Chroma vector store.")

        if os.path.exists(vectordb_path):
            shutil.rmtree(vectordb_path)

        vector_store = Chroma.from_documents(
            documents=docs,
            collection_name="collection",
            embedding=embedding_model,
            persist_directory=vectordb_path,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("Saved documents in Chroma vector store.")

    else:
        print("Unexpected error occurred in data extraction process.")


def load_data_from_vectorstore(
    embedding_model: HuggingFaceEmbeddings, vectordb_path: str
):
    if os.path.exists(vectordb_path):
        vectorstore = Chroma(
            collection_name="collection",
            persist_directory=vectordb_path,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("Loaded Chroma vector store.")

        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

    else:
        print("Unable to load Chroma vector store due to path does not exist.")
        return None


if __name__ == "__main__":
    llm_and_tokenizer = load_llm_and_tokenizer(llm_name=FALCON3_10B_INSTRUCT, device=device)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb_path = "../vectordb_chroma/"

    start = perf_counter()

    save_data_to_vectorstore(
        llm_and_tokenizer=llm_and_tokenizer,
        embedding_model=embedding_model,
        vectordb_path=vectordb_path,
        data_path="../../data/Pdf/2a85b52768ea5761b773be49b09d15f0b95415b0.pdf",
    )

    print(f"Total execution time: {perf_counter() - start:.2f} seconds.")

    retriever = load_data_from_vectorstore(
        embedding_model=embedding_model, vectordb_path=vectordb_path
    )

    query = "who reviewed the study?"

    retrieved_docs = retriever.get_relevant_documents(query)
    print(retrieved_docs)
    print(len(retrieved_docs))
