import os
from agents.constants.models import (
    EMBEDDING_MODEL,
    QWEN,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
    TABLE_ORGANIZER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
)
from agents.constants.vectorstore import MILVIUS_ENDPOINT, MILVIUS_PDF_COLLECTION_NAME
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from agents.utils.models import load_llm_and_tokenizer, set_random_seed, embed_text
from typing import Tuple, Optional, List
import re
import requests
from tqdm import tqdm
import torch
import shutil
import string
from time import perf_counter
from llm_preprocess_data import hf_process_data
from vllm_preprocess_data import vllm_process_data
import asyncio
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import warnings

warnings.filterwarnings("ignore")

milvus_client = MilvusClient(uri=MILVIUS_ENDPOINT)
device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
set_random_seed(42)


def clean_and_organize_external_data(
    file_name_without_ext: str,
    llm_and_tokenizer: Optional[
        Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]
    ] = None,
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

    cleaned_data = clean_text_to_json(cleaned_references)

    print("Cleaning completed.")

    text_data = []
    table_data = []
    organized_data = []

    print(f"Cleaned data:\n{cleaned_data}")
    print("Organizing cleaned data with LLM...")

    # Organize data for llm
    for index, data in enumerate(cleaned_data):
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

        if llm_and_tokenizer is None:
            # Using VLLM llm to process data
            table_data = asyncio.run(
                vllm_process_data(
                    data=table_data,
                    data_type="table",
                    system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
                    max_tokens=TABLE_ORGANIZER_LLM_MAX_TOKENS,
                )
            )
        else:
            # Using HF llm to process data
            table_data = hf_process_data(
                data=table_data,
                data_type="table",
                llm_and_tokenizer=llm_and_tokenizer,
                system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
                max_tokens=TABLE_ORGANIZER_LLM_MAX_TOKENS,
                batch_size=4,
            )

        # print(table_data)

        organized_data.extend(table_data)

    # Process text data
    if text_data:

        if llm_and_tokenizer is None:
            # Using VLLM llm to process data
            text_data = asyncio.run(
                vllm_process_data(
                    data=text_data,
                    data_type="text",
                    system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
                    max_tokens=AGENTIC_CHUNKER_LLM_MAX_TOKENS,
                )
            )
        else:
            # Using HF llm to process data
            text_data = hf_process_data(
                data=text_data,
                data_type="text",
                llm_and_tokenizer=llm_and_tokenizer,
                system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
                max_tokens=AGENTIC_CHUNKER_LLM_MAX_TOKENS,
                batch_size=4,
            )

        # print(text_data)

        organized_data.extend(text_data)

    print("Organizing completed.")
    print(f"Organized data:\n{organized_data}")

    return organized_data


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
    embedding_model: SentenceTransformer,
    data_path: str,
    llm_and_tokenizer: Optional[
        Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]
    ] = None,
):
    response_status, file_name_without_ext = extract_data_from_source(data_path)
    if response_status == 200:
        print("Data extraction completed successfully.")
        organized_data = clean_and_organize_external_data(
            file_name_without_ext, llm_and_tokenizer
        )

        print("Preparing data for milvus vector store...")

        docs = []

        with tqdm(
            total=len(organized_data),
            desc="Creating documents for vector store.",
            unit="Data",
        ) as pbar:
            for item in organized_data:
                header = (
                    f"Header: {item['header'].capitalize()}\n"
                    if item["header"] != "None"
                    else ""
                )

                if "table" in item.keys():
                    page_content = (
                        f"{header}Table: {item['table']}\nText: {item['text']}"
                    )
                    docs.append(page_content)

                else:
                    for text in item["text"]:
                        page_content = f"{header}Text: {text}"
                        docs.append(page_content)

                pbar.update(1)

        print(f"Prepared {len(docs)} data.")

        if milvus_client.has_collection(collection_name=MILVIUS_PDF_COLLECTION_NAME):
            milvus_client.drop_collection(collection_name=MILVIUS_PDF_COLLECTION_NAME)
        milvus_client.create_collection(
            collection_name=MILVIUS_PDF_COLLECTION_NAME,
            dimension=768,
            consistency_level="Strong",
            metric_type="IP",
        )

        data_embeddings = embed_text(embedding_model, docs)

        data_to_store = []

        for index, item in enumerate(docs):
            data_to_store.append(
                {"id": index, "vector": data_embeddings[index], "text": item}
            )

        res = milvus_client.insert(
            collection_name=MILVIUS_PDF_COLLECTION_NAME, data=data_to_store
        )
        print(
            f"Saved data into milvus vector store under collection_name as {MILVIUS_PDF_COLLECTION_NAME}."
        )
        print(res)

    else:
        print("Unexpected error occurred in data extraction process.")


if __name__ == "__main__":
    # Test save vector store
    embedding_model = SentenceTransformer(
        model_name_or_path=EMBEDDING_MODEL, trust_remote_code=True, device=device
    )

    # Choose mode: hf / vllm
    process_mode = "vllm"

    if process_mode == "hf":
        llm_and_tokenizer = load_llm_and_tokenizer(llm_name=QWEN, device=device)

        start = perf_counter()

        save_data_to_vectorstore(
            llm_and_tokenizer=llm_and_tokenizer,
            embedding_model=embedding_model,
            data_path="../../data/Pdf/2a85b52768ea5761b773be49b09d15f0b95415b0.pdf",
        )

        print(
            f"Total execution time for saving data to milvus vector store: {perf_counter() - start:.2f} seconds."
        )
    else:
        start = perf_counter()

        save_data_to_vectorstore(
            embedding_model=embedding_model,
            data_path="../../data/Pdf/2a85b52768ea5761b773be49b09d15f0b95415b0.pdf",
        )

        print(
            f"Total execution time for saving data to milvus vector store: {perf_counter() - start:.2f} seconds."
        )
