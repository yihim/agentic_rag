from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
from agents.constants.models import (
    EMBEDDING_MODEL,
    TABLE_ORGANIZER_LLM,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
    TABLE_ORGANIZER_LLM_MAX_TOKENS,
    VISION_LLM,
    VISION_LLM_SYSTEM_PROMPT,
    VISION_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM,
    AGENTIC_CHUNKER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
)
import re
import requests
import shutil
import string
from time import perf_counter
from llm_preprocess_data import process_data
import warnings

warnings.filterwarnings("ignore")


def clean_and_organize_external_data(file_name_without_ext: str):

    image_dir = "../../data_extraction/tmp/images/"
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
    image_data = []

    def extract_image_file_name(text: str) -> str:
        match = re.search(r"/([^/]+\.[a-zA-Z0-9]+)\)", text)
        return match.group(1)

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
                    image_data.append(
                        {"header": header, "image": extract_image_file_name(item)}
                    )
                else:
                    text.append(item)

        if text:
            text_data.append({"header": header, "text": text})

    # Process table data
    if table_data:
        table_data = process_data(
            data=table_data,
            data_type="table",
            llm_name=TABLE_ORGANIZER_LLM,
            system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
            max_tokens=TABLE_ORGANIZER_LLM_MAX_TOKENS,
            batch_size=8,
        )

        print(table_data)

    # Process image data
    if image_data:
        image_dir = os.path.join(image_dir, file_name_without_ext)
        image_data = process_data(
            data=image_data,
            data_type="image",
            llm_name=VISION_LLM,
            system_prompt=VISION_LLM_SYSTEM_PROMPT,
            max_tokens=VISION_LLM_MAX_TOKENS,
            batch_size=4,
            image_dir=image_dir,
        )
        print(image_data)

    # Process text data
    if text_data:
        text_data = process_data(
            data=text_data,
            data_type="text",
            llm_name=AGENTIC_CHUNKER_LLM,
            system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
            max_tokens=AGENTIC_CHUNKER_LLM_MAX_TOKENS,
            batch_size=12,
        )

        print(text_data)


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


def save_vector_to_store(path: str, data_path: str):
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    response_status, file_name_without_ext = extract_data_from_source(data_path)
    if response_status == 200:
        print("Data extraction completed successfully.")
        clean_and_organize_external_data(file_name_without_ext)

    # clean_and_organize_external_data("2a85b52768ea5761b773be49b09d15f0b95415b0")


if __name__ == "__main__":

    start = perf_counter()

    save_vector_to_store(
        path="../vectordb_chroma/",
        data_path="../../data/Pdf/2a85b52768ea5761b773be49b09d15f0b95415b0.pdf",
    )

    print(f"Total execution time: {perf_counter() - start:.2f} seconds.")
