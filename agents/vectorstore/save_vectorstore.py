from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
import torch
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
    AutoProcessor,
    TextStreamer,
)
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from agents.utils.models import quant_config_4_bit, set_random_seed
from typing import Tuple, Optional, Any, Dict, Union
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import List, Iterator
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic.v1 import BaseModel, Field
from bs4 import BeautifulSoup
import json
import re
import requests
import shutil
import string
from huggingface_hub import login
import openai
from tqdm import tqdm
from time import perf_counter, sleep
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed(42)


def load_llm(llm_name: str, vision: bool = False) -> Tuple[
    Optional[
        Union[
            MllamaForConditionalGeneration.from_pretrained,
            AutoModelForCausalLM.from_pretrained,
        ]
    ],
    Optional[Union[AutoProcessor.from_pretrained, AutoTokenizer.from_pretrained]],
]:
    if vision:
        llm = MllamaForConditionalGeneration.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quant_config_4_bit(),
        )

        processor = AutoProcessor.from_pretrained(llm_name, trust_remote_code=True)

        return llm, processor

    else:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quant_config_4_bit(),
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        llm.generation_config.pad_token_id = tokenizer.pad_token_id

        return llm, tokenizer


def make_llm_formatted_messages(
    system_prompt: str,
    user_prompt: Dict[Any, Any],
    tokenizer: Optional[
        Union[AutoTokenizer.from_pretrained, AutoProcessor.from_pretrained]
    ],
):
    messages = [
        {"role": "system", "content": system_prompt},
        user_prompt,
    ]

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_messages


def process_data(
    data, data_type, llm_name, system_prompt, max_tokens, batch_size, image_dir=None
):
    """
    Process table, image, or text data using the specified LLM.

    Args:
        data: The data to be processed (list of dictionaries).
        data_type: Type of data ('table', 'image', or 'text').
        llm_name: Name of the LLM to load.
        system_prompt: System prompt for the LLM.
        max_tokens: Maximum number of tokens for generation.
        batch_size: Batch size for processing.
        image_dir: Directory containing image files (required for image data).

    Returns:
        Processed data with added or updated 'text' field.
    """
    llm, tokenizer = load_llm(llm_name=llm_name, vision=(data_type == "image"))
    print(f"Loaded {llm_name}")

    batch_indices = []
    all_tensor_messages = []

    with tqdm(
        total=len(data), desc=f"Creating {data_type} batches", unit="Item"
    ) as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_indices.append(list(range(i, min(i + batch_size, len(data)))))
            batch_messages = []

            for item in batch:
                if data_type == "table":
                    user_prompt = {
                        "role": "user",
                        "content": f"Below is the HTML formatted table:\n{item['table']}\nPlease provide a contextualized description for it.",
                    }
                elif data_type == "image":
                    user_prompt = {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe the image concisely."},
                        ],
                    }
                elif data_type == "text":
                    title = item["header"]
                    text = " ".join(item["text"])
                    context = f"Title: {title}\nContent: {text}"
                    user_prompt = {
                        "role": "user",
                        "content": f"Decompose the following:\n{context}",
                    }

                batch_messages.append(
                    make_llm_formatted_messages(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tokenizer=tokenizer,
                    )
                )

            if data_type == "image":
                batch_images = [
                    Image.open(os.path.join(image_dir, item["image"])) for item in batch
                ]
                batch_tensors = tokenizer(
                    batch_images,
                    batch_messages,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
            else:
                batch_tensors = tokenizer(
                    batch_messages, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

            all_tensor_messages.append(batch_tensors)
            pbar.update(len(batch))

    total_processed = 0
    with tqdm(
        total=len(data), desc=f"Processing {data_type} batches", unit="Item"
    ) as pbar:
        for tensor_messages, indices in zip(all_tensor_messages, batch_indices):
            if data_type == "image":
                generated_ids = llm.generate(
                    **tensor_messages,
                    max_new_tokens=max_tokens,
                )
            else:
                generated_ids = llm.generate(
                    **tensor_messages,
                    max_new_tokens=max_tokens,
                    temperature=0,
                )

            generated_parts = []
            for input_ids, output_ids in zip(tensor_messages.input_ids, generated_ids):
                generated_parts.append(output_ids[input_ids.shape[0] :])

            responses = tokenizer.batch_decode(
                generated_parts, skip_special_tokens=True
            )

            for idx, response in zip(indices, responses):
                if data_type == "table":
                    json_match = re.search(r"```json(.*?)```", response, re.DOTALL)
                    if json_match:
                        extracted_response = json.loads(json_match.group(1).strip())
                    else:
                        extracted_response = json.loads(response)
                elif data_type == "text":
                    extracted_response = json.loads(response)
                else:
                    extracted_response = response

                data[idx]["text"] = extracted_response

                total_processed += 1
                pbar.update(1)
                pbar.set_postfix({"Processed": f"{total_processed}/{len(data)}"})

    print(f"{data_type.capitalize()} data processed successfully.")
    del llm
    torch.cuda.empty_cache()

    return data


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
