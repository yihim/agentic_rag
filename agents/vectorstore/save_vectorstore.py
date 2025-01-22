from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
import torch
from agents.constants.models import (
    EMBEDDING_MODEL,
    TABLE_ORGANIZER_LLM,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from agents.utils.models import quant_config_4_bit, set_random_seed
from typing import Tuple
from dotenv import load_dotenv
import requests
import shutil
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed(42)

def load_llm_and_tokenizer() -> (
    Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]
):
    tokenizer = AutoTokenizer.from_pretrained(
        TABLE_ORGANIZER_LLM, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        TABLE_ORGANIZER_LLM, quantization_config=quant_config_4_bit(), device_map=device
    )

    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    return llm, tokenizer


def make_llm_input_ids(tokenizer: AutoTokenizer.from_pretrained, context: str):
    messages = [
        {"role": "system", "content": TABLE_ORGANIZER_LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Below is the HTML formatted table:\n{context}\nPlease organize it.",
        },
    ]

    messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([messages], return_tensors="pt").to(device)

    return model_inputs


def clean_and_organize_external_data(extracted_data_path: str):
    with open(extracted_data_path, "r", encoding="utf-8") as f:
        extracted_data = f.read()

    extracted_data_list = [data for data in extracted_data.split("\n") if data]

    llm, tokenizer = load_llm_and_tokenizer()

    for data in extracted_data_list:
        if data.startswith("<html><body><table>"):
            input_ids = make_llm_input_ids(tokenizer, data)
            generated_ids = llm.generate(**input_ids, max_new_tokens=1024)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(input_ids.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)

def extract_data_from_source(data_path: str):
    file_name = os.path.basename(data_path)
    file_name_without_ext = file_name.split(".")[0]
    mounted_dir = "../../data_extraction/data/"
    shutil.copy(data_path, mounted_dir)
    response = requests.post("http://localhost:8000/extract", json={'file_path': f"./data/{file_name}"})
    if response.status_code == 200:
        extracted_md_data_path = f"../../data_extraction/tmp/md/{file_name_without_ext}.md"
        with open(extracted_md_data_path, "r", encoding="utf-8") as f:
            extracted_data = f.read()

        print(extracted_data)


def save_vector_to_store(path: str, data_path: str):
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    extract_data_from_source(data_path)
    # clean_and_organize_external_data(extracted_data_path)


if __name__ == "__main__":
    save_vector_to_store(
        path="../vectordb_chroma/",
        data_path="../../data/Pdf/2ED27NR7CISW7J4PHXXBZ6OFPVDFHMFB.pdf",
    )