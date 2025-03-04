import random
import numpy as np
import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Tuple, List


# 4 Bit quantization
def quant_config_4_bit():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# 8 Bit quantization
def quant_config_8_bit():
    return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)


# Embed data for vector store and query
def embed_text(embedding_model: SentenceTransformer, data: List[str]):
    return embedding_model.encode(data, normalize_embeddings=True).tolist()


# Load llm and tokenizer
def load_llm_and_tokenizer(llm_name: str, device: str) -> Tuple[
    AutoModelForCausalLM.from_pretrained,
    AutoTokenizer.from_pretrained,
]:
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=quant_config_4_bit(),
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    print(f"LLM in memory: {llm.get_memory_footprint() / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Loaded {llm_name}")

    return llm, tokenizer
