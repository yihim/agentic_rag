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
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agents.constants.models import VLLM_BASE_URL, VLLM_MODEL
import os

load_dotenv()


# Reproducibility
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


# Load langchain chat openai
def load_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.getenv("VLLM_API_KEY"),
        model=VLLM_MODEL,
        verbose=True,
        request_timeout=None,
        model_kwargs={"stream": True},
        stream_usage=True,
        max_retries=3,
    )
