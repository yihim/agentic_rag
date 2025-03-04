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
from constants.models import VLLM_BASE_URL, VLLM_MODEL, LLM_MAX_TOKENS
import os

load_dotenv()


# Embed data for vector store and query
def embed_text(embedding_model: SentenceTransformer, data: List[str]):
    return embedding_model.encode(data, normalize_embeddings=True).tolist()


# Load langchain chat openai
def load_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.getenv("VLLM_API_KEY"),
        model=VLLM_MODEL,
        verbose=True,
        max_tokens=LLM_MAX_TOKENS,
        request_timeout=None,
        max_retries=3,
    )
