import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from agents.constants.models import QWEN
from huggingface_hub import login

if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
    llm, tokenizer = AutoModelForCausalLM.from_pretrained(
        QWEN, trust_remote_code=True
    ), AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
