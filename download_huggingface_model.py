import os
from dotenv import load_dotenv
from agents.constants.models import (
    QWEN_14B_INSTRUCT,
    QWEN_14B_INSTRUCT_AWQ,
    QWEN_14B_INSTRUCT_BNB_4BIT,
    EMBEDDING_MODEL,
)
from huggingface_hub import login, snapshot_download

if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
    snapshot_download(repo_id=QWEN_14B_INSTRUCT_AWQ)  # change as needed
    snapshot_download(repo_id=EMBEDDING_MODEL)
