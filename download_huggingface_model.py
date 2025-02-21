import os
from dotenv import load_dotenv
from agents.constants.models import QWEN
from huggingface_hub import login, snapshot_download

if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
    snapshot_download(repo_id=QWEN, trust_remote_code=True)
