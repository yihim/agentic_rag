from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from agents.constants.models import QUERY_REWRITER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model, get_chat_model_response

load_dotenv()

client = load_chat_model()


def rewrite_query(query: str):
    messages = [
        SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT.format(query=query)),
    ]

    response = get_chat_model_response(client=client, messages=messages).content

    return response if response else None


if __name__ == "__main__":
    # Test query rewriter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)
    query = "I'm having trouble understanding the recent changes in our company's HR policies, especially regarding the new remote work procedures and benefits adjustments. Can you explain what has changed and how these updates might affect my daily workflow?"
    print(rewrite_query(query=query))
