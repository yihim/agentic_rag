from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from agents.constants.models import INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model, get_chat_model_response

load_dotenv()

client = load_chat_model()


def craft_initial_answer(query: str, context: str):
    messages = [
        SystemMessage(
            content=INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT.format(
                query=query, context=context
            )
        ),
    ]
    response = get_chat_model_response(client=client, messages=messages).content
    return response if response else None


if __name__ == "__main__":
    # Test initial answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query = "What are some effective strategies for improving productivity while working from home?"
    context = "I have been working from home for about a year, and I've noticed that distractions and an unstructured daily routine are impacting my productivity. I've tried creating to-do lists and scheduling my day, but I still struggle with staying focused. I'm particularly interested in practical techniques like time blocking, the Pomodoro method, and ways to minimize interruptions. Any easy-to-implement suggestions would be very helpful."
    print(craft_initial_answer(query=query, context=context))
