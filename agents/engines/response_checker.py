from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from agents.constants.models import RESPONSE_CHECKER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model, get_chat_model_response


load_dotenv()

client = load_chat_model()


def check_response(query: str, answer: str):
    messages = [
        SystemMessage(
            content=RESPONSE_CHECKER_SYSTEM_PROMPT.format(query=query, answer=answer)
        ),
    ]

    response = get_chat_model_response(client=client, messages=messages).content

    return response if response else None


if __name__ == "__main__":
    # Test response checker
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query = "How do I reset my forgotten email password?"
    yes_answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    no_answer = "Reset your password by contacting customer support."

    print(check_response(query=query, answer=yes_answer))
