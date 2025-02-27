from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from agents.constants.models import FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model, get_chat_model_response


load_dotenv()

client = load_chat_model()


def craft_final_answer(answer: str):
    messages = [
        SystemMessage(content=FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT.format(answer=answer)),
    ]

    response = get_chat_model_response(client=client, messages=messages).content

    return response if response else None


if __name__ == "__main__":
    # Test final answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    print(craft_final_answer(answer=answer))
