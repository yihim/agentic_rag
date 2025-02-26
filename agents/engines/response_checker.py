from pathlib import Path
import os
from agents.constants.models import VLLM_BASE_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from agents.constants.models import RESPONSE_CHECKER_SYSTEM_PROMPT


class ResponseCheckerOutput(BaseModel):
    valid_answer: str = Field(
        ...,
        description="This field must contain 'yes' if the provided answer fully addresses the user query, or 'no' if it does not.",
    )


load_dotenv()

client = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=os.getenv("VLLM_API_KEY"),
    model="/model",
    verbose=True,
    request_timeout=None,
)

structured_client = client.with_structured_output(ResponseCheckerOutput)


def check_response(query: str, answer: str):
    messages = [
        SystemMessage(
            content=RESPONSE_CHECKER_SYSTEM_PROMPT.format(query=query, answer=answer)
        ),
    ]

    response = structured_client.invoke(
        input=messages,
        temperature=0.01,
        seed=42,
        top_p=0.8,
        max_tokens=1024,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )

    return response.valid_answer


if __name__ == "__main__":
    # Test response checker
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query = "How do I reset my forgotten email password?"
    yes_answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    no_answer = "Reset your password by contacting customer support."

    print(check_response(query=query, answer=yes_answer))
