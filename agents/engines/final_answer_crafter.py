from pathlib import Path
import os
from agents.constants.models import VLLM_BASE_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from agents.constants.models import FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT, LLM_MAX_TOKENS


class FinalAnswerCrafterOutput(BaseModel):
    markdown_ans: str = Field(
        ...,
        description="The crafted final answer in markdown format.",
    )


load_dotenv()

client = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=os.getenv("VLLM_API_KEY"),
    model="/model",
    verbose=True,
    request_timeout=None,
)

structured_client = client.with_structured_output(FinalAnswerCrafterOutput)


def craft_final_answer(answer: str):
    messages = [
        SystemMessage(content=FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT.format(answer=answer)),
    ]

    response = structured_client.invoke(
        input=messages,
        temperature=0.01,
        seed=42,
        top_p=0.8,
        max_tokens=LLM_MAX_TOKENS,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )

    return response.markdown_ans


if __name__ == "__main__":
    # Test final answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    print(craft_final_answer(answer=answer))
