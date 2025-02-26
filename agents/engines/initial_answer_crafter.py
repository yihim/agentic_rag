from pathlib import Path
import os
from agents.constants.models import VLLM_BASE_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from agents.constants.models import INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT, LLM_MAX_TOKENS, VLLM_MODEL


class InitialAnswerCrafterOutput(BaseModel):
    initial_ans: str = Field(
        ...,
        description="The crafted initial answer.",
    )


load_dotenv()

client = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=os.getenv("VLLM_API_KEY"),
    model=VLLM_MODEL,
    verbose=True,
    request_timeout=None,
)

structured_client = client.with_structured_output(InitialAnswerCrafterOutput)


def craft_initial_answer(query: str, context: str):
    messages = [
        SystemMessage(content=INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT.format(query=query, context=context)),
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

    return response.initial_ans


if __name__ == "__main__":
    # Test initial answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query="What are some effective strategies for improving productivity while working from home?"
    context = "I have been working from home for about a year, and I've noticed that distractions and an unstructured daily routine are impacting my productivity. I've tried creating to-do lists and scheduling my day, but I still struggle with staying focused. I'm particularly interested in practical techniques like time blocking, the Pomodoro method, and ways to minimize interruptions. Any easy-to-implement suggestions would be very helpful."
    print(craft_initial_answer(query=query, context=context))
