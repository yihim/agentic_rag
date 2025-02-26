import os
from agents.constants.models import VLLM_BASE_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from agents.constants.models import TASK_ROUTER_SYSTEM_PROMPT, LLM_MAX_TOKENS
from typing import Literal


class TaskRouterAction(BaseModel):
    action: Literal[
        "rewrite_query",
        "check_kb",
        "web_search",
        "generate_answer",
        "check_response",
        "finalize_answer",
        "retry",
    ] = Field(description="The next action to take in the workflow.")
    reasoning: str = Field(description="Reasoning behind the action decision.")


load_dotenv()

client = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=os.getenv("VLLM_API_KEY"),
    model="/model",
    verbose=True,
    request_timeout=None,
)

structured_client = client.with_structured_output(TaskRouterAction)


def router_action(
    query: str,
    rewritten_query: str,
    current_step: str,
    kb_context: str,
    web_context: str,
    answer: str,
    response_check: str,
    task_history: str,
):
    messages = [
        SystemMessage(
            content=TASK_ROUTER_SYSTEM_PROMPT.format(
                query=query,
                rewritten_query=rewritten_query,
                current_step=current_step,
                kb_context=kb_context,
                web_context=web_context,
                answer=answer,
                response_check=response_check,
                task_history=task_history,
            )
        ),
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

    return response
