from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from agents.constants.models import (
    TASK_ROUTER_SYSTEM_PROMPT,
)
from typing import Literal
from agents.utils.models import load_chat_model, get_chat_model_response


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

client = load_chat_model()

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

    response = get_chat_model_response(client=client, messages=messages)

    return response
