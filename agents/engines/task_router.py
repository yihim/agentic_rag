from pydantic import BaseModel, Field
from constants.models import (
    TASK_ROUTER_SYSTEM_PROMPT,
)
from typing import Literal, List
from utils.models import load_chat_model
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)


class TaskRouterAction(BaseModel):
    action: Literal[
        "query_rewriter",
        "milvus_retriever",
        "tavily_searcher",
        "initial_answer_crafter",
        "response_checker",
        "final_answer_crafter",
    ] = Field(..., description="The next action to take in the workflow.")
    reasoning: str = Field(..., description="Reasoning behind the action decision.")


# task_router_llm = load_chat_model()
#
# task_router_llm = task_router_llm.with_structured_output(TaskRouterAction)


def router_action(
    llm,
    query: str,
    rewritten_query: str,
    current_step: str,
    kb_context: str,
    web_context: str,
    answer: str,
    response_check: str,
    task_action_history: List[str],
):
    status = "-" * 20, "ROUTER ACTION", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(("system", TASK_ROUTER_SYSTEM_PROMPT))

    chain = prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "rewritten_query": rewritten_query,
            "current_step": current_step,
            "kb_context": kb_context,
            "web_context": web_context,
            "answer": answer,
            "response_check": response_check,
            "task_action_history": task_action_history,
        }
    )

    return response
