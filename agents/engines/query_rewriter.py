from pathlib import Path
import os
from constants.models import QUERY_REWRITER_SYSTEM_PROMPT
from utils.models import load_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from time import perf_counter
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class QueryWriterOutput(BaseModel):
    rewritten_query: str = Field(..., description="The rewritten query.")


def rewrite_query(
    llm,
    query: str,
    chat_history: List[Union[HumanMessage, AIMessage]],
):
    status = "-" * 20, "REWRITE QUERY", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(("system", QUERY_REWRITER_SYSTEM_PROMPT))

    chain = prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "chat_history": chat_history,
        }
    )

    return response.rewritten_query if response.rewritten_query else None


if __name__ == "__main__":
    # Test query rewriter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query_writer_llm = load_chat_model()

    query_writer_llm = query_writer_llm.with_structured_output(QueryWriterOutput)

    query = "I'm having trouble understanding the recent changes in our company's HR policies, especially regarding the new remote work procedures and benefits adjustments. Can you explain what has changed and how these updates might affect my daily workflow?"
    chat_history = [HumanMessage(content=query)]

    start = perf_counter()
    logger.info(
        rewrite_query(
            llm=query_writer_llm,
            query=query,
            chat_history=chat_history,
        )
    )
    logger.info(f"{perf_counter() - start:.2f} seconds.")
