from pathlib import Path
from constants.models import QUERY_CLASSIFIER_SYSTEM_PROMPT
from utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
import logging

logger = logging.getLogger(__name__)


class QueryClassifierOutput(BaseModel):
    is_conversational: str = Field(
        ...,
        description="This field must contain 'true' if the query is conversational, or 'false' if it requires information retrieval.",
    )


def classify_query(llm, query: str):
    status = "-" * 20, "CLASSIFY QUERY", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(
        ("system", QUERY_CLASSIFIER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = chain.invoke({"query": query})
    return response.is_conversational if response.is_conversational else None


if __name__ == "__main__":
    # Test query classifier
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query_classifier_llm = load_chat_model()

    query_classifier_llm = query_classifier_llm.with_structured_output(
        QueryClassifierOutput
    )

    query = "What are some effective strategies for improving productivity while working from home?"
    start = perf_counter()
    logger.info(classify_query(llm=query_classifier_llm, query=query))
    logger.info(f"{perf_counter()- start:.2f} seconds.")
