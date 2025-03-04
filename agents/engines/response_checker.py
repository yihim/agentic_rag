from pathlib import Path
import os
from constants.models import RESPONSE_CHECKER_SYSTEM_PROMPT
from utils.models import load_chat_model
from pydantic import BaseModel, Field
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)


class ResponseCheckerOutput(BaseModel):
    check_result: str = Field(
        ...,
        description="This field must contain 'yes' if the provided answer fully addresses the user query, or 'no' if it does not.",
    )


def check_response(llm, query: str, answer: str):
    status = "-" * 20, "CHECK RESPONSE", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(
        ("system", RESPONSE_CHECKER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = chain.invoke({"query": query, "answer": answer})

    return response.check_result if response.check_result else None


if __name__ == "__main__":
    # Test response checker
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    response_checker_llm = load_chat_model()

    response_checker_llm = response_checker_llm.with_structured_output(
        ResponseCheckerOutput
    )

    query = "How do I reset my forgotten email password?"
    yes_answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    no_answer = "Reset your password by contacting customer support."
    start = perf_counter()
    logger.info(check_response(llm=response_checker_llm, query=query, answer=no_answer))
    logger.info(f"{perf_counter() - start:.2f} seconds.")
