from pathlib import Path
import os
from agents.constants.models import RESPONSE_CHECKER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model
from pydantic import BaseModel, Field
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate


class ResponseCheckerOutput(BaseModel):
    check_result: str = Field(
        ...,
        description="This field must contain 'yes' if the provided answer fully addresses the user query, or 'no' if it does not.",
    )


client = load_chat_model()

response_checker_llm = load_chat_model()

response_checker_llm = response_checker_llm.with_structured_output(
    ResponseCheckerOutput
)


def check_response(llm, query: str, answer: str):
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

    query = "How do I reset my forgotten email password?"
    yes_answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    no_answer = "Reset your password by contacting customer support."
    start = perf_counter()
    print(check_response(llm=response_checker_llm, query=query, answer=no_answer))
    print(f"{perf_counter() - start:.2f} seconds.")
