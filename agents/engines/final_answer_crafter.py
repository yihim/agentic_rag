from pathlib import Path
from agents.constants.models import FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os


class FinalAnswerCrafterOutput(BaseModel):
    markdown_ans: str = Field(..., description="The markdown formatted answer.")


final_answer_crafter_llm = load_chat_model()

final_answer_crafter_llm = final_answer_crafter_llm.with_structured_output(
    FinalAnswerCrafterOutput
)


def craft_final_answer(llm, answer: str):
    print("-" * 20, "CRAFT FINAL ANSWER", "-" * 20)
    prompt = ChatPromptTemplate.from_messages(
        ("system", FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = chain.invoke({"answer": answer})

    return response.markdown_ans if response.markdown_ans else None


if __name__ == "__main__":
    # Test final answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
    start = perf_counter()
    print(craft_final_answer(llm=final_answer_crafter_llm, answer=answer))
    print(f"{perf_counter() - start:.2f} seconds.")
