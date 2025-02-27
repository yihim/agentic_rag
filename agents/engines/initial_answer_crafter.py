from pathlib import Path
from agents.constants.models import INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os


class InitialAnswerCrafterOutput(BaseModel):
    initial_ans: str = Field(..., description="The answer")


initial_answer_crafter_llm = load_chat_model()

initial_answer_crafter_llm = initial_answer_crafter_llm.with_structured_output(
    InitialAnswerCrafterOutput
)


def craft_initial_answer(llm, query: str, context: str):
    prompt = ChatPromptTemplate.from_messages(
        ("system", INITIAL_ANSWER_CRAFTER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = chain.invoke({"query": query, "context": context})
    return response.initial_ans if response.initial_ans else None


if __name__ == "__main__":
    # Test initial answer crafter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query = "What are some effective strategies for improving productivity while working from home?"
    context = "I have been working from home for about a year, and I've noticed that distractions and an unstructured daily routine are impacting my productivity. I've tried creating to-do lists and scheduling my day, but I still struggle with staying focused. I'm particularly interested in practical techniques like time blocking, the Pomodoro method, and ways to minimize interruptions. Any easy-to-implement suggestions would be very helpful."
    start = perf_counter()
    print(
        craft_initial_answer(
            llm=initial_answer_crafter_llm, query=query, context=context
        )
    )
    print(f"{perf_counter()- start:.2f} seconds.")
