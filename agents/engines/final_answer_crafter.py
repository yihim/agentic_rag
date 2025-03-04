from pathlib import Path
from langchain_core.runnables import RunnableConfig
from constants.models import FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT
from utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

logger = logging.getLogger(__name__)


async def craft_final_answer(llm, answer: str, config: RunnableConfig):
    status = "-" * 20, "CRAFT FINAL ANSWER", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(
        ("system", FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = await chain.ainvoke({"answer": answer}, config=config)

    return response.content


# # Testing
# def craft_final_answer(llm, answer: str):
#     logger.info("-" * 20, "CRAFT FINAL ANSWER", "-" * 20)
#     prompt = ChatPromptTemplate.from_messages(
#         ("system", FINAL_ANSWER_CRAFTER_SYSTEM_PROMPT)
#     )
#
#     chain = prompt | llm
#
#     response = chain.invoke({"answer": answer})
#
#     return response.content
#
# if __name__ == "__main__":
#     # Test final answer crafter
#     root_dir = Path(__file__).parent.parent.parent
#     os.chdir(root_dir)
#
#     final_answer_crafter_llm = load_chat_model()
#
#     answer = "To reset your forgotten email password, go to the login page and click on 'Forgot Password'. Enter your registered email address, complete any CAPTCHA or verification steps, and then check your inbox for a reset link. Follow the instructions in the email to create a new password."
#     start = perf_counter()
#     logger.info(craft_final_answer(llm=final_answer_crafter_llm, answer=answer))
#     logger.info(f"{perf_counter() - start:.2f} seconds.")
