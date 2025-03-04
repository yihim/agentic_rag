from pathlib import Path
from constants.models import CONVERSATION_RESPONDER_SYSTEM_PROMPT
from utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
import os
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import logging

logger = logging.getLogger(__name__)


async def response_conversation(
    llm,
    query: str,
    chat_history: List[Union[HumanMessage, AIMessage]],
    config: RunnableConfig,
):
    status = "-" * 20, "RESPOND CONVERSATION", "-" * 20
    logger.info(status)
    prompt = ChatPromptTemplate.from_messages(
        ("system", CONVERSATION_RESPONDER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = await chain.ainvoke(
        {
            "query": query,
            "chat_history": chat_history,
        },
        config=config,
    )
    return response.content


# # Testing
# async def response_conversation(
#     llm,
#     query: str,
#     chat_history: List[Union[HumanMessage, AIMessage]],
#     conversation_summary: str,
# ):
#     logger.info("-" * 20, "RESPOND CONVERSATION", "-" * 20)
#     prompt = ChatPromptTemplate.from_messages(
#         ("system", CONVERSATION_RESPONDER_SYSTEM_PROMPT)
#     )
#
#     chain = prompt | llm
#
#     response = chain.invoke(
#         {
#             "query": query,
#             "chat_history": chat_history,
#             "conversation_summary": conversation_summary,
#         }
#     )
#     return response.content
#
#
# if __name__ == "__main__":
#     # Test query classifier
#     root_dir = Path(__file__).parent.parent.parent
#     os.chdir(root_dir)
#
#     conversation_responder_llm = load_chat_model()
#
#     query = "What are some effective strategies for improving productivity while working from home?"
#     chat_history = [HumanMessage(content=query)]
#     conversation_summary = ""
#
#     start = perf_counter()
#     logger.info(
#         response_conversation(
#             llm=conversation_responder_llm,
#             query=query,
#             chat_history=chat_history,
#             conversation_summary=conversation_summary,
#         )
#     )
#     logger.info(f"{perf_counter()- start:.2f} seconds.")
