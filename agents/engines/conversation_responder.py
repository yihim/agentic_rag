from pathlib import Path
from agents.constants.models import CONVERSATION_RESPONDER_SYSTEM_PROMPT
from agents.utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage


class ConversationalResponderOutput(BaseModel):
    response: str = Field(..., description="The appropriate response to the query.")


conversation_responder_llm = load_chat_model()

conversation_responder_llm = conversation_responder_llm.with_structured_output(
    ConversationalResponderOutput
)


def response_conversation(llm, query: str, chat_history: List[Union[HumanMessage, AIMessage]], conversation_summary: str):
    print("-" * 20, "RESPOND CONVERSATION", "-" * 20)
    prompt = ChatPromptTemplate.from_messages(
        ("system", CONVERSATION_RESPONDER_SYSTEM_PROMPT)
    )

    chain = prompt | llm

    response = chain.invoke({"query": query, "chat_history": chat_history, "conversation_summary": conversation_summary})
    return response.response if response.response else None


if __name__ == "__main__":
    # Test query classifier
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    query = "What are some effective strategies for improving productivity while working from home?"
    chat_history = [HumanMessage(content=query)]
    conversation_summary = ""

    start = perf_counter()
    print(
        response_conversation(
            llm=conversation_responder_llm, query=query, chat_history=chat_history, conversation_summary=conversation_summary)
    )
    print(f"{perf_counter()- start:.2f} seconds.")
