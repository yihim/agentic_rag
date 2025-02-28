from pathlib import Path
from agents.utils.models import load_chat_model
from time import perf_counter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage


class ConversationSummarizerOutput(BaseModel):
    summary: str = Field(..., description="The conversation summary.")


def summarize_conversation(
    llm, summary: str, messages: List[Union[HumanMessage, AIMessage]]
):
    print("-" * 20, "SUMMARIZE CONVERSATION", "-" * 20)

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = messages + [HumanMessage(content=summary_message)]

    response = llm.invoke(messages)
    return response.summary if response.summary else None


if __name__ == "__main__":
    # Test conversation summarizer
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)

    conversation_summarizer_llm = load_chat_model()

    conversation_summarizer_llm = conversation_summarizer_llm.with_structured_output(
        ConversationSummarizerOutput
    )

    summary = "We discussed your work involving models like .gguf and Llama.cpp, focusing on optimizing performance and comparing different frameworks like Hugging Face. You're interested in serving models efficiently, especially using quantized variants for speed enhancements."
    messages = [
        HumanMessage(
            content="I haven’t experimented with custom agent behaviors yet, but I’m thinking about creating specialized agents for different types of tasks. Any best practices?"
        ),
        AIMessage(
            content="That’s a great approach! Best practices include defining clear scopes for each agent, implementing a fallback mechanism if an agent fails, and using memory efficiently to maintain context across interactions. Do you plan to use LangGraph for more structured workflows?"
        ),
        HumanMessage(
            content="Yes, I want to use LangGraph to create a decision tree where agents can escalate tasks if needed. Any thoughts on optimizing this?"
        ),
        AIMessage(
            content="Using LangGraph for hierarchical task escalation is a solid strategy. To optimize, you might want to:\n1. Use a shared state to track agent decisions.\n2. Implement a priority queue for urgent tasks.\n3. Log interactions to refine agent routing over time.\n\nWould you like an example of how to structure this?"
        ),
        HumanMessage(
            content="That would be helpful! Also, do you think vector search with Milvus could assist in task assignment?"
        ),
        AIMessage(
            content="Yes! You could store agent expertise embeddings in Milvus and perform similarity searches to assign the most relevant agent dynamically. This would enhance scalability and adaptability. Do you have a specific embedding model in mind?"
        ),
    ]
    # summary = ""
    # messages = [HumanMessage(content="Hey, I'm working on deploying a .gguf model with Llama.cpp. Any tips on improving performance?"),
    #             AIMessage(content="Absolutely! For optimal performance, ensure your CUDA settings are configured correctly, and consider using a 4-bit quantized version of Llama-3.2-11B-Vision-Instruct."),
    #             HumanMessage(content="CUDA is set up, but Hugging Face seems faster. Any thoughts?"),
    #             AIMessage(content="Hugging Face often offers streamlined APIs that can boost speed in certain setups. Are you using their FastAPI integration?"),
    #             HumanMessage(content="Not yet. I'm exploring FastAPI and considering Milvus for vector storage. Any advice on Dockerizing it?"),
    #             AIMessage(content="Dockerizing Milvus on Windows with Docker Desktop is feasible. Ensure your container setup aligns with Milvus requirements for seamless integration."),
    #             HumanMessage(content="Thanks! What's your take on agent routing in LangChain for task assignment?"),
    #             AIMessage(content="Agent routing in LangChain/LangGraph offers robust task allocation capabilities. Have you encountered any specific challenges with it?"),
    #             HumanMessage(content="Not yet, but I'm intrigued by its potential for workload distribution. How do you handle complex task dependencies?"),
    #             AIMessage(content="Complex dependencies can be managed through clear task prioritization and agent communication protocols within LangChain. Have you experimented with custom agent behaviors?")]
    start = perf_counter()
    print(
        summarize_conversation(
            llm=conversation_summarizer_llm, summary=summary, messages=messages
        )
    )
    print(f"{perf_counter()- start:.2f} seconds.")
