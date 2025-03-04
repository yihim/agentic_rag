from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import uvicorn
import sys
import warnings
import torch
from function import create_multi_agents
from sentence_transformers import SentenceTransformer
from constants.models import EMBEDDING_MODEL
from typing import List, AsyncGenerator, Union
from langchain_core.messages import HumanMessage, AIMessage
import json
from dataclasses import dataclass, asdict


warnings.filterwarnings("ignore")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to decode messages
def decode_messages(json_str: str) -> List[Union[HumanMessage, AIMessage]]:
    """Decode a JSON string back to a list of message objects"""
    data = json.loads(json_str)
    result = []

    for item in data:
        if item.get("type") == "human":
            result.append(
                HumanMessage(
                    content=item["content"],
                    additional_kwargs=item.get("additional_kwargs", {}),
                )
            )
        elif item.get("type") == "ai":
            result.append(
                AIMessage(
                    content=item["content"],
                    additional_kwargs=item.get("additional_kwargs", {}),
                )
            )

    return result


class AgentsRequest(BaseModel):
    query: str
    chat_history: str
    thread_id: str


app = FastAPI(
    title="Multi-Agents API",
    description="API for responding queries",
    version="1.0.0",
)

embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL, device=device, trust_remote_code=True
)

graph = create_multi_agents(embedding_model=embedding_model)


async def generate_stream(request: AgentsRequest) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": request.thread_id, "recursion_limit": 35}}

    chat_history_decoded = decode_messages(request.chat_history)

    state = {
        "messages": chat_history_decoded,
        "query": request.query,
        "rewritten_query": "",
        "kb_context": "",
        "web_context": "",
        "answer": "",
        "response_check": "",
        "current_step": "",
        "task_action_history": [],
        "task_action_reason_history": [],
        "is_conversational": "",
    }

    async for msg, metadata in graph.astream(
        input=state, config=config, stream_mode="messages"
    ):
        if msg.content:
            yield msg.content


@app.post("/chat")
async def response_query(request: AgentsRequest):
    return StreamingResponse(
        generate_stream(request),
        media_type="text/plain",  # for text streaming
        # media_type="application/x-ndjson" # if wanted to stream in json object
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020, reload=True)
