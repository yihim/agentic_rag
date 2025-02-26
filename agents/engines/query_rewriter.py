from pathlib import Path
import os
from agents.constants.models import VLLM_BASE_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from agents.constants.models import QUERY_REWRITER_SYSTEM_PROMPT


class QueryRewriterOutput(BaseModel):
    improved_query: str = Field(..., description="The rewritten query.")


load_dotenv()

client = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=os.getenv("VLLM_API_KEY"),
    model="/model",
    verbose=True,
    request_timeout=None,
)

structured_client = client.with_structured_output(QueryRewriterOutput)


def rewrite_query(query: str):
    messages = [
        SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = structured_client.invoke(
        input=messages,
        temperature=0.01,
        seed=42,
        top_p=0.8,
        max_tokens=1024,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )

    return response.improved_query


if __name__ == "__main__":
    # Test query rewriter
    root_dir = Path(__file__).parent.parent.parent
    os.chdir(root_dir)
    query = "I'm having trouble understanding the recent changes in our company's HR policies, especially regarding the new remote work procedures and benefits adjustments. Can you explain what has changed and how these updates might affect my daily workflow?"
    print(rewrite_query(query=query))
