import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class TavilySearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to be processed",
        min_length=1,
        max_length=1000,
    )


class TavilySearchResult(BaseModel):
    answer: str = Field(..., description="Main answer synthesized from search results")
    sources: Optional[List[str]] = Field(
        default=None, description="Sources of information"
    )
    confidence_score: Optional[float] = Field(
        default=None, description="Confidence score of the answer", ge=0.0, le=100.0
    )


def tavily_search(query: TavilySearchInput) -> TavilySearchResult:
    """
    Perform a search using Tavily API with structured input and output.
    """
    response = tavily_client.search(
        query=query.query,
        include_answer="advanced",
        max_results=5,
    )

    sources = [result.get("url", "") for result in response.get("results", [])]
    scores = [result.get("score", "") for result in response.get("results", [])]
    avg_scores = sum(scores) / len(scores)

    return TavilySearchResult(
        answer=response.get("answer", ""),
        sources=sources,
        confidence_score=round(avg_scores * 100, 2),
    )


web_search_tool = [
    StructuredTool.from_function(
        func=tavily_search,
        name="Tavily search",
        description="A tool for retrieving relevant information when the LLM requires external knowledge beyond its available sources.",
        args_schema=TavilySearchInput,
        return_direct=True,
    )
]


if __name__ == "__main__":
    # Testing
    query = "what is agentic ai?"

    search_input = TavilySearchInput(query=query)

    result = tavily_search(search_input)

    print("Answer:", result.answer)
    print("\nSources:", result.sources)
    print("\nConfidence Score:", result.confidence_score)
