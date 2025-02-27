import os
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class TavilySearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to be processed.",
        min_length=1,
        max_length=1000,
    )


class TavilySearchResult(BaseModel):
    answer: str = Field(..., description="Main answer synthesized from search results.")
    sources: Optional[List[str]] = Field(
        default=None, description="Sources of information."
    )
    confidence_score: Optional[str] = Field(
        default=None, description="Confidence score of the answer."
    )


def tavily_search(query: TavilySearchInput) -> TavilySearchResult:
    """
    Perform a search using Tavily API with structured input and output.
    """
    print("-" * 20, "TAVILY SEARCH", "-" * 20)
    response = tavily_client.search(
        query=query.query,
        include_answer="advanced",
        max_results=5,
    )

    sources = [result.get("url", "") for result in response.get("results", [])]
    scores = [result.get("score", "") for result in response.get("results", [])]
    avg_scores = str(round(sum(scores) / len(scores) * 100, 2)) + "%"

    return TavilySearchResult(
        answer=response.get("answer", ""),
        sources=sources,
        confidence_score=avg_scores,
    )


if __name__ == "__main__":
    # Test web search
    query = "what is agentic ai?"

    search_input = TavilySearchInput(query=query)

    result = tavily_search(search_input)
    formatted_search_results = f"Tavily Search Results:\n\nContent: {result.answer}\nSources: {result.sources}\nConfidence Score: {result.confidence_score}"
    print(formatted_search_results)
