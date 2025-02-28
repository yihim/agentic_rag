from pymilvus import MilvusClient, MilvusException
from agents.constants.vectorstore import MILVIUS_ENDPOINT, MILVIUS_COLLECTION_NAME
from sentence_transformers import SentenceTransformer
from agents.constants.models import EMBEDDING_MODEL
import torch
from agents.utils.models import embed_text
from pprint import pprint
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

data_source = "pdf"
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL, device=device, trust_remote_code=True
)


class MilvusRetrieveOutput(BaseModel):
    context: List[Tuple[str, float]] = Field(
        default=None, description="A list of relevant contexts with confident scores."
    )


def milvus_retriever(query: str) -> Optional[MilvusRetrieveOutput]:
    try:
        print("-" * 20, "MILVUS RETRIEVER", "-" * 20)
        milvus_client = MilvusClient(uri=MILVIUS_ENDPOINT)
        milvus_client.flush(collection_name=MILVIUS_COLLECTION_NAME)
        total_data_stored = milvus_client.get_collection_stats(
            collection_name=MILVIUS_COLLECTION_NAME
        )["row_count"]
        print(
            f"Total data stored for collection_name - {MILVIUS_COLLECTION_NAME} is {total_data_stored}."
        )

        search_result = milvus_client.search(
            collection_name=MILVIUS_COLLECTION_NAME,
            data=embed_text(embedding_model=embedding_model, data=[query]),
            limit=int(0.1 * total_data_stored),
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
        res_with_distance = [
            (res["entity"]["text"], res["distance"])
            for res in search_result[0]
            if res["distance"] > 0.7
        ]
        if res_with_distance:
            pprint(f"{len(res_with_distance)} results found for query: {query}")
            return res_with_distance
        else:
            pprint(f"No results found for query: {query}.")
            return None
    except MilvusException as e:
        print(f"Unexpected error occurred in Milvus: {e.message}")
        return None


if __name__ == "__main__":
    # Test search vector store
    query = "What is agentic AI and how does it work?"
    results = milvus_retriever(query=query)
    if results is not None:
        formatted_context = "Local Knowledge Base Results:\n\n"
        for index, (item, score) in enumerate(results):
            formatted_context += f"\n{item}\n\n"
        print(formatted_context)
