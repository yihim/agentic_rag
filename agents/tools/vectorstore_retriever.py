from agents.vectorstore.save_load_vectorstore import load_data_from_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from agents.constants.models import EMBEDDING_MODEL
import torch
from langchain.tools.retriever import create_retriever_tool

device = "cuda" if torch.cuda.is_available() else "cpu"
vectordb_path = "../vectordb_chroma"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

retriever_tool = [
    create_retriever_tool(
        retriever=load_data_from_vectorstore(
            embedding_model=embedding_model, vectordb_path=vectordb_path
        ),
        name="Chroma Retriever",
        description="Use this tool to search and retrieve relevant information from the local knowledge base.",
    )
]

if __name__ == "__main__":
    retriever = load_data_from_vectorstore(
        embedding_model=embedding_model, vectordb_path=vectordb_path
    )

    query = "what is nvidia?"

    retrieved_docs = retriever.get_relevant_documents(query)
    print(retrieved_docs)
    print(len(retrieved_docs))
