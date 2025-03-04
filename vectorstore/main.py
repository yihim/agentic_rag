from fastapi import FastAPI
from pydantic import BaseModel
import logging
import sys
import torch
import warnings
from function import save_data_to_vectorstore
from constants.models import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
from time import perf_counter

warnings.filterwarnings("ignore")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


class VectorstoreReqeust(BaseModel):
    data_source: str
    data_path: str
    data_source_url: str
    file_name: str
    file_name_without_ext: str


app = FastAPI(
    title="Multi-Agents API",
    description="API for saving data into vector store",
    version="1.0.0",
)

embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL, trust_remote_code=True, device=device
)
embedding_dim = embedding_model.get_sentence_embedding_dimension()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/save-vectorstore")
def save_vectorstore(request: VectorstoreReqeust):

    data_source_url = request.data_source_url if request.data_source_url else None
    data_path = request.data_path if request.data_path else None
    file_name = request.file_name if request.file_name else None
    file_name_without_ext = (
        request.file_name_without_ext if request.file_name_without_ext else None
    )

    start = perf_counter()

    save_data_to_vectorstore(
        data_source=request.data_source,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        data_path=data_path,
        data_source_url=data_source_url,
        file_name=file_name,
        file_name_without_ext=file_name_without_ext,
    )

    logger.info(f"Total execution time: {perf_counter() - start:.2f} seconds.")
    return {"status": "Success"}
