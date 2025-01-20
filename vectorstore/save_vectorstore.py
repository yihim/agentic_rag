from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import os
import pprint
from magic_pdf.config.enums import SupportedPdfParseMethod
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from constants.models import EMBEDDING_MODEL
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_vector_to_store(path: str, data_path: str):
    try:
        # embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
        name_without_suff = os.path.basename(data_path).split(".")[0]

        local_image_dir, local_md_dir = "./output/images", "output"
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_md_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(
            local_image_dir
        ), FileBasedDataWriter(local_md_dir)

        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(data_path)

        ds = PymuDocDataset(pdf_bytes)

        if ds.classify() == SupportedPdfParseMethod.OCR:
            ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                md_writer, f"{name_without_suff}.md", image_dir
            )
        else:
            ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                md_writer, f"{name_without_suff}.md", image_dir
            )

    except Exception as e:
        print(f"Unexpected error occurred in save_vector_to_store function: {e}")


if __name__ == "__main__":
    save_vector_to_store(
        path="../vectordb_chroma/",
        data_path="../data/Pdf/2ED27NR7CISW7J4PHXXBZ6OFPVDFHMFB.pdf",
    )
