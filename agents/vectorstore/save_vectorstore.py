import os
from agents.constants.models import (
    EMBEDDING_MODEL,
    QWEN,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
    AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
    LLM_MAX_TOKENS
)
from agents.constants.vectorstore import (
    MILVIUS_ENDPOINT,
    MILVIUS_COLLECTION_NAME,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from agents.utils.models import load_llm_and_tokenizer, set_random_seed, embed_text
from typing import Tuple, Optional, List
import requests
from tqdm import tqdm
import torch
import shutil
from agents.utils.vectorstore import clean_references, clean_text_to_json
from time import perf_counter
from llm_preprocess_data import hf_process_data
from vllm_preprocess_data import vllm_process_data
import asyncio
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)
import re
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import uuid
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

milvus_client = MilvusClient(uri=MILVIUS_ENDPOINT)
device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
set_random_seed(42)
root_dir = Path.cwd()


def clean_and_organize_external_data(
    data_source: str,
    file_name_without_ext: str,
    llm_and_tokenizer: Optional[
        Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]
    ] = None,
):

    md_dir = (
        os.path.join(root_dir, f"data_extracted/pdf_tmp/md/{file_name_without_ext}.md")
        if data_source == "pdf"
        else os.path.join(
            root_dir, f"data_extracted/web_tmp/{file_name_without_ext}.md"
        )
    )

    with open(md_dir, "r", encoding="utf-8") as f:
        extracted_data = f.read()

    print("Cleaning extracted data...")

    cleaned_references = clean_references(extracted_data)

    cleaned_data = clean_text_to_json(cleaned_references)

    print("Cleaning completed.")

    text_data = []
    table_data = []
    organized_data = []

    print(f"Cleaned data:\n{cleaned_data}")
    print("Organizing cleaned data with LLM...")

    # Organize data for llm
    if data_source == "pdf":
        for data in cleaned_data:
            header = data["header"]
            content = data["content"].split("\n")

            text = []

            for item in content:
                if item:
                    if item.startswith("<html><body><table>"):
                        table_data.append({"header": header, "table": item})
                    elif item.startswith("![]"):
                        pass
                    else:
                        text.append(item)

            if text:
                text_data.append({"header": header, "text": text})
    else:
        for data in cleaned_data:
            header = data["header"]
            content = data["content"].split("\n")

            text = []

            for item in content:
                if item:
                    if item.startswith("!["):
                        image_title = re.search(r"!\[(.*?)\]", item)
                        image_reference = re.search(r"\((.*?)\)", item)
                        if image_title and image_reference:
                            text_data.append(
                                {
                                    "header": header,
                                    "text": f"Image titled: {image_title.group(1)}\nImage reference: {image_reference.group(1)}",
                                }
                            )
                    else:
                        text.append(item)

            if text:
                text_data.append({"header": header, "text": text})

    # Process table data
    if table_data:

        if llm_and_tokenizer is None:
            # Using VLLM llm to process data
            table_data = asyncio.run(
                vllm_process_data(
                    data=table_data,
                    data_type="table",
                    system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
                    max_tokens=LLM_MAX_TOKENS,
                )
            )
        else:
            # Using HF llm to process data
            table_data = hf_process_data(
                data=table_data,
                data_type="table",
                llm_and_tokenizer=llm_and_tokenizer,
                system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
                max_tokens=LLM_MAX_TOKENS,
                batch_size=4,
            )

        # print(table_data)

        organized_data.extend(table_data)

    # Process text data
    if text_data:

        if llm_and_tokenizer is None:
            # Using VLLM llm to process data
            text_data = asyncio.run(
                vllm_process_data(
                    data=text_data,
                    data_type="text",
                    system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
                    max_tokens=LLM_MAX_TOKENS,
                )
            )
        else:
            # Using HF llm to process data
            text_data = hf_process_data(
                data=text_data,
                data_type="text",
                llm_and_tokenizer=llm_and_tokenizer,
                system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
                max_tokens=LLM_MAX_TOKENS,
                batch_size=4,
            )

        # print(text_data)

        organized_data.extend(text_data)

    print("Organizing completed.")
    print(f"Organized data:\n{organized_data}")

    return organized_data


def extract_pdf_data_from_source(data_path: Path) -> Tuple[int, str]:
    file_name = os.path.basename(data_path)
    file_name_without_ext = file_name.split(".")[0]
    mounted_dir = root_dir / "data_extracted" / "pdf_data"
    shutil.copy(data_path, mounted_dir)
    print(f"Extracting PDF data from {data_path}")
    response = requests.post(
        "http://localhost:8000/extract", json={"file_path": f"./data/{file_name}"}
    )
    return response.status_code, file_name_without_ext


async def extract_web_data_from_source(url: str) -> Optional[str]:

    md_file_name = uuid.uuid4()
    md_dir = root_dir / "data_extracted" / "web_tmp"
    md_dir.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    )
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        exclude_external_links=False,
        process_iframes=True,
        remove_overlay_elements=True,
        cache_mode=CacheMode.DISABLED,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.3),
            options={
                "ignore_links": True,
                "ignore_images": False,
            },
        ),
    )

    try:
        print(f"Extracting Web data from {url}")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=run_config,
                cache_mode=CacheMode.DISABLED,
                magic=True,
                simulate_user=True,
                override_navigator=True,
            )

            if result.success:
                print(result.markdown_v2.fit_markdown)
                with open(
                    os.path.join(md_dir, str(md_file_name) + ".md"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(result.markdown_v2.fit_markdown)
                return str(md_file_name)
            else:
                print(f"Crawl failed: {result.error_message}")
                return None
    except Exception as e:
        print(f"An exception occurred during crawling: {e}")
        return None


def save_data_to_vectorstore(
    data_source: str,
    embedding_model: SentenceTransformer,
    embedding_dim: int,
    data_path: Optional[Path] = None,
    data_source_url: Optional[str] = None,
    llm_and_tokenizer: Optional[
        Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]
    ] = None,
):
    extracted_md_file_name = None

    if data_source == "pdf" and data_path is None:
        print("Please provide a valid path.")
    elif data_source == "pdf" and data_path is not None:
        response_status, file_name_without_ext = extract_pdf_data_from_source(data_path)
        if response_status == 200:
            extracted_md_file_name = file_name_without_ext
        else:
            print("Unexpected error occurred in data extraction process.")
    elif data_source == "web" and data_source_url is None:
        print("Please provide a valid url.")
    elif data_source == "web" and data_source_url is not None:
        file_name_without_ext = asyncio.run(
            extract_web_data_from_source(url=data_source_url)
        )
        extracted_md_file_name = file_name_without_ext
    else:
        print("Invalid data source. ('pdf' or 'web' only)")

    if extracted_md_file_name is not None:
        print("Data extraction completed successfully.")
        organized_data = clean_and_organize_external_data(
            data_source, extracted_md_file_name, llm_and_tokenizer
        )

        print("Preparing data for milvus vector store...")

        docs = []

        with tqdm(
            total=len(organized_data),
            desc="Creating data for vector store.",
            unit="Data",
        ) as pbar:
            for item in organized_data:
                header = (
                    f"Header: {item['header'].capitalize()}\n"
                    if item["header"] != "None"
                    else ""
                )

                if "table" in item.keys():
                    page_content = (
                        f"{header}Table: {item['table']}\nText: {item['text']}"
                    )
                    docs.append(page_content)

                else:
                    for text in item["text"]:
                        page_content = f"{header}Text: {text}"
                        docs.append(page_content)

                pbar.update(1)

        print(f"Prepared {len(docs)} data.")

        if milvus_client.has_collection(collection_name=MILVIUS_COLLECTION_NAME):
            milvus_client.drop_collection(collection_name=MILVIUS_COLLECTION_NAME)
        milvus_client.create_collection(
            collection_name=MILVIUS_COLLECTION_NAME,
            dimension=embedding_dim,
            consistency_level="Strong",
            metric_type="IP",
        )

        data_embeddings = embed_text(embedding_model, docs)

        data_to_store = []

        for index, item in enumerate(docs):
            data_to_store.append(
                {"id": index, "vector": data_embeddings[index], "text": item}
            )

        res = milvus_client.insert(
            collection_name=MILVIUS_COLLECTION_NAME, data=data_to_store
        )
        print(
            f"Saved data into milvus vector store under collection_name as {MILVIUS_COLLECTION_NAME}."
        )
        print(res)
    else:
        print("Variable 'extracted_md_file_name' is None.")


if __name__ == "__main__":

    root_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(root_dir)

    print(f"Current working directory: {Path.cwd()}")

    # Test save vector store
    embedding_model = SentenceTransformer(
        model_name_or_path=EMBEDDING_MODEL, trust_remote_code=True, device=device
    )
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    # Choose mode: hf / vllm
    process_mode = "vllm"

    # Choose source: pdf / web
    data_source = "web"

    data_path = (
        (root_dir / "data" / "Pdf" / "2a85b52768ea5761b773be49b09d15f0b95415b0.pdf")
        if data_source == "pdf"
        else None
    )
    data_source_url = (
        "https://qwenlm.github.io/blog/qwen2.5/#qwen25"
        if data_source == "web"
        else None
    )

    if process_mode == "hf":
        llm_and_tokenizer = load_llm_and_tokenizer(llm_name=QWEN, device=device)

        start = perf_counter()

        save_data_to_vectorstore(
            data_source=data_source,
            llm_and_tokenizer=llm_and_tokenizer,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            data_path=data_path,
            data_source_url=data_source_url,
        )

        print(
            f"Total execution time for saving data into milvus vector store: {perf_counter() - start:.2f} seconds."
        )
    else:
        start = perf_counter()

        save_data_to_vectorstore(
            data_source=data_source,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            data_path=data_path,
            data_source_url=data_source_url,
        )

        print(
            f"Total execution time for saving data into milvus vector store: {perf_counter() - start:.2f} seconds."
        )
