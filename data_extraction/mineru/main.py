from function import extract_data_from_source
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SupportedFileTypes(str, Enum):
    PDF = "pdf"
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"


class ExtractionRequest(BaseModel):
    file_path: str


app = FastAPI(
    title="Document Extraction API",
    description="API for extracting data from various document formats",
    version="1.0.0",
)


def validate_file_path(file_path: str) -> bool:
    path = Path(file_path)
    return path.exists() and path.is_file()


@app.post("/extract", response_model=dict)
async def extract_document(request: ExtractionRequest):
    """
    Extract data from the provided document path

    Args:
        request: ExtractionRequest containing file_path and optional output_dir

    Returns:
        dict: Contains the path to the extracted data
    """
    if not validate_file_path(request.file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File not found or inaccessible: {request.file_path}",
        )

    try:
        extracted_path = extract_data_from_source(request.file_path)
        return {
            "status": "success",
            "extracted_file_path": extracted_path,
            "message": "Data extracted successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
