from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images, read_local_office
from magic_pdf.config.enums import SupportedPdfParseMethod
import os
import logging
import torch
import gc

# Get logger for this module
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        # Clear PyTorch GPU cache
        torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        logger.info("GPU memory cleared")
    else:
        logger.info("No GPU available")


def extract_data_from_source(data_path: str) -> str:
    split_file_name = os.path.basename(data_path).split(".")
    name_without_suff = split_file_name[0]
    file_type = split_file_name[1].lower()

    local_image_dir, local_md_dir = f"./tmp/images/{name_without_suff}/", "./tmp/md/"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_md_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    try:
        if file_type in ["jpeg", "jpg", "png"]:
            logger.info(f"Source is image file, processing {name_without_suff}")
            ds = read_local_images(data_path)[0]

            ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                md_writer, f"{name_without_suff}.md", image_dir
            )

        elif file_type in ["doc", "docx", "ppt", "pptx"]:
            logger.info(f"Source is office file, processing {name_without_suff}")
            ds = read_local_office(data_path)[0]

            if ds.classify() == SupportedPdfParseMethod.OCR:
                ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                    md_writer, f"{name_without_suff}.md", image_dir
                )
            else:
                ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                    md_writer, f"{name_without_suff}.md", image_dir
                )

        elif file_type == "pdf":
            logger.info(f"Source is PDF file, processing {name_without_suff}")
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
        logger.error(
            f"Unexpected error occurred in extract_data_from_source function: {e}",
            exc_info=True,
        )
        print(f"Unexpected error occurred in extract_data_from_source function: {e}")
    finally:
        extracted_data_path = os.path.join(local_md_dir, name_without_suff + ".md")
        logger.info(f"Extracted data saved in: {extracted_data_path}")
        clear_gpu_memory()
        return extracted_data_path


if __name__ == "__main__":
    extracted_data_path = extract_data_from_source(
        data_path="../../data/Pdf/1dcf57a5007b56254583423ba31107d22459bccf.pdf"
    )
