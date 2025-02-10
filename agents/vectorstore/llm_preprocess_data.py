from tqdm import tqdm
from typing import Optional, Union, Tuple, Dict, Any
import torch
from transformers import (
    MllamaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from agents.utils.models import quant_config_4_bit, set_random_seed
from PIL import Image
import os
import json
import re
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.getenv("HF_TOKEN_WRITE"), add_to_git_credential=True)
set_random_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_llm(llm_name: str, vision: bool = False) -> Tuple[
    Optional[
        Union[
            MllamaForConditionalGeneration.from_pretrained,
            AutoModelForCausalLM.from_pretrained,
        ]
    ],
    Optional[Union[AutoProcessor.from_pretrained, AutoTokenizer.from_pretrained]],
]:
    if vision:
        llm = MllamaForConditionalGeneration.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quant_config_4_bit(),
        )

        processor = AutoProcessor.from_pretrained(llm_name, trust_remote_code=True)

        return llm, processor

    else:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quant_config_4_bit(),
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        llm.generation_config.pad_token_id = tokenizer.pad_token_id

        return llm, tokenizer


def make_llm_formatted_messages(
    system_prompt: str,
    user_prompt: Dict[Any, Any],
    tokenizer: Optional[
        Union[AutoTokenizer.from_pretrained, AutoProcessor.from_pretrained]
    ],
):
    messages = [
        {"role": "system", "content": system_prompt},
        user_prompt,
    ]

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_messages


def process_data(
    data, data_type, llm_name, system_prompt, max_tokens, batch_size, image_dir=None
):
    """
    Process table, image, or text data using the specified LLM.

    Args:
        data: The data to be processed (list of dictionaries).
        data_type: Type of data ('table', 'image', or 'text').
        llm_name: Name of the LLM to load.
        system_prompt: System prompt for the LLM.
        max_tokens: Maximum number of tokens for generation.
        batch_size: Batch size for processing.
        image_dir: Directory containing image files (required for image data).

    Returns:
        Processed data with added or updated 'text' field.
    """
    llm, tokenizer = load_llm(llm_name=llm_name, vision=(data_type == "image"))
    print(f"Loaded {llm_name}")

    batch_indices = []
    all_tensor_messages = []

    with tqdm(
        total=len(data), desc=f"Creating {data_type} batches", unit="Item"
    ) as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_indices.append(list(range(i, min(i + batch_size, len(data)))))
            batch_messages = []

            for item in batch:
                if data_type == "table":
                    user_prompt = {
                        "role": "user",
                        "content": f"Below is the HTML formatted table:\n{item['table']}\nPlease provide a contextualized description for it.",
                    }
                elif data_type == "image":
                    user_prompt = {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe the image concisely."},
                        ],
                    }
                elif data_type == "text":
                    title = item["header"]
                    text = " ".join(item["text"])
                    context = f"Title: {title}\nContent: {text}"
                    user_prompt = {
                        "role": "user",
                        "content": f"Decompose the following:\n{context}",
                    }

                batch_messages.append(
                    make_llm_formatted_messages(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tokenizer=tokenizer,
                    )
                )

            if data_type == "image":
                batch_images = [
                    Image.open(os.path.join(image_dir, item["image"])) for item in batch
                ]
                batch_tensors = tokenizer(
                    batch_images,
                    batch_messages,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
            else:
                batch_tensors = tokenizer(
                    batch_messages, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

            all_tensor_messages.append(batch_tensors)
            pbar.update(len(batch))

    total_processed = 0
    with tqdm(
        total=len(data), desc=f"Processing {data_type} batches", unit="Item"
    ) as pbar:
        for tensor_messages, indices in zip(all_tensor_messages, batch_indices):
            if data_type == "image":
                generated_ids = llm.generate(
                    **tensor_messages,
                    max_new_tokens=max_tokens,
                )
            else:
                generated_ids = llm.generate(
                    **tensor_messages,
                    max_new_tokens=max_tokens,
                    temperature=0,
                )

            generated_parts = []
            for input_ids, output_ids in zip(tensor_messages.input_ids, generated_ids):
                generated_parts.append(output_ids[input_ids.shape[0] :])

            responses = tokenizer.batch_decode(
                generated_parts, skip_special_tokens=True
            )

            for idx, response in zip(indices, responses):
                if data_type == "table":
                    json_match = re.search(r"```json(.*?)```", response, re.DOTALL)
                    if json_match:
                        extracted_response = json.loads(json_match.group(1).strip())
                    else:
                        extracted_response = json.loads(response)
                elif data_type == "text":
                    extracted_response = json.loads(response)
                else:
                    extracted_response = response

                data[idx]["text"] = extracted_response

                total_processed += 1
                pbar.update(1)
                pbar.set_postfix({"Processed": f"{total_processed}/{len(data)}"})

    print(f"{data_type.capitalize()} data processed successfully.")
    del llm
    torch.cuda.empty_cache()

    return data
