from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json
import re
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_llm_formatted_messages(
    system_prompt: str,
    user_prompt: Dict[Any, Any],
    tokenizer: AutoTokenizer.from_pretrained,
):
    messages = [
        {"role": "system", "content": system_prompt},
        user_prompt,
    ]

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_messages


def hf_process_data(
    data: List[Dict[str, str]],
    data_type: str,
    llm_and_tokenizer: Tuple[
        AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained
    ],
    system_prompt: str,
    max_tokens: int,
    batch_size: int,
):
    """
    Process table or text data using the specified LLM.

    Args:
        data: The data to be processed (list of dictionaries).
        data_type: Type of data ('table' or 'text').
        llm_and_tokenizer: loaded llm with it's corresponding tokenizer.
        system_prompt: System prompt for the LLM.
        max_tokens: Maximum number of tokens for generation.
        batch_size: Batch size for processing.

    Returns:
        Processed data with added or updated 'text' field.
    """

    llm, tokenizer = llm_and_tokenizer

    batch_indices = []
    all_tensor_messages = []

    with tqdm(
        total=len(data), desc=f"Creating {data_type} batches", unit="Data"
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

            batch_tensors = tokenizer(
                batch_messages, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            all_tensor_messages.append(batch_tensors)
            pbar.update(len(batch))

    total_processed = 0
    with tqdm(
        total=len(data), desc=f"Processing {data_type} batches", unit="Data"
    ) as pbar:
        for tensor_messages, indices in zip(all_tensor_messages, batch_indices):
            generated_ids = llm.generate(
                **tensor_messages,
                max_new_tokens=max_tokens,
                temperature=0.001,
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
                        extracted_response = json.loads(json_match.group(1).strip())[
                            "description"
                        ]
                    else:
                        extracted_response = json.loads(response)["description"]
                else:
                    extracted_response = json.loads(response)

                data[idx]["text"] = extracted_response

                total_processed += 1
                pbar.update(1)
                pbar.set_postfix({"Processed": f"{total_processed}/{len(data)}"})

    print(f"{data_type.capitalize()} data processed successfully.")

    return data
