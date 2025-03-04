import aiohttp
import os
from typing import List, Dict
from dotenv import load_dotenv
from constants.models import (
    VLLM_API_CHAT_COMPLETIONS_URL,
    VLLM_API_REQUEST_PAYLOAD_TEMPLATE,
)
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

load_dotenv()

HEADERS = {
    "Authorization": f"Bearer {os.getenv('VLLM_API_KEY')}",
    "Content-Type": "application/json",
}


async def send_request(session, payload):
    """Send a single request asynchronously."""
    async with session.post(
        VLLM_API_CHAT_COMPLETIONS_URL, headers=HEADERS, json=payload
    ) as response:
        result = await response.json()
        return result


async def process_data(
    data: List[Dict[str, str]], data_type: str, system_prompt: str, max_tokens: int
):
    print(f"Processing {len(data)} {data_type} data concurrently...")
    request_payloads = []

    if data_type == "table":
        for item in data:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Below is the HTML formatted table:\n{item['table']}\nPlease provide a contextualized description for it.",
                },
            ]
            payload = VLLM_API_REQUEST_PAYLOAD_TEMPLATE.copy()
            payload["max_tokens"] = max_tokens
            payload["messages"] = messages
            payload["guided_json"] = {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            }

            request_payloads.append(payload)
    else:
        for item in data:
            title = item["header"]
            text = "\n".join(item["text"])
            context = f"Title: {title}\nContent: {text}"
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Decompose the following:\n{context}",
                },
            ]

            payload = VLLM_API_REQUEST_PAYLOAD_TEMPLATE.copy()
            payload["max_tokens"] = max_tokens
            payload["messages"] = messages
            payload["guided_json"] = {"type": "array", "items": {"type": "string"}}

            request_payloads.append(payload)

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, payload) for payload in request_payloads]
        results = await tqdm_async.gather(
            *tasks, desc=f"Processing {data_type} requests", unit="Request"
        )

    error_occurred_msg = None
    for result in results:
        if result["object"] == "error":
            error_occurred_msg = f"Error occurred: {result['message']}"
            break
    if error_occurred_msg is None:
        results = [result["choices"][0]["message"]["content"] for result in results]

        total_processed = 0
        with tqdm(
            total=len(data), desc=f"Processing {data_type} responses", unit="Data"
        ) as pbar:
            for idx, result in enumerate(results):
                # print(f"Result of {idx}: {result}")
                if result:
                    if data_type == "table":
                        extracted_response = json.loads(result)["description"]
                    else:
                        extracted_response = json.loads(result)

                    data[idx]["text"] = extracted_response

                pbar.update(1)
                total_processed += 1
                pbar.set_postfix({"Processed": f"{total_processed}/{len(data)}"})

            return data
    else:
        print(error_occurred_msg)
        return None
