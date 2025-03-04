import streamlit as st
import uuid
import tempfile
import requests
from pathlib import Path
from constants import AGENTS_URL, VECTORSTORE_URL
import os
import shutil
import nest_asyncio
from langchain_core.messages import HumanMessage, AIMessage
import json
from dataclasses import dataclass, asdict
from typing import List, Union

nest_asyncio.apply()


# Define message classes
@dataclass
class BaseMessage:
    content: str
    additional_kwargs: dict = None

    def __post_init__(self):
        if self.additional_kwargs is None:
            self.additional_kwargs = {}


@dataclass
class HumanMessage(BaseMessage):
    type: str = "human"


@dataclass
class AIMessage(BaseMessage):
    type: str = "ai"


# Function to encode messages
def encode_messages(messages: List[Union[HumanMessage, AIMessage]]) -> str:
    """Encode a list of message objects to JSON string"""
    serializable = [asdict(msg) for msg in messages]
    return json.dumps(serializable)


root_dir = Path(__file__).resolve().parent

# Configure page
st.set_page_config(page_title="Multi-Agent Agentic RAG", page_icon="🤖", layout="wide")

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4().hex[:8])

# Create header
st.title("🤖 Multi-Agent Agentic RAG")
st.markdown(
    """
An intelligent system that combines multiple AI agents to provide accurate answers from your documents or the web.

## How it works:
1. Upload your knowledge source (PDF or website URL)
2. Our system indexes your content for fast retrieval
3. Ask questions in natural language
4. The system finds relevant information from your sources or searches the web when needed
"""
)

# Create sidebar for knowledge source input
with st.sidebar:
    st.header("📚 Knowledge Source")
    source_type = st.radio("Select Source Type", ["PDF Document", "Website URL"])

    if source_type == "PDF Document":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        source_path = pdf_file
    else:
        web_url = st.text_input("Enter Website URL", placeholder="https://example.com")
        source_path = web_url

    if st.button("📥 Ingest Data", type="primary"):
        # Initialize payload
        payload = {}

        if source_type == "PDF Document" and pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getbuffer())
                pdf_tmp_file_path = tmp_file.name

            file_name = os.path.basename(pdf_tmp_file_path)
            file_name_without_ext = file_name.split(".")[0]
            mounted_dir = root_dir / "pdf_data"
            shutil.copy(pdf_tmp_file_path, mounted_dir)

            payload = {
                "data_source": "pdf",
                "data_path": str(pdf_tmp_file_path),  # Convert Path to string
                "data_source_url": "",
                "file_name": file_name,
                "file_name_without_ext": file_name_without_ext,
            }

        elif source_type == "Website URL" and web_url:
            payload = {
                "data_source": "web",
                "data_path": "",
                "data_source_url": web_url,
                "file_name": "",
                "file_name_without_ext": "",
            }
        else:
            st.warning("Please provide a valid PDF file or website URL.")
            payload = None

        if payload:
            with st.spinner("Ingesting data into vector store..."):
                try:
                    response = requests.post(
                        VECTORSTORE_URL,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("status") == "Success":
                            st.success("Successfully saved data into vector store.")
                        else:
                            st.error(
                                f"Error: {response_data.get('message', 'Unknown error')}"
                            )
                    else:
                        st.error(f"Error: Status code {response.status_code}")
                except Exception as e:
                    st.error(f"Error occurred during data ingestion: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Rewritten with proper synchronous handling for Streamlit
def stream_response(user_input: str, chat_history: List, thread_id: str):
    """
    Streams the response from the agent using a synchronous approach compatible with Streamlit
    """
    response_parts = []

    # Add the user message to chat history using our custom message class
    chat_history.append(HumanMessage(content=user_input))

    payload = {
        "query": user_input,
        "chat_history": encode_messages(chat_history),
        "thread_id": thread_id,
    }

    try:
        # Make a streaming request
        with requests.post(
            AGENTS_URL,
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code == 200:
                # Process the streaming response in chunks
                for chunk in response.iter_content(
                    chunk_size=None, decode_unicode=True
                ):
                    if chunk:
                        response_parts.append(chunk)
                        # Yield the accumulated response for display
                        yield "".join(response_parts)

                # Add the final response to chat history
                final_response = "".join(response_parts)
                chat_history.append(AIMessage(content=final_response))
                return final_response
            else:
                error_message = f"Error: {response.status_code}"
                yield error_message
                chat_history.append(AIMessage(content=error_message))
                return error_message
    except Exception as e:
        error_message = f"Sorry, an unexpected error occurred: {e}"
        yield error_message
        chat_history.append(AIMessage(content=error_message))
        return error_message


# Process user input and generate response
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat and session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Process and stream the response in a synchronous manner
        for response_chunk in stream_response(
            prompt,
            st.session_state.chat_history,
            st.session_state.thread_id,
        ):
            message_placeholder.markdown(response_chunk)
            full_response = response_chunk

        # Add assistant response to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# Add CSS to hide Streamlit branding
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""",
    unsafe_allow_html=True,
)
