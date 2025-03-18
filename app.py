import streamlit as st
import uuid
import tempfile
import requests
from pathlib import Path
from constants import AGENTS_URL, VECTORSTORE_URL, AGENTS_TASK_ACTION_HISTORY_URL
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

# Configure page with a more modern layout
st.set_page_config(
    page_title="IntelliRAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling - add styling for action history
st.markdown(
    """
<style>
    /* Theme detection using CSS variables that Streamlit injects */
    :root {
        --text-color: #1E3A8A;
        --subtext-color: #4B5563;
        --background-color: #F9FAFB;
        --sidebar-bg: #F3F4F6;
        --card-bg: #EFF6FF;
        --card-border: #3B82F6;
        --user-msg-bg: #E3F2FD;
        --ai-msg-bg: #F3F4F6;
        --border-color: #E5E7EB;
        --action-history-bg: #F9FAFB;
    }

    /* Dark theme adjustments */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #93C5FD;
            --subtext-color: #D1D5DB;
            --background-color: #1F2937;
            --sidebar-bg: #111827;
            --card-bg: #1E3A8A;
            --card-border: #60A5FA;
            --user-msg-bg: #1E40AF;
            --ai-msg-bg: #374151;
            --border-color: #4B5563;
            --action-history-bg: #111827;
        }
    }

    /* Streamlit's dark theme adjustment */
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --text-color: #93C5FD;
        --subtext-color: #D1D5DB;
        --background-color: #1F2937;
        --sidebar-bg: #111827;
        --card-bg: #1E3A8A;
        --card-border: #60A5FA;
        --user-msg-bg: #1E40AF;
        --ai-msg-bg: #374151;
        --border-color: #4B5563;
        --action-history-bg: #111827;
    }

    /* Core styling using variables */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: var(--text-color);
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .subheader {
        font-size: 1.2rem;
        color: var(--subtext-color);
        margin-bottom: 2rem;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: 600;
        height: 3em;
    }

    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid var(--border-color);
    }

    .chat-message-user {
        background-color: var(--user-msg-bg);
        padding: 10px 15px;
        border-radius: 18px 18px 2px 18px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
    }

    .chat-message-ai {
        background-color: var(--ai-msg-bg);
        padding: 10px 15px;
        border-radius: 18px 18px 18px 2px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
    }

    .action-history-container {
        margin-left: 20px;
        margin-bottom: 20px;
        width: 80%;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4().hex[:8])
if "action_history" not in st.session_state:
    st.session_state.action_history = {}

# Create enhanced header
st.markdown(
    '<h1 class="main-header">🧠 IntelliRAG Assistant</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="subheader">Your intelligent research companion powered by multiple AI agents</p>',
    unsafe_allow_html=True,
)

# Create columns for the main layout
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### 📚 Knowledge Source")

    source_type = st.radio(
        "Select Source Type:", ["PDF Document", "Website URL"], key="source_type_radio"
    )

    if source_type == "PDF Document":
        pdf_file = st.file_uploader(
            "Upload PDF:", type=["pdf"], help="Maximum file size: 200MB"
        )
        source_path = pdf_file

        if pdf_file:
            st.success(f"Selected: {pdf_file.name}")
    else:
        web_url = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter a complete URL including https://",
        )
        source_path = web_url

        if web_url:
            st.info(f"URL ready: {web_url}")

    ingest_button = st.button("📥 Ingest Knowledge Source", type="primary")

    if ingest_button:
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
                "data_path": str(pdf_tmp_file_path),
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
            st.warning("⚠️ Please provide a valid PDF file or website URL.")
            payload = None

        if payload:
            with st.spinner("🔄 Ingesting data into vector store..."):
                try:
                    response = requests.post(
                        VECTORSTORE_URL,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("status") == "Success":
                            st.success("✅ Successfully saved data into vector store!")
                        else:
                            st.error(
                                f"❌ Error: {response_data.get('message', 'Unknown error')}"
                            )
                    else:
                        st.error(f"❌ Error: Status code {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error occurred during data ingestion: {e}")

    # Add information about the system
    st.markdown("### 🔍 How It Works")

    st.markdown(
        """
    1. Upload your knowledge source (PDF or website URL)
    2. Our system indexes your content for fast retrieval
    3. Ask questions in natural language
    4. The system finds relevant information from your sources or searches the web when needed
    5. View the agent's actions by clicking "View Actions" below each response
    """
    )

    # Add system status indicator
    st.markdown("### 🔄 System Status")

    st.info(f"🧵 Session ID: {st.session_state.thread_id}")

    st.markdown("</div>", unsafe_allow_html=True)

# Main chat area in the first column
with col1:
    # Display features
    if not st.session_state.messages:
        st.markdown("### 🌟 Features")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📊 Document Analysis")
            st.markdown(
                "Upload PDFs or provide websites to extract insights and answer questions about your content."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### 🔎 Smart Search")
            st.markdown(
                "Our system finds the most relevant information from your sources."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("#### 🤖 Multiple AI Agents")
            st.markdown(
                "Specialized agents work together to provide comprehensive answers."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### 💬 Natural Conversation")
            st.markdown(
                "Chat naturally with the system about complex topics in your documents."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Function to fetch action history
    def get_action_history():
        """Fetch action history for a specific message if possible"""
        try:
            # For now, just fetching the latest history:
            response = requests.get(AGENTS_TASK_ACTION_HISTORY_URL)
            if response.status_code == 200:
                return response.json().get("history", "No action history available")
            else:
                return f"Failed to fetch action history: {response.status_code}"
        except Exception as e:
            return f"Exception occurred fetching action history: {str(e)}"

    # Display chat messages with enhanced styling
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(
                f'<div style="text-align: right;"><div class="chat-message-user">{message["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="text-align: left;"><div class="chat-message-ai">{message["content"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Add action history dropdown after each AI message
            # Create unique key for this expander and message
            expander_key = f"action_history_{idx}"
            message_key = f"message_{idx}"

            # Create expander (dropdown) for action history
            with st.expander("🔍 View Actions", expanded=False):
                # Check if we already fetched this specific message's action history
                if message_key not in st.session_state.action_history:
                    with st.spinner("Loading action history..."):
                        # Get action history for this specific message/response
                        action_history = get_action_history()
                        # Store it in the dictionary with the message-specific key
                        st.session_state.action_history[message_key] = action_history

                # Display the saved action history for this specific message
                st.markdown(
                    '<div class="action-history-content">', unsafe_allow_html=True
                )
                st.code(st.session_state.action_history[message_key], language="text")
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Process user input and generate response
if prompt := st.chat_input("Ask me anything about your document or URL..."):
    # Add user message to chat and session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message (this will show after rerun)
    with col1:
        st.markdown(
            f'<div style="text-align: right;"><div class="chat-message-user">{prompt}</div></div>',
            unsafe_allow_html=True,
        )

    # Display assistant response
    with col1:
        message_placeholder = st.empty()
        full_response = ""

        # Rewritten stream_response function
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

        # Process and stream the response in a synchronous manner
        for response_chunk in stream_response(
            prompt,
            st.session_state.chat_history,
            st.session_state.thread_id,
        ):
            message_placeholder.markdown(
                f'<div style="text-align: left;"><div class="chat-message-ai">{response_chunk}</div></div>',
                unsafe_allow_html=True,
            )
            full_response = response_chunk

        # Add assistant response to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # Get and store the action history immediately
        message_idx = len(st.session_state.messages) - 1
        message_key = f"message_{message_idx}"
        st.session_state.action_history[message_key] = get_action_history()

        # Display the action history expander for the current message
        with st.expander("🔍 View Actions", expanded=False):
            st.code(st.session_state.action_history[message_key], language="text")
            st.markdown("</div>", unsafe_allow_html=True)
