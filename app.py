import streamlit as st
from engine import CudaCortex

# --- App Configuration ---
st.set_page_config(
    page_title="CUDA-Cortex Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
# Using session_state to store the engine and chat history
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
@st.cache_resource
def get_engine():
    """Load and cache the CudaCortex engine."""
    engine = CudaCortex()
    with st.spinner("🔧 Setting up the AI engine... This may take a moment."):
        engine.setup()
    return engine

# --- Sidebar ---
with st.sidebar:
    st.title("🚀 CUDA-Cortex Agent")
    st.markdown("This app is an AI-powered chatbot that can answer technical questions about the NVIDIA CUDA-Samples GitHub repository.")
    st.markdown("It uses a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware answers.")

    if st.button("Initialize AI Engine", use_container_width=True):
        st.session_state.engine = get_engine()
        st.success("Engine initialized successfully!")

# --- Main Chat Interface ---
st.header("Chat with the CUDA Expert")

if st.session_state.engine is None:
    st.info("Please initialize the AI engine from the sidebar to begin.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about CUDA..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = st.session_state.engine.ask(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})