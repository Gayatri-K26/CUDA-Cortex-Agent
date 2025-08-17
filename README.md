# CUDA-Cortex-Agent

An AI-powered chatbot that answers technical questions about the NVIDIA CUDA-Samples repository. This project uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers directly from source code and documentation. Uses 4-bit quantization to run a powerful 7-billion parameter model efficiently.

This project was built to serve as an intelligent assistant for developers working with NVIDIA CUDA. Instead of manually searching through hundreds of files in the official cuda-samples repository, a developer can simply ask a question in natural language. The agent retrieves relevant code snippets and documentation and uses a Large Language Model to synthesize a precise answer.

**Tech Stack**
AI/ML: PyTorch, Hugging Face Transformers, LangChain

Model: mistralai/Mistral-7B-Instruct-v0.2 (with 4-bit quantization via bitsandbytes)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS (Facebook AI Similarity Search)

Application Framework: Streamlit

Prototyping Environment: Google Colab (with NVIDIA T4 GPU)
