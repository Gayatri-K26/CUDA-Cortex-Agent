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

**🚀 Setup and Installation**
To run this project locally, you will need a machine with an NVIDIA GPU and the CUDA Toolkit installed.

Step 1: Clone this Repository

Step 2: Download the Knowledge Base Data
The AI's knowledge comes from the official NVIDIA samples. Clone that repository inside the project folder.

git clone https://github.com/NVIDIA/cuda-samples.git

Step 3: Install Dependencies
Install all the required Python packages using the requirements.txt file.

Step 4: Authenticate with Hugging Face
This project requires access to the Mistral-7B model, which is a gated repository. You will need to authenticate your machine with your Hugging Face account.

Get Access on the Website:

Create a free account on HuggingFace.co.

Visit the Mistral-7B-Instruct model page and agree to the terms of use. You must do this first to be granted access.

Get Your Access Token:

In your Hugging Face account settings, go to the "Access Tokens" section.

Create a new token with at least "read" permissions. Copy this token.

Log in from Your Terminal:

Run the following command in your terminal:

huggingface-cli login

Paste your access token when prompted. This will securely save your credentials on your local machine.

Step 5: Run the Application
You are now ready to launch the app :))

streamlit run app.py

The application will be available at http://localhost:8501. The first launch may take several minutes to build the vector database from the CUDA samples. Subsequent launches will be much faster.
