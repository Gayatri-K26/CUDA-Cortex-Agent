import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class CudaCortex:
    """
    The main engine for the CUDA-Cortex Agent.
    Handles model loading, vector store management, and the RAG chain.
    """
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_id = model_id
        self.vector_store_path = "faiss_index_cuda_samples"

        # Initialize all components to None
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.chain = None

    def _initialize_embeddings(self):
        """Initializes the sentence transformer embeddings model."""
        if self.embeddings is None:
            print("Initializing embeddings model...")
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            print("Embeddings model initialized.")

    def _initialize_llm(self):
        """Initializes the Large Language Model with 4-bit quantization."""
        if self.llm is None:
            print(f"Initializing LLM: {self.model_id}...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("LLM initialized.")

    def _create_or_load_vector_store(self, data_path="./cuda-samples/"):
        """Creates a new vector store or loads an existing one."""
        if os.path.exists(self.vector_store_path):
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded.")
        else:
            print("Creating new vector store. This may take a few minutes...")
            print("Loading documents...")
            documents = self._load_documents_from_directory(data_path)

            print(f"Loaded {len(documents)} documents. Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)

            print(f"Split into {len(docs)} chunks. Creating embeddings...")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)

            print("Saving vector store...")
            self.vector_store.save_local(self.vector_store_path)
            print("Vector store created and saved.")

    def _load_documents_from_directory(self, directory):
        """Helper function to load processable documents from a directory."""
        documents = []
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Data directory not found at {directory}. Make sure to clone the NVIDIA/cuda-samples repo.")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.cu', '.cpp', '.h', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")
        return documents

    def setup(self):
        """
        A single method to set up the entire engine.
        This ensures components are loaded in the correct order.
        """
        self._initialize_llm()
        self._initialize_embeddings()
        self._create_or_load_vector_store()

        prompt_template = """
        You are an expert NVIDIA CUDA programming assistant. Use the following pieces of context from the CUDA-Samples repository to answer the user's question. If the context does not contain the answer, state that you cannot answer based on the provided information. Do not make up an answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.chain = LLMChain(llm=self.llm, prompt=PROMPT)
        print("Engine setup complete.")

    def ask(self, query):
        """
        Asks a question to the RAG chain.
        """
        if not self.chain or not self.vector_store:
            raise RuntimeError("Engine is not set up. Please run setup() first.")

        # Retrieve relevant docs
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)

        # Format the context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Get the result from the chain
        result = self.chain.invoke({"context": context, "question": query})
        return result['text']