
#PDF-Based RAG System Using Hugging Face and LangChain

📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents and receive answers based on the document's content.

It combines semantic search, keyword search, and a reranking model to retrieve relevant information before generating an answer using a locally running Hugging Face language model.

The system also provides source citations, page numbers, and execution-time metrics.

✨ Features

PDF Loading: Extract text from PDF documents using PyPDFLoader.

Text Chunking: Split documents into smaller chunks using RecursiveCharacterTextSplitter.

Semantic Search: Retrieve relevant content using Hugging Face embeddings and FAISS.

Keyword Search: Use BM25Retriever to find relevant text based on keywords.

Reranking: Improve retrieval relevance using a Cross-Encoder model.

Local LLM: Generate answers using Qwen2.5-1.5B-Instruct.

Source Citations: Display the PDF filename and page number for retrieved sources.

Unknown Answer Handling: Instruct the model to respond with "I don't know based on the provided document." when the answer is not supported by the context.

Performance Metrics: Measure retrieval time and total response time.

🛠️ Technologies Used

Python

LangChain

Hugging Face Transformers

Hugging Face Sentence Transformers

FAISS

BM25

PyPDFLoader

PyTorch

Cross-Encoder

python-dotenv

📂 Project Structure

langchain_models/
│
├──character_loader.py
├── langchain_rag_sample_10_pages.pdf.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── .env
├── requirements.txt
└└── README.md── README.md

Note: The FAISS index files are generated when the index is saved. The .env file is optional unless you use environment variables elsewhere in your project.

⚙️ Models Used

Component

Model

Embeddings

BAAI/bge-small-en-v1.5

Language Model

Qwen/Qwen2.5-1.5B-Instruct

Reranker

cross-encoder/ms-marco-MiniLM-L-6-v2

Vector Database

FAISS

🚀 Installation and Setup

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git

Replace YOUR-USERNAME and YOUR-REPOSITORY with your GitHub details.

2. Navigate to the Project Folder

cd YOUR-REPOSITORY

3. Create a Virtual Environment

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

4. Install Dependencies

Install the required packages:

pip install langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers transformers torch pypdf pdfplumber rank-bm25 python-dotenv

5. Add Your PDF

Place your PDF file in the 3.EmbeddedModels directory.

Ensure the PDF path in your Python script matches the actual filename:

PDF_PATH = r"3.EmbeddedModels\langchain_rag_sample_10_pages.pdf.pdf"

6. Run the Application

From the project root directory, run:

python .\3.EmbeddedModels\character_loader.py

The first run may take longer because the embedding, reranker, and language models must be downloaded. A compatible local machine with sufficient memory is required.

💬 Example Usage

After running the application, enter a question:

Ask a question (or type exit): What is machine learning?

The application retrieves relevant PDF content and generates an answer based on the available context.

Example output format:

Answer:
Machine learning is ... [Source 1]

Sources:
[Source 1] langchain_rag_sample_10_pages.pdf.pdf - Page 2

Metrics:
Retrieval time: 0.250 seconds
Total time: 2.800 seconds

The answer and timing values above are illustrative; actual results depend on your PDF, hardware, and models.

Handling Questions Outside the PDF

If the retrieved context does not support an answer, the application is instructed to respond:

I don't know based on the provided document.

The current implementation relies on the language model's compliance with the prompt, so this behavior is not guaranteed for every question.

🔍 How the RAG Pipeline Works

Load the PDF document.

Split the extracted text into chunks.

Generate embeddings for the chunks.

Store and retrieve vectors using FAISS.

Retrieve keyword matches using BM25.

Combine and rerank the retrieved results.

Build a context containing the relevant document text.

Generate an answer using the local Hugging Face language model.

Display the answer, source pages, and execution metrics.

📊 Benefits

Answers questions using your own PDF documents.

Combines semantic and keyword-based retrieval.

Improves relevance through reranking.

Runs the language model locally after downloading it.

Does not require a paid hosted LLM API for generation.

Provides source information to help users verify answers.

Note: Initial model downloads require an internet connection. Local execution still requires adequate system resources, and the embedding and reranking models also consume memory.

🔮 Future Improvements

Support multiple PDF documents.

Add a Streamlit user interface.

Improve answer verification to reduce hallucinations.

Add conversation history.

Support document upload through a web interface.

Improve retrieval evaluation and performance monitoring.

<img width="670" height="370" alt="IMG-20261014-WA0009" src="https://github.com/user-attachments/assets/2fc07de5-fe77-4259-8a18-7d7b67a6ef01" />
<img width="670" height="370" alt="IMG-20261014-WA0009" src="https://github.com/user-attachments/assets/c4362fad-292d-4a0e-8dc4-80275f9510c2" />


