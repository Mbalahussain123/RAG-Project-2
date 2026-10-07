PDF RAG Chatbot

Project Description

This project is a RAG (Retrieval Augmented Generation) chatbot that answers questions from a PDF document.

Technologies

- Python
- LangChain
- Hugging Face
- PDFPlumber
- FAISS
- Sentence Transformers
- RAG

RAG Flow
PDF → PDFPlumberLoader → Text Splitting → Hugging Face Embeddings → FAISS → Retrieval → Hugging Face LLM → Answer

How to Run
Install the required packages:
pip install -r requirements.txt

Run the application:
python app.py

Features
- Loads a PDF document
- Extracts text from the PDF
- Splits text into smaller chunks
- Creates Hugging Face embeddings
- Stores embeddings in FAISS
- Retrieves relevant information
- Generates answers using a Hugging Face LLM
