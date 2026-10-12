
# RAG-Based Document Question Answering System

## Overview
This project is a Retrieval-Augmented Generation (RAG) application that answers user questions based on information available in PDF documents. It retrieves relevant text from the document and uses a language model to generate answers.

## Features
- Load and extract text from PDF documents.
- Split text into smaller chunks.
- Generate embeddings using Hugging Face.
- Store and search embeddings using FAISS.
- Retrieve relevant document chunks.
- Generate answers using a Hugging Face language model.
- Display source PDF page numbers.
- Handle missing PDFs and common errors.

## Technologies Used
- Python
- LangChain
- Hugging Face
- FAISS
- Sentence Transformers
- PyPDFLoader

##Project Workflow
1. Load the PDF document.
2. Split the extracted text into chunks.
3. Convert chunks into embeddings.
4. Store embeddings in FAISS.
5. Retrieve relevant chunks for the user's question.
6. Generate an answer using the retrieved context.
7. Display the answer and source page numbers.

## Project Structure

RAG-Project-2/
├── app.py
├── .gitignore
├── README.md
├── Requirements.txt
├── langchain_rag_sample_10_pages.pdf
└── screenshots/

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Mbalahussain123/RAG-Project-2.git
cd RAG-Project-2
```

###2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Configure the API token
Create a `.env` file in the project folder:

```text
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

Replace the placeholder with your Hugging Face token.

### 4. Run the application
```bash
python app.py
```

## Example Questions
- What is Python?
- What is java programming?
- What is RAG?
- what is Langchain

## Expected Output
The application displays an answer based on retrieved PDF content and, when available, the source PDF page numbers.

## Future Improvements
- Support multiple PDF documents.
- Improve retrieval accuracy.
- Add a web-based user interface.
- Evaluate answer quality using test questions.
- Improve handling of questions not answered by the document.

## Security
Never upload API tokens or secrets to GitHub. Keep the `.env` file private and add it to `.gitignore`.
