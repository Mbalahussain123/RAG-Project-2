
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

## Project Workflow
1. Load the PDF document.
2. Split the extracted text into chunks.
3. Convert chunks into embeddings.
4. Store embeddings in FAISS.
5. Retrieve relevant chunks for the user's question.
6. Generate an answer using the retrieved context.
7. Display the answer and source page numbers.

## Project Structure
```text
RAG-Project-2/
├── app.py
├── .gitignore
├── README.md
├── Requirements.txt
├── langchain_rag_sample_10_pages.pdf
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Mbalahussain123/RAG-Project-2.git
cd RAG-Project-2
```

### 2. Install dependencies
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
- What is LangChain?
- What is RAG?
- What are embeddings?
- What is FAISS?

## Expected Output
The application displays an answer based on retrieved PDF content and, when available, the source PDF page numbers.

## Future Improvements
- Support multiple PDF documents.
- Improve retrieval accuracy.
- Add a web-based user interface.
- Evaluate answer quality using test questions.
- Improve handling of questions not answered by the document.

- <img width="1185" height="198" alt="IMG-20261012-WA0029" src="https://github.com/user-attachments/assets/6e1cdaae-8529-4095-8acb-69b8454e6760" />
<img width="1374" height="571" alt="IMG-20261012-WA0030" src="https://github.com/user-attachments/assets/c467b716-d761-403a-ab6c-1ee79f3ce098" />


