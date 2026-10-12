RAG-Based Document Question Answering System
Overview
This project is a Retrieval-Augmented Generation (RAG) application that answers user questions based on information available in PDF documents. It retrieves relevant text from the document and uses a language model to generate answers.

Features
Load and extract text from PDF documents.
Split text into smaller chunks.
Generate embeddings using Hugging Face.
Store and search embeddings using FAISS.
Retrieve relevant document chunks.
Generate answers using a Hugging Face language model.
Display source PDF page numbers.
Handle missing PDFs and common errors.
Technologies Used
Python
LangChain
Hugging Face
FAISS
Sentence Transformers
PyPDFLoader
##Project Workflow

Load the PDF document.
Split the extracted text into chunks.
Convert chunks into embeddings.
Store embeddings in FAISS.
Retrieve relevant chunks for the user's question.
Generate an answer using the retrieved context.
Display the answer and source page numbers.
Project Structure
RAG-Project-2/ ├── app.py ├── .gitignore ├── README.md ├── Requirements.txt ├── langchain_rag_sample_10_pages.pdf └── screenshots/

Installation
1. Clone the repository
git clone https://github.com/Mbalahussain123/RAG-Project-2.git
cd RAG-Project-2
###2. Install dependencies

pip install -r Requirements.txt
3. Configure the API token
Create a .env file in the project folder:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
Replace the placeholder with your Hugging Face token.

4. Run the application
python app.py
Example Questions
What is Python?
What is java programming?
What is RAG?
what is Langchain
Expected Output
The application displays an answer based on retrieved PDF content and, when available, the source PDF page numbers.

Future Improvements
Support multiple PDF documents.
Improve retrieval accuracy.
Add a web-based user interface.
Evaluate answer quality using test questions.
Improve handling of questions not answered by the document.

Results
<img width="1185" height="198" alt="WhatsApp Image 2026-10-12 at 10 37 07 AM" src="https://github.com/user-attachments/assets/12ccbb8d-2d52-4bb3-aae7-6581cba2e490" />
<img width="1374" height="571" alt="WhatsApp Image 2026-10-12 at 10 37 49 AM" src="https://github.com/user-attachments/assets/f86559db-b7a0-4b67-b2ee-cd9806b7aa8d" />



