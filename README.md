# Conversation RAG PDF Chatbot

An AI-powered PDF chatbot that lets users ask questions about their documents, receive context-based answers, and ask follow-up questions in the same conversation.

##📌 About the Project

I built this project to understand how Retrieval-Augmented Generation (RAG) works in real-world applications.

The chatbot searches the PDF for relevant information before generating an answer. It also uses conversation memory to understand follow-up questions, making it easier to interact with documents naturally.

## ✨ Features

- **PDF Processing:** Extract text from PDF documents.
- **Text Chunking:** Split documents into smaller sections.
- **Semantic Search:** Find relevant content using Hugging Face embeddings and FAISS.
- **Keyword Search:** Retrieve information using BM25.
- **Hybrid Search:** Combine semantic and keyword retrieval.
- **Reranking:** Improve the relevance of retrieved passages using a cross-encoder.
- **AI-Generated Answers:** Generate responses using a Hugging Face language model.
- **Conversation Memory:** Maintain conversation context using LangGraph checkpointing.
- **Source References:** Show document references when available.
- **Simple Chat Interface:** Interact with PDFs using Streamlit.

## 🛠️ Technologies Used

- Python
- LangChain
- LangGraph
- Hugging Face Transformers
- Hugging Face Embeddings
- FAISS
- BM25
- Cross-Encoder
- PyTorch
- Streamlit
## 📂 Project Structure
```text
Conversation-RAG-PDF-Chatbot/
│
├─app.py
│   ├── streamlit_app.py
     langchain_rag_sample_10_pages.pdf.pdf
├── faiss_index/       
├── requirements.txt    
├── .gitignore                 
└── README.md                   
```

*Note: This structure is based on the files used in the project. Adjust it to match the actual files and folders in your GitHub repository.*

## ⚙️ How It Works

1. Load a PDF and extract its text.
2. Split the text into smaller chunks.
3. Generate embeddings and build the FAISS index.
4. Retrieve relevant passages using FAISS and BM25.
5. Rerank the retrieved passages.
6. Send the relevant context and conversation history to the language model.
7. Generate an answer and display available source references.
8. Use conversation memory to support follow-up questions.

## 🤖 Models Used
| Component | Model or Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Language model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector database | FAISS |
| Keyword retrieval | BM25 |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Mbalahussain123/RAG-Project-2.git
cd RAG-Project-2
```
### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```
### 4. Run the chatbot

```powershell
python -m streamlit run ".\3.EmbeddedModels\streamlit_app.py"
```
Open the local URL shown in the terminal.
## 📚 What I Learned

- How to build a complete RAG workflow.
- How embeddings and vector search work.
- How to combine keyword and semantic retrieval.
- How reranking helps select relevant content.
- How to connect an LLM with retrieved document context.
- How to manage conversational state using LangGraph.
- How to build an interactive application with Streamlit.

## 🔮 Future Improvements

- Support multiple PDF documents.
- Add persistent conversation history.
- Evaluate retrieval quality with test questions and metrics.
- Improve source verification and answer accuracy.
- Add automated tests and better error handling.

## 👨‍💻 Author

**Balu Hussain**

GitHub: [Mbalahussain123](https://github.com/Mbalahussain123).*

<img width="1600" height="832" alt="IMG-20261029-WA0020" src="https://github.com/user-attachments/assets/0bf5214e-810a-46c2-8749-a7ea97e67784" />

<img width="1239" height="643" alt="IMG-20261029-WA0022" src="https://github.com/user-attachments/assets/a7b651ad-31df-4aa8-bcfd-874c1b6f515f" />

<img width="1150" height="468" alt="IMG-20261029-WA0021" src="https://github.com/user-attachments/assets/27ec0dd0-8626-44a2-9eed-b4eae05ad91a" />




