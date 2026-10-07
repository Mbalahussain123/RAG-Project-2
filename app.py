PDF RAG Chatbot

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load .env
load_dotenv()

# 1. Load PDF file
loader = PDFPlumberLoader("document.pdf")
docs = loader.load()

print("Pages:", len(docs))

# 2. Split PDF text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))

# 3. Create Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 4. Create FAISS vector store
vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)

# 5. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# 6. Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.1
)

# 7. Prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the context below.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# 8. Ask question
question = input("Ask your question: ")

# 9. Retrieve relevant documents
retrieved_docs = retriever.invoke(question)

# 10. Create context
context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

# 11. Create final prompt
final_prompt = prompt.invoke({
    "context": context,
    "question": question
})

# 12. Generate answer
response = llm.invoke(final_prompt.to_string())

# 13. Print answer
print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response)
