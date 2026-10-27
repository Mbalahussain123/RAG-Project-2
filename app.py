import hashlib
import json
import time
from io import BytesIO

import numpy as np
import streamlit as st
import torch

from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
FINAL_K = 2
CANDIDATE_K = 10

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Ask questions about a PDF using hybrid search, "
    "reranking, and a Hugging Face language model."
)


# =========================================================
# LOAD EMBEDDINGS AND RERANKER
# =========================================================

@st.cache_resource
def load_retrieval_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    reranker = CrossEncoder(RERANKER_MODEL)

    return embeddings, reranker


# =========================================================
# LOAD LOCAL CHAT MODEL
# This model runs locally after its weights are downloaded.
# =========================================================

@st.cache_resource
def load_chat_model():
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def generate_answer(question, documents):
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        page_number = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source {i}, Page {page_number}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a careful question-answering assistant.

Rules:
1. Answer using only the provided document context.
2. Do not use outside knowledge or make up facts.
3. If the context does not contain enough evidence, reply exactly:
   I don't know based on the provided document.
4. Keep the answer clear and concise.
5. Cite supporting sources using labels such as [Source 1].
6. If the question cannot be answered from the context, do not guess.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"DOCUMENT CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Answer using only the document context."
            ),
        },
    ]

    tokenizer, model = load_chat_model()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove the input prompt from the generated output.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    if not answer:
        return "I don't know based on the provided document."

    return answer


# =========================================================
# READ PDF
# Page numbers are stored as 1-based numbers.
# =========================================================

def extract_pdf_documents(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    documents = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_index + 1,
                        "source": "uploaded PDF",
                    },
                )
            )

    if not documents:
        raise ValueError(
            "No readable text was found. "
            "The PDF may be scanned and require OCR."
        )

    return documents


# =========================================================
# BUILD DOCUMENT INDEX
# =========================================================

@st.cache_resource(show_spinner="Reading PDF and building indexes...")
def build_document_system(pdf_bytes):
    pages = extract_pdf_documents(pdf_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")

    # Give each chunk a stable ID for deduplication.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    embeddings, reranker = load_retrieval_models()

    # Semantic search index.
    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # Keyword search index.
    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return {
        "chunks": chunks,
        "vector_store": vector_store,
        "bm25": bm25,
        "reranker": reranker,
    }


# =========================================================
# HYBRID SEARCH USING RECIPROCAL RANK FUSION (RRF)
# =========================================================

def retrieve_documents(question, system, final_k=FINAL_K):
    start_time = time.perf_counter()

    chunks = system["chunks"]
    vector_store = system["vector_store"]
    bm25 = system["bm25"]
    reranker = system["reranker"]

    candidate_k = min(CANDIDATE_K, len(chunks))

    # 1. Semantic search.
    semantic_results = vector_store.similarity_search_with_score(
        question,
        k=candidate_k,
    )

    semantic_docs = [doc for doc, score in semantic_results]

    # 2. Keyword search.
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    keyword_docs = [chunks[int(i)] for i in bm25_indices]

    # 3. Combine both ranked lists using RRF.
    # RRF does not require semantic and BM25 scores to share a scale.
    fused_scores = {}
    documents_by_id = {}

    def document_key(doc):
        return doc.metadata.get(
            "chunk_id",
            (doc.metadata.get("page"), doc.page_content),
        )

    rrf_constant = 60

    for ranked_docs in (semantic_docs, keyword_docs):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = document_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (
                1.0 / (rrf_constant + rank)
            )
            documents_by_id[key] = doc

    fused_docs = [
        documents_by_id[key]
        for key in sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )
    ]

    # 4. Rerank fused candidates by question relevance.
    pairs = [
        (question, doc.page_content)
        for doc in fused_docs
    ]

    rerank_scores = reranker.predict(pairs)

    ranked_pairs = sorted(
        zip(rerank_scores, fused_docs),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    ranked_docs = [doc for score, doc in ranked_pairs[:final_k]]

    elapsed = time.perf_counter() - start_time

    return ranked_docs, elapsed


# =========================================================
# OPTIONAL RETRIEVAL EVALUATION
#
# Example JSON:
# [
#   {
#     "question": "What is machine learning?",
#     "relevant_pages": [2, 3]
#   }
# ]
# =========================================================

def evaluate_retrieval(dataset, system, k=FINAL_K):
    hits = []
    reciprocal_ranks = []
    latencies = []

    for item in dataset:
        question = item.get("question", "").strip()
        relevant_pages = set(item.get("relevant_pages", []))

        if not question or not relevant_pages:
            continue

        # Request k results so evaluation matches the chosen k.
        docs, latency = retrieve_documents(
            question,
            system,
            final_k=k,
        )

        retrieved_pages = [
            doc.metadata.get("page")
            for doc in docs
        ]

        hit_rank = next(
            (
                rank
                for rank, page in enumerate(retrieved_pages, start=1)
                if page in relevant_pages
            ),
            None,
        )

        hits.append(1 if hit_rank is not None else 0)
        reciprocal_ranks.append(
            1.0 / hit_rank if hit_rank is not None else 0.0
        )
        latencies.append(latency)

    if not hits:
        return None

    return {
        f"Hit Rate@{k}": sum(hits) / len(hits),
        f"MRR@{k}": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "Mean retrieval time (seconds)": (
            sum(latencies) / len(latencies)
        ),
        "Evaluated questions": len(hits),
    }


# =========================================================
# USER INTERFACE
# =========================================================

uploaded_pdf = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"],
)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.caption(f"Document fingerprint: {pdf_hash[:16]}")

    try:
        system = build_document_system(pdf_bytes)

        st.success(
            f"PDF indexed successfully: "
            f"{len(system['chunks'])} chunks created."
        )

        st.subheader("Ask a question")

        question = st.text_input(
            "Enter your question",
            placeholder="What is the main topic of this PDF?",
        )

        if st.button("Get Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching the PDF and generating an answer..."):
                    docs, retrieval_time = retrieve_documents(
                        question.strip(),
                        system,
                    )

                    generation_start = time.perf_counter()
                    answer = generate_answer(question.strip(), docs)
                    generation_time = (
                        time.perf_counter() - generation_start
                    )

                st.subheader("Answer")
                st.write(answer)

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Retrieved chunks",
                    len(docs),
                )
                col2.metric(
                    "Retrieval time",
                    f"{retrieval_time:.2f} s",
                )
                col3.metric(
                    "Generation time",
                    f"{generation_time:.2f} s",
                )

                st.subheader("Retrieved sources")

                for index, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("page", "Unknown")

                    with st.expander(
                        f"Source {index} — Page {page}"
                    ):
                        st.write(doc.page_content)

        st.divider()
        st.subheader("Evaluate retrieval (optional)")

        st.write(
            "Upload a JSON file containing questions and the page "
            "numbers where their answers appear."
        )

        evaluation_file = st.file_uploader(
            "Upload evaluation JSON",
            type=["json"],
            key="evaluation_json",
        )

        evaluation_k = st.slider(
            "Number of results to evaluate (k)",
            min_value=1,
            max_value=min(10, len(system["chunks"])),
            value=min(FINAL_K, len(system["chunks"])),
        )

        if st.button("Run Evaluation"):
            if evaluation_file is None:
                st.warning("Please upload an evaluation JSON file.")
            else:
                try:
                    dataset = json.load(evaluation_file)

                    if not isinstance(dataset, list):
                        raise ValueError(
                            "The JSON root must be a list of questions."
                        )

                    with st.spinner("Evaluating retrieval..."):
                        metrics = evaluate_retrieval(
                            dataset,
                            system,
                            k=evaluation_k,
                        )

                    if metrics is None:
                        st.warning(
                            "No valid evaluation questions were found. "
                            "Each item needs a question and relevant_pages."
                        )
                    else:
                        st.subheader("Evaluation results")
                        st.json(metrics)

                        st.caption(
                            "Hit Rate measures how often at least one "
                            "relevant page was retrieved. MRR rewards "
                            "finding a relevant page near the top. "
                            "These are retrieval metrics, not proof that "
                            "generated answers are factually correct."
                        )

                except (json.JSONDecodeError, ValueError) as error:
                    st.error(f"Invalid evaluation data: {error}")

    except Exception as error:
        st.error(f"Could not process the PDF: {error}")

else:
    st.info("Upload a PDF to begin.")
