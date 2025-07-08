# rag_interface.py
import streamlit as st
import os
import sys
import multiprocessing
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

sys.path.append(os.path.abspath("../../"))
from models.RAG_Pipeline import PromptBuilder, LLMClient_For_Llama, RAGAgent
from core.ComplaintVectorPipeline import ComplaintVectorPipeline

# Allow module imports
sys.path.append(os.path.abspath("../../"))

# Thread control
num_threads = min(8, multiprocessing.cpu_count())

# --- Prompt Template ---
template = (
    "You are a financial complaints analyst for CrediTrust. Your role is to answer user questions about customer complaints using only the retrieved complaint excerpts provided below.\n\n"
    "These excerpts may include complaints across multiple financial products (e.g., BNPL, credit cards, loans). If the retrieved context offers insight into the question, summarize the key issues clearly and concisely and in a structured manner using bullet points and the like. "
    "If the context does not contain enough information, say: “The context does not provide enough information to answer this question.”\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="RAG QA Interface", layout="centered")

# --- Cached Loaders ---
@st.cache_resource
def load_model_cached():
    return Llama(
        model_path="../../../models/mistral_condensed/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=1024,
        n_threads=num_threads
    )

@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    return faiss.read_index("../../../vector_store/complaints_index.faiss")

@st.cache_resource
def load_dataframes():
    chunk_df = pd.read_csv("../../../data/processed/chunked_complaints.csv", low_memory=False, dtype=str)
    filtered_df = pd.read_csv("../../../data/processed/final_filtered_complaints.csv", low_memory=False, dtype=str)
    return chunk_df, filtered_df

@st.cache_resource
def load_rag_agent():
    llm = load_model_cached()
    model = load_embeddings_model()
    index = load_index()
    chunk_df, filtered_df = load_dataframes()

    pipeline = ComplaintVectorPipeline()
    pipeline.model = model
    pipeline.index = index
    pipeline.chunk_df = chunk_df
    pipeline.filtered_df = filtered_df

    builder = PromptBuilder(template)
    llm_client = LLMClient_For_Llama(llm)
    return RAGAgent(pipeline, builder, llm_client)

# --- Load RAG Agent ---
rag_agent = load_rag_agent()

# --- UI Header ---
st.title("🔍 RAG Question Answering App")
st.markdown("Ask questions based on your document corpus and get contextual answers.")

# --- Initialize Session State ---
if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

# --- Input and Buttons ---
st.text_input(
    "Enter your question here:",
    placeholder="e.g., What issues do users report about savings accounts?",
    key="query"
)

def clear_inputs():
    st.session_state["query"] = ""
    st.session_state["answer"] = ""
    
col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("Ask")

with col2:
    st.button("Clear", on_click=clear_inputs)

# --- Logic ---
if submit and st.session_state.query:
    with st.spinner("🤔 Generating answer..."):
        st.session_state.answer = rag_agent.run(st.session_state.query)
elif submit and not st.session_state.query:
    st.warning("Please enter a question before submitting.")


# --- Display Answer ---
if st.session_state.answer:
    st.markdown("### 📘 Answer")
    st.write(st.session_state.answer)
