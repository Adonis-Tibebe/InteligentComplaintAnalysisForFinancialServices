# test_complaint_vector_pipeline.py
import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../../src"))
from core.ComplaintVectorPipeline import ComplaintVectorPipeline

@pytest.fixture
def small_test_df():
    return pd.DataFrame({
        "Complaint ID": [101, 102],
        "Normalized Consumer complaint narrative": [
            "Bank delayed my transfer and gave no explanation.",
            "The loan application was rejected without any clear reason."
        ],
        "word_count": [35, 40],
        "Target_Product": ["Money Transfer", "Loan"],
        "Issue": ["Delay", "Rejection"],
        "Sub-issue": ["No communication", "Unclear decision"]
    })

def test_chunking_is_functional(small_test_df):
    pipeline = ComplaintVectorPipeline(chunk_size=30, chunk_overlap=10)
    chunk_df = pipeline.preprocess_and_chunk(small_test_df)
    assert not chunk_df.empty
    assert all(chunk_df["word_count"] > 0)
    assert "chunk_text" in chunk_df.columns

def test_embedding_is_lean(small_test_df):
    pipeline = ComplaintVectorPipeline()
    pipeline.preprocess_and_chunk(small_test_df)
    embeddings = pipeline.model.encode(
        pipeline.chunk_df["chunk_text"].tolist(),
        batch_size=2,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    assert embeddings.shape[0] == len(pipeline.chunk_df)
    assert embeddings.shape[1] == 384

def test_search_returns_valid_chunks(small_test_df):
    pipeline = ComplaintVectorPipeline()
    pipeline.preprocess_and_chunk(small_test_df)
    pipeline.embed_chunks()
    pipeline.build_faiss_index()

    results = pipeline.search("transfer delay", top_k=1)
    assert not results.empty
    assert "chunk_text" in results.columns
    assert "similarity" in results.columns