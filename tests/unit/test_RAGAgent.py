# test_rag_agent.py
import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../../src"))
from models.RAG_Pipeline import PromptBuilder, LLMClient, RAGAgent

class DummySearcher:
    def search(self, query, top_k=5, return_full_text=True):
        # Return a mock DataFrame-like object with "chunk_text" column
        return pd.DataFrame({
            "chunk_text": ["The bank delayed my transfer.", "No clear reason for the loan denial."]
        })

class DummyGenerator:
    def __call__(self, prompt, max_new_tokens=250, do_sample=True):
        return [{"generated_text": prompt + " Answer: We found several issues related to transfers."}]

@pytest.fixture
def minimal_rag_agent():
    template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    builder = PromptBuilder(template)
    llm_client = LLMClient(DummyGenerator())
    agent = RAGAgent(searcher=DummySearcher(), prompt_builder=builder, llm_client=llm_client)
    return agent

def test_rag_response(minimal_rag_agent):
    query = "What complaints do users have about money transfers?"
    result = minimal_rag_agent.run(query)
    assert isinstance(result, str)
    assert "issues related to transfers" in result