import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import faiss
import numpy as np

# Add local streamlit directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "streamlit")))

from chat_modules.chat_retriever import TemporaryRetriever
from chat_modules.chat_service import extract_metadata_from_text
from utils import renumber_citations, highlight_keywords
from src.retriever import Citation

def test_temporary_retriever_retrieve():
    # Construct a mock FAISS index
    index = faiss.IndexFlatIP(384)
    index.add(np.zeros((1, 384), dtype=np.float32))
    
    chunks = [
        {
            "chunk_id": "temp_chunk_0",
            "title": "Mock Paper",
            "arxiv_id": "1234.5678",
            "section": "Abstract",
            "page_start": 1,
            "page_end": 1,
            "chunk_word_count": 10,
            "text": "This is a temporary chunk of text."
        }
    ]
    
    retriever = TemporaryRetriever(index, chunks, {"title": "Mock Paper", "arxiv_id": "1234.5678"})
    
    # Mock embedding model encoding
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
    retriever.model = mock_model
    
    res = retriever.retrieve("test query", category="methodology")
    
    assert res.query == "test query"
    assert len(res.vector_context) == 1
    assert res.vector_context[0].title == "Mock Paper"
    assert len(res.citations) == 1
    assert res.retrieval_metadata["routing_strategy"] == "temporary_in_memory"

def test_citation_renumbering():
    answer_text = "LoRA is great [Citation 2]. Prefix Tuning is also good [Citation 1]."
    citations = [
        Citation(paper_title="P1", arxiv_id="1", section="sec", page_start=1, page_end=1, chunk_id="c1", similarity_score=0.8, graph_bonus=0.0, combined_score=0.8),
        Citation(paper_title="P2", arxiv_id="2", section="sec", page_start=1, page_end=1, chunk_id="c2", similarity_score=0.9, graph_bonus=0.0, combined_score=0.9)
    ]
    
    renumbered_text, new_citations = renumber_citations(answer_text, citations)
    
    # [Citation 2] appears first, so it becomes [1] (pointing to P2)
    # [Citation 1] appears second, so it becomes [2] (pointing to P1)
    assert renumbered_text == "LoRA is great [1]. Prefix Tuning is also good [2]."
    assert len(new_citations) == 2
    assert new_citations[0].paper_title == "P2"
    assert new_citations[1].paper_title == "P1"

def test_highlight_keywords():
    text = "This paper details Prefix Tuning and LoRA."
    highlighted = highlight_keywords(text, "prefix lora")
    assert '<span style="background-color: rgba(99, 102, 241, 0.15); color: #4f46e5; padding: 2px 4px; border-radius: 4px; font-weight: 500;">Prefix</span>' in highlighted or 'prefix' in highlighted.lower()
    assert '<span style="background-color: rgba(99, 102, 241, 0.15); color: #4f46e5; padding: 2px 4px; border-radius: 4px; font-weight: 500;">LoRA</span>' in highlighted or 'lora' in highlighted.lower()

@patch("chat_modules.chat_service.call_llm")
def test_extract_metadata(mock_call_llm):
    mock_call_llm.return_value = '{"title": "Deep Learning", "arxiv_id": "1111.2222", "authors": ["John Doe"], "abstract": "Test abstract", "categories": ["cs.AI"]}'
    res = extract_metadata_from_text("Dummy first page text")
    assert res["title"] == "Deep Learning"
    assert res["arxiv_id"] == "1111.2222"
    assert res["authors"] == ["John Doe"]
