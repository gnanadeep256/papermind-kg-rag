import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.vector_retriever import VectorRetriever

def test_vector_retriever_lifecycle() -> None:
    """Verify VectorRetriever resources are loaded correctly."""
    with patch("src.llm.get_embedding_model") as mock_get_model, \
         patch("src.evaluation.embedding_cache.CachedEmbeddingModel") as mock_cached_model, \
         patch("src.vector_retriever.faiss.read_index") as mock_read_index, \
         patch("src.vector_retriever.load_config") as mock_load_config, \
         patch("builtins.open", MagicMock()), \
         patch("json.load") as mock_json_load, \
         patch("os.path.exists") as mock_exists:
         
        mock_load_config.return_value = {"embeddings": {"model_name": "BAAI/bge-small-en-v1.5"}}
         
        mock_exists.return_value = True
        mock_json_load.return_value = [
            {"chunk_id": "c1", "arxiv_id": "a1", "title": "t1", "section": "s1", "page_start": 1, "page_end": 2, "chunk_word_count": 1, "text": "text1"},
            {"chunk_id": "c2", "arxiv_id": "a2", "title": "t2", "section": "s2", "page_start": 3, "page_end": 3, "chunk_word_count": 1, "text": "text2"}
        ]
        
        retriever = VectorRetriever()
        retriever.load()
        
        mock_get_model.assert_called_once_with("BAAI/bge-small-en-v1.5")
        mock_read_index.assert_called_once_with(retriever.index_path)
        assert len(retriever.metadata) == 2

def test_vector_retriever_search() -> None:
    """Verify retriever search query embeds input and maps top-k indices to metadata correctly."""
    with patch("src.llm.get_embedding_model") as mock_get_model, \
         patch("src.evaluation.embedding_cache.CachedEmbeddingModel") as mock_cached_model, \
         patch("src.vector_retriever.faiss.read_index") as mock_read_index, \
         patch("src.vector_retriever.load_config") as mock_load_config, \
         patch("builtins.open", MagicMock()), \
         patch("json.load") as mock_json_load, \
         patch("os.path.exists") as mock_exists:
         
        mock_load_config.return_value = {"embeddings": {"model_name": "BAAI/bge-small-en-v1.5"}}
         
        mock_exists.return_value = True
        mock_json_load.return_value = [
            {"chunk_id": "c1", "arxiv_id": "a1", "title": "t1", "section": "s1", "page_start": 1, "page_end": 2, "chunk_word_count": 1, "text": "text1"},
            {"chunk_id": "c2", "arxiv_id": "a2", "title": "t2", "section": "s2", "page_start": 3, "page_end": 3, "chunk_word_count": 1, "text": "text2"}
        ]
        
        # Configure the mock CachedEmbeddingModel to return query vector
        mock_cached_model.return_value.encode.return_value = [np.zeros(384, dtype=np.float32)]
        
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index
        # Return distances (similarities) and indices
        mock_index.search.return_value = (
            np.array([[0.92, 0.54]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64)
        )
        
        retriever = VectorRetriever()
        results = retriever.search("test query", k=2)
        
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] == pytest.approx(0.92)
        assert results[0]["section"] == "s1"
        assert results[0]["page_start"] == 1
        assert results[0]["page_end"] == 2
        assert results[0]["chunk_word_count"] == 1
        assert results[1]["chunk_id"] == "c2"
        assert results[1]["score"] == pytest.approx(0.54)
        assert results[1]["page_start"] == 3
        assert results[1]["page_end"] == 3
        assert results[1]["chunk_word_count"] == 1
        
        mock_cached_model.return_value.encode.assert_called_once_with(
            ["Represent this sentence for searching relevant passages: test query"],
            normalize_embeddings=True
        )
        mock_index.search.assert_called_once()
