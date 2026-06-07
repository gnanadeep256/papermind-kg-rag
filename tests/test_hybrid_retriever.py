import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.hybrid_retriever import HybridRetriever, Citation, RetrievalResult, RetrievalExplanation

@pytest.fixture
def mock_retrievers():
    """Fixture to mock VectorRetriever and Neo4jKGRetriever classes."""
    with patch("src.hybrid_retriever.VectorRetriever") as mock_vector_class, \
         patch("src.hybrid_retriever.Neo4jKGRetriever") as mock_kg_class:
         
        mock_vector = MagicMock()
        mock_kg = MagicMock()
        
        # Mock model inside vector retriever
        mock_model = MagicMock()
        mock_vector.model = mock_model
        # Return a flat array of zeros for dummy encoding
        mock_model.encode.return_value = np.zeros((1, 384))
        
        mock_vector_class.return_value = mock_vector
        mock_kg_class.return_value = mock_kg
        
        yield mock_vector, mock_kg

def test_hybrid_retriever_load(mock_retrievers):
    """Verify retriever component loading and entity caching."""
    mock_vector, mock_kg = mock_retrievers
    mock_driver = MagicMock()
    mock_kg.driver = mock_driver
    
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Mock Neo4j cache response
    mock_session.run.return_value = [
        {"name": "USAD 2.0", "type": "Method", "entity_id": "usad 2.0"},
        {"name": "Unitree G1", "type": "Concept", "entity_id": "unitree g1"},
        {"name": "A mock title", "type": "Paper", "entity_id": "123.456"}
    ]
    
    # Set mock encode to return embeddings for 3 entities
    mock_vector.model.encode.return_value = np.random.rand(3, 384)
    
    retriever = HybridRetriever()
    retriever.load()
    
    mock_vector.load.assert_called_once()
    mock_kg.connect.assert_called_once()
    assert len(retriever.entity_cache) == 3
    assert retriever.entity_cache["usad 2.0"]["entity_id"] == "usad 2.0"
    assert retriever.entity_cache["usad 2.0"]["type"] == "Method"
    assert "embedding" in retriever.entity_cache["usad 2.0"]

def test_hybrid_retriever_intent_detection(mock_retrievers):
    """Verify query intent classification logic."""
    _, _ = mock_retrievers
    retriever = HybridRetriever()
    
    entities_paper = [{"name": "Mock Paper", "type": "Paper", "entity_id": "p1"}]
    entities_method = [{"name": "Mock Method", "type": "Method", "entity_id": "m1"}]
    entities_dataset = [{"name": "Mock Dataset", "type": "Dataset", "entity_id": "d1"}]
    
    # 1. Entity-type intent detection
    assert retriever.detect_intent("Explain something", entities_paper) == "paper"
    assert retriever.detect_intent("Run evaluation", entities_method) == "method"
    assert retriever.detect_intent("ImageNet", entities_dataset) == "dataset"
    
    # 2. Keyword-based intent detection
    assert retriever.detect_intent("Summarize this paper", []) == "paper"
    assert retriever.detect_intent("How does the algorithm work?", []) == "method"
    assert retriever.detect_intent("What datasets are used?", []) == "dataset"
    
    # 3. Default fallback
    assert retriever.detect_intent("How do robots learn general manipulation tasks?", []) == "research"

def test_hybrid_retriever_evidence_diversity(mock_retrievers):
    """Verify that chunks contributed by a single paper are capped correctly."""
    mock_vector, mock_kg = mock_retrievers
    
    # Mock FAISS returning 5 chunks from the same paper, and 2 from another
    mock_vector.search.return_value = [
        {"chunk_id": "p1_chunk_0", "arxiv_id": "p1", "title": "Paper One", "section": "Abstract", "page_start": 1, "page_end": 1, "text": "text1", "score": 0.9, "chunk_word_count": 5},
        {"chunk_id": "p1_chunk_1", "arxiv_id": "p1", "title": "Paper One", "section": "Intro", "page_start": 1, "page_end": 2, "text": "text2", "score": 0.85, "chunk_word_count": 5},
        {"chunk_id": "p1_chunk_2", "arxiv_id": "p1", "title": "Paper One", "section": "Method", "page_start": 3, "page_end": 3, "text": "text3", "score": 0.8, "chunk_word_count": 5},
        {"chunk_id": "p2_chunk_0", "arxiv_id": "p2", "title": "Paper Two", "section": "Abstract", "page_start": 1, "page_end": 1, "text": "text4", "score": 0.75, "chunk_word_count": 5},
        {"chunk_id": "p2_chunk_1", "arxiv_id": "p2", "title": "Paper Two", "section": "Intro", "page_start": 1, "page_end": 2, "text": "text5", "score": 0.7, "chunk_word_count": 5}
    ]
    
    retriever = HybridRetriever()
    retriever.entity_cache = {}
    mock_kg.get_paper_subgraph.return_value = {"nodes": [], "relationships": []}
    
    with patch.object(retriever, "get_connected_entities_for_papers", return_value=[]):
        res = retriever.retrieve("test query", top_k_vector=4, max_chunks_per_paper=2)
        
        # Vector candidate counts should show raw_chunks size
        assert res.retrieval_metadata["vector_candidates"] == 5
        
        # After filtering and contiguity packing:
        # p1_chunk_0 and p1_chunk_1 are contiguous (indices 0 and 1) -> merged into one chunk.
        # p1_chunk_2 is not contiguous with the merged block because 2 is adjacent to 1, but wait:
        # the list had [p1_chunk_0, p1_chunk_1, p1_chunk_2].
        # In sorted index order: 0, 1, 2. They are all contiguous!
        # So p1_chunk_0, p1_chunk_1, p1_chunk_2 should merge into ONE chunk.
        # For p2: p2_chunk_0 and p2_chunk_1 are contiguous (0 and 1) -> merged into ONE chunk.
        # Thus, final vector context will have 2 packed chunks.
        assert len(res.vector_context) == 2
        
        # Verify deduplicated papers
        assert len(res.source_papers) == 2
        arxiv_ids = {p["arxiv_id"] for p in res.source_papers}
        assert arxiv_ids == {"p1", "p2"}

def test_hybrid_retriever_graph_bonus_ranking(mock_retrievers):
    """Verify that chunks whose papers connect to query matched entities get score boosts."""
    mock_vector, mock_kg = mock_retrievers
    
    mock_vector.search.return_value = [
        {"chunk_id": "p1_chunk_0", "arxiv_id": "p1", "title": "Paper One", "section": "Abstract", "page_start": 1, "page_end": 1, "text": "text1", "score": 0.60, "chunk_word_count": 5},
        {"chunk_id": "p2_chunk_0", "arxiv_id": "p2", "title": "Paper Two", "section": "Abstract", "page_start": 1, "page_end": 1, "text": "text2", "score": 0.62, "chunk_word_count": 5}
    ]
    
    retriever = HybridRetriever()
    # Configure weights for the test to ensure bonus ranks p1 higher
    retriever.semantic_weight = 0.50
    retriever.graph_weight = 0.50
    retriever.retrieval_config = {
        "policies": {
            "method": {
                "ranking": {
                    "semantic": 0.50,
                    "graph_overlap": 0.50
                },
                "thresholds": {
                    "semantic": 0.0,
                    "overlap": 0.0
                },
                "budgeting": {
                    "min_words": 1200
                }
            }
        }
    }
    
    retriever.entity_cache = {
        "usad 2.0": {"name": "USAD 2.0", "type": "Method", "entity_id": "usad 2.0", "embedding": np.ones(384)}
    }
    
    # Mock model encode to return exact match for query
    mock_vector.model.encode.return_value = np.ones((1, 384))
    
    # Mock Neo4j subgraphs
    mock_kg.get_method_context.return_value = {
        "nodes": [
            {"entity_id": "usad 2.0", "name": "USAD 2.0", "entity_type": "Method"},
            {"entity_id": "p1", "title": "Paper One", "entity_type": "Paper"}
        ],
        "relationships": [
            {"source": "p1", "target": "usad 2.0", "relation": "INTRODUCES"}
        ]
    }
    mock_kg.get_papers_about_method.return_value = []
    mock_kg.get_paper_subgraph.return_value = {"nodes": [], "relationships": []}
    
    with patch.object(retriever, "get_connected_entities_for_papers", return_value=[]):
        res = retriever.retrieve("USAD 2.0", top_k_vector=2, max_chunks_per_paper=1)
        
        # p2_chunk_0: similarity = 0.62, graph_bonus = 0.0 -> combined = 0.50 * 0.62 = 0.31
        # p1_chunk_0: similarity = 0.60, path_score = 1.0, normalized_degree = 1.0 (degree 1 / max 1)
        #             graph_bonus = 0.04 * 1.0 + 0.01 * 1.0 = 0.05
        #             combined = 0.50 * 0.60 + 0.50 * 0.05 = 0.30 + 0.025 = 0.325
        # So p1_chunk_0 should be ranked first!
        assert len(res.vector_context) == 2
        assert res.vector_context[0].chunk_id == "p1_chunk_0"
        assert res.vector_context[0].combined_score == pytest.approx(0.80)
        assert res.vector_context[0].graph_bonus == pytest.approx(0.05)
        
        # Verify explanations
        exps = res.vector_context[0].explanations
        types = {e["type"] if isinstance(e, dict) else e.type for e in exps}
        assert "semantic_match" in types
        assert "graph_neighbor" in types
        assert "introduced_method" in types

def test_hybrid_retriever_context_packing(mock_retrievers):
    """Verify that contiguous chunks are correctly packed (merged)."""
    mock_vector, mock_kg = mock_retrievers
    
    # 3 chunks from p1: index 3, 4 and 6. 3 and 4 should merge; 6 is separate.
    mock_vector.search.return_value = [
        {"chunk_id": "p1_chunk_3", "arxiv_id": "p1", "title": "Paper One", "section": "Method", "page_start": 3, "page_end": 3, "text": "This is paragraph three.", "score": 0.8, "chunk_word_count": 4},
        {"chunk_id": "p1_chunk_4", "arxiv_id": "p1", "title": "Paper One", "section": "Method", "page_start": 4, "page_end": 4, "text": "This is paragraph four.", "score": 0.75, "chunk_word_count": 4},
        {"chunk_id": "p1_chunk_6", "arxiv_id": "p1", "title": "Paper One", "section": "Results", "page_start": 5, "page_end": 5, "text": "This is paragraph six.", "score": 0.7, "chunk_word_count": 4}
    ]
    
    retriever = HybridRetriever()
    retriever.entity_cache = {}
    mock_kg.get_paper_subgraph.return_value = {"nodes": [], "relationships": []}
    
    with patch.object(retriever, "get_connected_entities_for_papers", return_value=[]):
        res = retriever.retrieve("test query", top_k_vector=3, max_chunks_per_paper=3)
        
        # We should end up with 2 packed chunks:
        # 1. Merged chunk_3 and chunk_4
        # 2. Independent chunk_6
        assert len(res.vector_context) == 2
        
        # Detailed packing assertions
        merged_chunk = res.vector_context[0]
        independent_chunk = res.vector_context[1]
        
        assert merged_chunk.chunk_word_count == 8
        assert independent_chunk.chunk_word_count == 4
        assert merged_chunk.page_start == 3
        assert merged_chunk.page_end == 4
        assert "\n\n" in merged_chunk.text
        
        # Verify coverage in metadata
        coverage = res.retrieval_metadata["coverage"]
        assert coverage["papers"] == 1
        assert "methods" in coverage
        assert "datasets" in coverage
        assert "concepts" in coverage
        
        merged = res.vector_context[0]
        assert merged.chunk_id == "p1_chunk_3"
        assert merged.text == "This is paragraph three.\n\nThis is paragraph four."
        assert merged.page_start == 3
        assert merged.page_end == 4
        assert merged.chunk_word_count == 8
        assert merged.similarity_score == 0.8
        
        assert res.vector_context[1].chunk_id == "p1_chunk_6"

def test_hybrid_retriever_empty(mock_retrievers):
    """Verify retrieval returns valid empty schemas when no matches are found."""
    mock_vector, mock_kg = mock_retrievers
    mock_vector.search.return_value = []
    
    retriever = HybridRetriever()
    retriever.entity_cache = {}
    
    res = retriever.retrieve("some query")
    
    assert res.query == "some query"
    assert res.vector_context == []
    assert res.graph_context == {"nodes": [], "relationships": []}
    assert res.source_papers == []
    assert res.citations == []
    assert res.retrieval_metadata["intent"] == "research"
    assert res.retrieval_metadata["vector_candidates"] == 0
    assert res.retrieval_metadata["graph_nodes"] == 0
    assert res.retrieval_metadata["graph_relationships"] == 0

def test_hybrid_retriever_evidence_policy(mock_retrievers):
    """Verify the strategy pattern execution, dynamic configuration scoring, and telemetry nested output."""
    mock_vector, mock_kg = mock_retrievers
    
    # 2 dataset chunks
    mock_vector.search.return_value = [
        {"chunk_id": "c1", "arxiv_id": "p1", "title": "Paper One", "section": "Method", "page_start": 1, "page_end": 1, "text": "This evaluates Code2LoRA benchmark.", "score": 0.8, "chunk_word_count": 10},
        {"chunk_id": "c2", "arxiv_id": "p2", "title": "Paper Two", "section": "Method", "page_start": 2, "page_end": 2, "text": "This mentions RepoPeftBench and evaluates Code2LoRA.", "score": 0.75, "chunk_word_count": 10}
    ]
    
    retriever = HybridRetriever()
    retriever.entity_cache = {
        "code2lora": {"name": "Code2LoRA", "type": "Method", "entity_id": "code2lora"},
        "repopeftbench": {"name": "RepoPeftBench", "type": "Dataset", "entity_id": "repopeftbench"}
    }
    
    # Mock Neo4j subgraphs
    mock_kg.get_method_context.return_value = {
        "nodes": [
            {"entity_id": "code2lora", "name": "Code2LoRA", "entity_type": "Method"},
            {"entity_id": "repopeftbench", "name": "RepoPeftBench", "entity_type": "Dataset"},
            {"entity_id": "p1", "title": "Paper One", "entity_type": "Paper"},
            {"entity_id": "p2", "title": "Paper Two", "entity_type": "Paper"}
        ],
        "relationships": [
            {"source": "p1", "target": "code2lora", "relation": "INTRODUCES"},
            {"source": "p2", "target": "repopeftbench", "relation": "USES"},
            {"source": "code2lora", "target": "repopeftbench", "relation": "EVALUATED_ON"}
        ]
    }
    mock_kg.get_papers_about_method.return_value = []
    mock_kg.get_paper_subgraph.return_value = {"nodes": [], "relationships": []}
    
    with patch.object(retriever, "get_connected_entities_for_papers", return_value=[]):
        res = retriever.retrieve("What datasets evaluate Code2LoRA?", top_k_vector=2, max_chunks_per_paper=1)
        
        # Verify intent was classified as dataset
        assert res.retrieval_metadata["intent"] == "dataset"
        
        # Verify custom policy stats telemetry
        policy_stats = res.retrieval_metadata["policy"]
        assert policy_stats["name"] == "dataset"
        assert policy_stats["tier_distribution"]["tier1"] > 0
        assert "evidence" in policy_stats
        
        # Verify SelectedEvidenceChunk attributes
        assert len(res.vector_context) > 0
        chunk = res.vector_context[0]
        assert chunk.retrieval_reason is not None
        assert chunk.retrieval_reason.policy == "dataset"
        assert chunk.retrieval_reason.tier == 1
        assert chunk.retrieval_reason.ranking.weight_breakdown.semantic > 0.0
        
        # Verify registration capability on PolicyFactory
        from src.evidence.policy_factory import PolicyFactory
        from src.evidence.base_policy import BaseEvidencePolicy
        
        class MockCustomPolicy(BaseEvidencePolicy):
            def classify(self, uc, gc, qe): return uc, [], []
            def score(self, t1, t2, t3, gc, qe):
                for c in t1:
                    c["retrieval_reason"] = {
                        "policy": "custom", "tier": 1, "strategy": "custom_strategy",
                        "ranking": {"semantic_score": 1.0, "graph_overlap": 1.0, "graph_bonus": 0.0, "final_score": 1.0, "weight_breakdown": {}},
                        "source": {}, "merge": {"merged_chunks": 1, "merged_word_count": 0, "merged_chunk_ids": [], "provenance_sources": []}
                    }
                return t1, t2, t3
            def rank(self, t1, t2, t3): return t1, t2, t3
            
        PolicyFactory.register("custom_intent", MockCustomPolicy)
        custom_policy = PolicyFactory.create("custom_intent", retriever.retrieval_config, retriever.entity_cache)
        assert isinstance(custom_policy, MockCustomPolicy)
