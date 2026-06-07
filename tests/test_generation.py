import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.hybrid_retriever import HybridRetriever, Citation, RetrievalResult
from src.answer_generator import GroundedAnswerGenerator, GenerationResult
from src.question_classifier import QuestionClassifier
from src.context_builder import ContextBuilder
from src.context_compressor import ContextCompressor
from src.citation_validator import CitationValidator
from src.confidence_estimator import ConfidenceEstimator

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=HybridRetriever)
    retriever.config = {
        "generation": {
            "provider": "gemini",
            "primary_model": "gemini-2.5-flash",
            "min_confidence_threshold": 0.45,
            "debug_mode": True,
            "semantic_drift_threshold": 0.55,
            "weights": {
                "semantic": 0.45,
                "graph": 0.20,
                "citation": 0.20,
                "rerank": 0.15
            }
        },
        "retrieval": {
            "citation_alignment": {
                "warning_threshold": 0.65,
                "reject_threshold": 0.50
            },
            "replace_invalid_citations": True
        }
    }
    
    # Mock vector retriever and embedding model
    retriever.vector_retriever = MagicMock()
    mock_model = MagicMock()
    # Mock encode to return a unit vector array
    mock_model.encode.return_value = [np.array([1.0, 0.0])]
    retriever.vector_retriever.model = mock_model
    retriever.vector_retriever.model_name = "BAAI/bge-small-en-v1.5"
    return retriever

def test_question_classifier():
    classifier = QuestionClassifier()
    assert classifier.classify("Compare method X and method Y") == "comparison"
    assert classifier.classify("Summarize this paper") == "summary"
    assert classifier.classify("What is the evaluation protocol?") == "evaluation"
    assert classifier.classify("What are the advantages of TAM?") == "advantages"
    assert classifier.classify("Random query") == "default"

def test_context_builder_facts_sorting():
    builder = ContextBuilder()
    
    # Verify graph facts sorting by relationship importance priority
    # INTRODUCES (0), SOLVES (1), EVALUATED_ON (2), USES (3), MENTIONS (4), BELONGS_TO (5)
    graph_context = {
        "nodes": [
            {"entity_id": "p1", "title": "Paper One", "type": "Paper"},
            {"entity_id": "m1", "name": "Method One", "type": "Method"},
            {"entity_id": "d1", "name": "Dataset One", "type": "Dataset"},
            {"entity_id": "t1", "name": "Task One", "type": "Task"},
            {"entity_id": "c1", "name": "Category One", "type": "Category"}
        ],
        "relationships": [
            {"source": "p1", "target": "c1", "relation": "BELONGS_TO"},
            {"source": "p1", "target": "m1", "relation": "INTRODUCES"},
            {"source": "m1", "target": "d1", "relation": "EVALUATED_ON"},
            {"source": "m1", "target": "t1", "relation": "SOLVES"}
        ]
    }
    
    facts = builder.build_graph_facts(graph_context)
    assert len(facts) == 4
    # INTRODUCES must come first
    assert "introduces the method" in facts[0]
    # SOLVES must come second
    assert "solves task" in facts[1]
    # EVALUATED_ON must come third
    assert "evaluated on dataset" in facts[2]
    # BELONGS_TO must come last
    assert "belongs to category" in facts[3]

def test_context_compressor_length_aware():
    compressor = ContextCompressor()
    
    # Case 1: Paragraphs share >= 95% similarity and length difference < 20% -> keep shorter
    passages_shorter = [
        {"formatted_text": "Source Citation: [Citation 1]\nText:\nThis is a unique paragraph. It is long enough to exceed the fifty character limit for deduplication."},
        {"formatted_text": "Source Citation: [Citation 2]\nText:\nThis is unique paragraph. It is long enough to exceed fifty character limit for deduplication."} # Slightly shorter, < 20% diff
    ]
    compressed_shorter = compressor.compress(passages_shorter)
    # The first one should be marked as duplicate (omitted) and the second kept because it's shorter
    assert "Duplicate content omitted" in compressed_shorter[0]["formatted_text"]
    assert "This is unique paragraph" in compressed_shorter[1]["formatted_text"]
    
    # Case 2: Paragraphs share >= 95% similarity and length difference >= 20% -> keep longer
    passages_longer = [
        {"formatted_text": "Source Citation: [Citation 1]\nText:\nThis is unique paragraph. It is long enough to exceed fifty character limit for deduplication."}, # Shorter
        {"formatted_text": "Source Citation: [Citation 2]\nText:\nThis is a unique paragraph. It is long enough to exceed the fifty character limit for deduplication, and has some extra words here to make it longer."} # > 20% longer
    ]
    compressed_longer = compressor.compress(passages_longer)
    # The first one should be discarded (omitted) and the second kept because it's longer
    assert "Duplicate content omitted" in compressed_longer[0]["formatted_text"]
    assert "has some extra words" in compressed_longer[1]["formatted_text"]

def test_citation_validator_semantic_alignment():
    # Setup mock retriever with mock embedding model
    mock_model = MagicMock()
    # Make sentence and paragraph embeddings match perfectly or mismatch
    def mock_encode(texts, **kwargs):
        print(f"MOCK ENCODE CALLED with texts: {texts}")
        if "sentence" in texts[0]:
            print("MATCHED sentence")
            return [np.array([1.0, 0.0])] # Sentence vector
        elif "matched paragraph" in texts[0]:
            print("MATCHED matched paragraph")
            return [np.array([1.0, 0.0])] # Paragraph vector (similarity = 1.0)
        elif "warning paragraph" in texts[0]:
            print("MATCHED warning paragraph")
            return [np.array([0.6, 0.8])] # Paragraph vector (similarity = 0.60, triggers warning)
        else:
            print("MATCHED else block")
            return [np.array([0.0, 1.0])] # Paragraph vector (similarity = 0.0, rejected)

            
    mock_model.encode.side_effect = mock_encode
    
    validator = CitationValidator(
        retriever=mock_model,
        warning_threshold=0.65,
        reject_threshold=0.50,
        replace_invalid=True
    )
    
    available = [
        Citation(paper_title="P1", arxiv_id="1", section="sec", page_start=1, page_end=1, chunk_id="c1", similarity_score=0.8, graph_bonus=0.0, combined_score=0.8),
        Citation(paper_title="P2", arxiv_id="2", section="sec", page_start=2, page_end=2, chunk_id="c2", similarity_score=0.75, graph_bonus=0.0, combined_score=0.75),
        Citation(paper_title="P3", arxiv_id="3", section="sec", page_start=3, page_end=3, chunk_id="c3", similarity_score=0.7, graph_bonus=0.0, combined_score=0.7)
    ]
    
    vector_context = [
        {"chunk_id": "c1", "text": "matched paragraph text"},
        {"chunk_id": "c2", "text": "warning paragraph text"},
        {"chunk_id": "c3", "text": "rejected paragraph text"}
    ]
    
    answer = "This sentence is aligned [Citation 1]. This sentence is warning [Citation 2]. This sentence is mismatch [Citation 3]."
    cleaned, used, invalid = validator.validate(answer, available, vector_context)
    
    # Citation 1 should be kept
    assert "[Citation 1]" in cleaned
    # Citation 2 should be kept (warning only)
    assert "[Citation 2]" in cleaned
    assert len(validator.warnings) == 1
    # Citation 3 should be replaced by invalid placeholder
    assert "[Citation 3]" not in cleaned
    assert "[Invalid Citation Removed]" in cleaned
    assert len(invalid) == 1
    
    # Used citations should include P1 and P2
    assert len(used) == 2
    assert used[0].arxiv_id == "1"
    assert used[1].arxiv_id == "2"
    
    # Check precision
    assert validator.citation_precision == pytest.approx(2/3)

def test_confidence_estimator_breakdown():
    # Test confidence breakdown with weights and connectivity math
    weights = {"semantic": 0.45, "graph": 0.20, "citation_coverage": 0.10, "citation_precision": 0.10, "rerank": 0.15}
    estimator = ConfidenceEstimator(min_confidence_threshold=0.45, weights=weights)
    
    retrieval_result = RetrievalResult(
        query="test",
        graph_context={
            "nodes": [{"entity_id": "e1"}],
            "relationships": []
        },
        vector_context=[
            {"similarity_score": 0.8, "reranker_score": 0.9, "chunk_word_count": 10}
        ],
        source_papers=[],
        citations=[],
        retrieval_metadata={
            "query_entities": ["e1", "e2"] # e1 is present (reachable), e2 is missing -> connectivity = 0.50
        }
    )
    
    used_citations = [
        Citation(paper_title="P1", arxiv_id="1", section="sec", page_start=1, page_end=1, chunk_id="c1", similarity_score=0.8, graph_bonus=0.0, combined_score=0.8)
    ]
    
    # Pass citation_precision = 0.50
    confidence, coverage, details = estimator.estimate(retrieval_result, used_citations, citation_precision=0.50)
    
    # semantic = 0.8, graph = 0.5, citation_coverage = 1.0, citation_precision = 0.5, reranker = 0.9
    # math: 0.45 * 0.8 + 0.20 * 0.5 + 0.10 * 1.0 + 0.10 * 0.5 + 0.15 * 0.9 = 0.36 + 0.10 + 0.10 + 0.05 + 0.135 = 0.745
    assert confidence == pytest.approx(0.745)
    assert details["confidence_breakdown"]["semantic"] == 0.8
    assert details["confidence_breakdown"]["graph"] == 0.5
    assert details["confidence_breakdown"]["citation_coverage"] == 1.0
    assert details["confidence_breakdown"]["citation_precision"] == 0.5
    assert details["confidence_breakdown"]["reranker"] == 0.9
    assert details["confidence_breakdown"]["final"] == confidence

@patch("src.answer_generator.requests.post")
def test_answer_generator_gemini_success(mock_post, mock_retriever):
    # Mock Gemini response json
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "This is a gemini answer [Citation 1]."}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    # Mock hybrid retriever return
    retrieval_result = RetrievalResult(
        query="Explain USAD",
        graph_context={"nodes": [], "relationships": []},
        vector_context=[
            {"chunk_id": "c1", "arxiv_id": "1", "title": "Paper 1", "section": "sec", "page_start": 1, "page_end": 1, "chunk_word_count": 10, "text": "text", "similarity_score": 0.8, "graph_bonus": 0.0, "combined_score": 0.8, "explanations": []}
        ],
        source_papers=[{"arxiv_id": "1", "title": "Paper 1"}],
        citations=[
            Citation(paper_title="Paper 1", arxiv_id="1", section="sec", page_start=1, page_end=1, chunk_id="c1", similarity_score=0.8, graph_bonus=0.0, combined_score=0.8)
        ],
        retrieval_metadata={"intent": "method", "entity_matches": 0}
    )
    mock_retriever.retrieve.return_value = retrieval_result

    # Mock BGE encode inside drift detection
    mock_retriever.vector_retriever.model.encode.return_value = [np.array([1.0, 0.0])]

    generator = GroundedAnswerGenerator(mock_retriever)
    
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake_key"}):
        res = generator.generate_answer("Explain USAD")
        
        assert res.query == "Explain USAD"
        assert "gemini answer" in res.answer
        assert len(res.citations) == 1
        assert res.confidence > 0.0
        assert res.metadata["provider_used"] == "gemini"
        assert not res.metadata["fallback_used"]
        assert res.provenance["papers_used"] == 1
        # Telemetry updates assert
        assert "confidence_breakdown" in res.metadata
        assert "citation_precision" in res.metadata
        assert res.metadata["drift_score"] == pytest.approx(1.0)
