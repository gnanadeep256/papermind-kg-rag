import pytest
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np

from src.evaluation.metrics import QueryEvaluationResult, AggregatedEvaluationResult, LatencyPercentiles
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.policy_evaluator import PolicyEvaluator
from src.evaluation.citation_evaluator import CitationEvaluator
from src.evaluation.grounding_evaluator import GroundingEvaluator
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.robustness_evaluator import RobustnessEvaluator
from src.evaluation.calibration_evaluator import CalibrationEvaluator
from src.evaluation.abstention_evaluator import AbstentionEvaluator
from src.evaluation.latency_evaluator import LatencyEvaluator
from src.evaluation.regression import RegressionEngine

# Mock Chunk for helper structures
class MockChunk:
    def __init__(self, text, arxiv_id, chunk_id, title="", similarity_score=0.8, graph_bonus=0.0, combined_score=0.8, retrieval_reason=None):
        self.text = text
        self.arxiv_id = arxiv_id
        self.chunk_id = chunk_id
        self.title = title
        self.similarity_score = similarity_score
        self.graph_bonus = graph_bonus
        self.combined_score = combined_score
        self.retrieval_reason = retrieval_reason

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    # Mock return values for model.encode: unit vector mock
    model.encode.return_value = [np.array([1.0, 0.0])]
    return model

def test_retrieval_evaluator():
    evaluator = RetrievalEvaluator()
    query_case = {
        "expected_papers": ["p1", "p2"],
        "expected_entities": ["RepoPeftBench"]
    }
    
    retrieved = [
        MockChunk("Evaluating RepoPeftBench model", "p1", "c1", "Paper 1"),
        MockChunk("Unrelated text block", "p3", "c2", "Paper 3")
    ]
    
    generator_result = {
        "vector_context": retrieved,
        "graph_context": {
            "nodes": [{"entity_id": "RepoPeftBench"}, {"entity_id": "other"}],
            "relationships": []
        }
    }
    
    metrics = evaluator.evaluate(query_case, generator_result)
    
    # Retrieved chunks: 2, relevant: chunk 1 (matches RepoPeftBench & p1). chunk 2 is irrelevant.
    assert metrics["context_precision"] == pytest.approx(0.5)
    # Expected: p1, p2, RepoPeftBench. Recalled: p1, RepoPeftBench -> 2/3 recalled
    assert metrics["context_recall"] == pytest.approx(2/3)
    assert metrics["paper_recall"] == 1.0  # p1 is in retrieved papers
    assert metrics["entity_recall"] == 1.0  # RepoPeftBench in text
    assert metrics["paper_diversity"] == 1.0  # p1 and p3 are 2 papers for 2 chunks
    assert metrics["entity_diversity"] == 1.0

def test_policy_evaluator():
    evaluator = PolicyEvaluator()
    query_case = {
        "category": "dataset"
    }
    
    generator_result = {
        "vector_context": [
            MockChunk("text", "p1", "c1", retrieval_reason=MagicMock(ranking=MagicMock(graph_overlap=0.8)))
        ],
        "retrieval_metadata": {
            "intent": "dataset",
            "policy": {
                "name": "dataset",
                "fallback_used": True,
                "tier_distribution": {"tier1": 1, "tier2": 1, "tier3": 0},
                "evidence": {"utilization": 0.45}
            }
        }
    }
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["policy_selected"] == "dataset"
    assert metrics["policy_correct"] == 1.0
    assert metrics["policy_fallback"] == 1.0
    assert metrics["tier1_ratio"] == 0.5
    assert metrics["tier2_ratio"] == 0.5
    assert metrics["budget_utilization"] == 0.45
    assert metrics["graph_overlap_ratio"] == 0.8

def test_citation_evaluator(mock_embedding_model):
    evaluator = CitationEvaluator(embedding_model=mock_embedding_model)
    query_case = {}
    
    generator_result = {
        "answer": "This is a statement [Citation 1].",
        "invalid_citations": [],
        "vector_context": [
            MockChunk("This is chunk one text.", "p1", "c1")
        ],
        "used_citations": [
            MagicMock(chunk_id="c1", arxiv_id="p1")
        ],
        "retrieval_metadata": {
            "citation_precision": 1.0
        }
    }
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_coverage"] == 1.0
    assert metrics["citation_hallucination_rate"] == 0.0
    # mock dot product is 1.0 (from mock embedding encode return)
    assert metrics["semantic_alignment_score"] == pytest.approx(1.0)

def test_grounding_evaluator(mock_embedding_model):
    evaluator = GroundingEvaluator(embedding_model=mock_embedding_model)
    query_case = {}
    
    generator_result = {
        "answer": "This evaluates RepoPeftBench [Citation 1].",
        "vector_context": [
            MockChunk("RepoPeftBench is a benchmark for evaluations.", "p1", "c1")
        ]
    }
    
    metrics = evaluator.evaluate(query_case, generator_result)
    # Check overlap ROUGE F1 score is positive
    assert metrics["evidence_overlap"] > 0.0
    assert metrics["citation_support"] == 1.0
    assert metrics["semantic_grounding"] == pytest.approx(1.0)

def test_generation_evaluator(mock_embedding_model):
    evaluator = GenerationEvaluator(embedding_model=mock_embedding_model)
    query_case = {
        "must_contain": ["RepoPeftBench"],
        "expected_entities": ["Code2LoRA"]
    }
    
    generator_result = {
        "answer": "Code2LoRA uses RepoPeftBench [Citation 1].",
        "vector_context": [
            MockChunk("Code2LoRA benchmark evaluations.", "p1", "c1")
        ]
    }
    
    # Mock LLM faithfulness response
    evaluator._query_llm = MagicMock(return_value='{"statements": [{"statement": "All claims are supported.", "supported": true}]}')
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["completeness"] == 1.0
    assert metrics["groundedness"] == pytest.approx(1.0)
    assert metrics["llm_faithfulness"] == 1.0
    assert metrics["hybrid_faithfulness"] > 0.8

def test_robustness_evaluator(mock_embedding_model):
    evaluator = RobustnessEvaluator(embedding_model=mock_embedding_model)
    
    base_result = {
        "answer": "Answer text [Citation 1].",
        "citations": [MagicMock(arxiv_id="p1")]
    }
    
    var_results = [
        {
            "answer": "Answer text [Citation 1].",
            "citations": [MagicMock(arxiv_id="p1")]
        }
    ]
    
    metrics = evaluator.evaluate_variations(base_result, var_results)
    assert metrics["robustness_semantic_consistency"] == pytest.approx(1.0)
    assert metrics["robustness_entity_consistency"] == pytest.approx(1.0)
    assert metrics["robustness_score"] == pytest.approx(1.0)

def test_calibration_evaluator():
    evaluator = CalibrationEvaluator(num_bins=5)
    
    # Perfect calibration mock cases
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    correctness = [0.0, 0.0, 1.0, 1.0, 1.0] # acc: 0/1, 0/1, 1/1, 1/1, 1/1
    
    metrics = evaluator.evaluate_run(confidences, correctness)
    # Brier score: mean((conf - corr)^2) -> mean([0.01, 0.09, 0.25, 0.09, 0.01]) = 0.45 / 5 = 0.09
    assert metrics["brier_score"] == pytest.approx(0.09)
    # ECE should be calculated correctly
    assert "expected_calibration_error" in metrics

def test_abstention_evaluator():
    evaluator = AbstentionEvaluator()
    
    should_abstain = [True, True, False, False]
    did_abstain = [True, False, False, True]
    
    metrics = evaluator.evaluate_run(should_abstain, did_abstain)
    assert metrics["abstention_accuracy"] == 0.5  # 2 correct out of 4
    # sa=True, da=True -> TP = 1
    # sa=False, da=True -> FP = 1
    # sa=True, da=False -> FN = 1
    # sa=False, da=False -> TN = 1
    # Prec: 1/2 = 0.5, Rec: 1/2 = 0.5, F1: 0.5
    assert metrics["abstention_precision"] == 0.5
    assert metrics["abstention_recall"] == 0.5
    assert metrics["abstention_f1"] == 0.5

def test_latency_evaluator():
    evaluator = LatencyEvaluator()
    
    total_latencies = [1000.0, 2000.0, 3000.0, 4000.0]
    metrics = evaluator.evaluate_run(total_latencies)
    assert metrics["mean"] == 2500.0
    assert metrics["median"] == 2500.0
    assert metrics["p95"] == pytest.approx(3850.0)  # linear interpolation of 95th percentile
    assert metrics["p99"] == pytest.approx(3970.0)


def test_regression_comparison(tmp_path):
    engine = RegressionEngine(reports_dir=str(tmp_path))
    
    # Save a mock run_20260601_120000 benchmark.json
    run_dir = tmp_path / "run_20260601_120000"
    run_dir.mkdir()
    
    prev_aggregated = AggregatedEvaluationResult(
        run_timestamp="20260601_120000",
        total_queries=10,
        overall_score=85.0,
        avg_retrieval_recall=0.80,
        avg_citation_precision=0.90,
        avg_hybrid_faithfulness=0.88,
        avg_groundedness=0.78,
        avg_abstention_accuracy=0.90,
        avg_robustness=0.85,
        avg_latency_ms=1200.0,
        latency_percentiles=LatencyPercentiles(mean=1200.0, median=1100.0, p95=1800.0, p99=2200.0),
        expected_calibration_error=0.08,
        brier_score=0.10,
        avg_paper_diversity=0.80,
        avg_entity_diversity=0.75,
        policy_routing_accuracy=0.90,
        policy_fallback_rate=0.20,
        avg_budget_utilization=0.50,
        category_breakdown={}
    )
    
    with open(run_dir / "benchmark.json", "w") as f:
        f.write(prev_aggregated.model_dump_json())
        
    # Query latest run relative to run_20260602_120000
    prev_run = engine.get_latest_previous_run("20260602_120000")
    assert prev_run is not None
    assert prev_run.run_timestamp == "20260601_120000"
    
    # Create current run result
    current_aggregated = AggregatedEvaluationResult(
        run_timestamp="20260602_120000",
        total_queries=10,
        overall_score=88.5,
        avg_retrieval_recall=0.82,
        avg_citation_precision=0.95,
        avg_hybrid_faithfulness=0.92,
        avg_groundedness=0.80,
        avg_abstention_accuracy=0.90,
        avg_robustness=0.87,
        avg_latency_ms=1100.0,
        latency_percentiles=LatencyPercentiles(mean=1100.0, median=1000.0, p95=1600.0, p99=2000.0),
        expected_calibration_error=0.07,
        brier_score=0.09,
        avg_paper_diversity=0.82,
        avg_entity_diversity=0.78,
        policy_routing_accuracy=0.92,
        policy_fallback_rate=0.18,
        avg_budget_utilization=0.52,
        category_breakdown={}
    )
    
    diffs = engine.compare_runs(current_aggregated, prev_run)
    assert len(diffs) > 0
    # Overall Score comparison should show improvement (+3.5)
    overall_diff = next(d for d in diffs if d.metric_name == "Overall Score")
    assert overall_diff.differential == pytest.approx(3.5)
    assert "+" in overall_diff.label
    assert "↑" in overall_diff.label

def test_confidence_interval_serialization():
    from src.evaluation.metrics import ConfidenceInterval
    ci = ConfidenceInterval(lower=80.0, upper=90.0, width=10.0)
    dumped = ci.model_dump_json()
    loaded = ConfidenceInterval.model_validate_json(dumped)
    assert loaded.lower == 80.0
    assert loaded.upper == 90.0
    assert loaded.width == 10.0

def test_retrieval_evaluator_diversity_edge_cases():
    evaluator = RetrievalEvaluator()
    # Case A: no graph nodes exist
    query_case = {"expected_papers": [], "expected_entities": []}
    generator_result = {
        "vector_context": [MockChunk("text", "p1", "c1")],
        "graph_context": {"nodes": [], "relationships": []}
    }
    res = evaluator.evaluate(query_case, generator_result)
    assert res["method_diversity"] == 1.0
    assert res["dataset_diversity"] == 1.0
    assert res["author_diversity"] == 1.0
    assert res["method_entropy"] == 0.0
    assert res["dataset_entropy"] == 0.0
    
    # Case B: duplicated nodes
    generator_result_dups = {
        "vector_context": [MockChunk("text", "p1", "c1")],
        "graph_context": {
            "nodes": [
                {"entity_id": "m1", "entity_type": "Method"},
                {"entity_id": "m1", "entity_type": "Method"},
                {"entity_id": "m2", "entity_type": "Method"}
            ],
            "relationships": []
        }
    }
    res_dups = evaluator.evaluate(query_case, generator_result_dups)
    # 2 unique out of 3 total methods -> 2/3
    assert res_dups["method_diversity"] == pytest.approx(2.0 / 3.0)
    assert res_dups["method_entropy"] > 0.0

def test_sqlite_embedding_cache(tmp_path):
    from src.evaluation.embedding_cache import CachedEmbeddingModel
    db_path = str(tmp_path / "cache.sqlite")
    mock_base_model = MagicMock()
    mock_base_model.encode.return_value = np.array([[0.1, 0.2]])
    
    # 1. Miss the cache first
    cached_model = CachedEmbeddingModel(mock_base_model, model_name="model_A", db_path=db_path)
    emb1 = cached_model.encode(["hello"], normalize_embeddings=True)
    assert mock_base_model.encode.call_count == 1
    assert emb1.tolist() == [[0.1, 0.2]]
    
    # 2. Hit the cache
    emb2 = cached_model.encode(["hello"], normalize_embeddings=True)
    assert mock_base_model.encode.call_count == 1  # call count doesn't increment
    assert emb2.tolist() == [[0.1, 0.2]]
    
    # 3. Cache invalidation on model name change
    cached_model_diff_name = CachedEmbeddingModel(mock_base_model, model_name="model_B", db_path=db_path)
    emb3 = cached_model_diff_name.encode(["hello"], normalize_embeddings=True)
    assert mock_base_model.encode.call_count == 2  # call count increments because of different model name
    assert emb3.tolist() == [[0.1, 0.2]]
    
    # 4. Cache hit for model_B
    cached_model_diff_name.encode(["hello"], normalize_embeddings=True)
    assert mock_base_model.encode.call_count == 2
    cached_model.close()
    cached_model_diff_name.close()

def test_deterministic_bootstrap():
    # Mock a list of query results where everyone is identical
    # This should yield zero-width confidence intervals
    from src.evaluation.metrics import QueryEvaluationResult
    
    # Create mock identical QueryEvaluationResults
    mock_res = QueryEvaluationResult(
        query_id="q1", query="query", category="method", answer="answer",
        is_abstention=False, allow_abstain=False, latency_ms=100.0, latency_breakdown={},
        abstention_mode="correct_answer", abstention_accuracy=1.0,
        context_precision=1.0, context_recall=1.0, paper_recall=1.0, entity_recall=1.0,
        paper_diversity=1.0, entity_diversity=1.0, policy_selected="method",
        policy_correct=True, policy_fallback=False, tier1_ratio=1.0, tier2_ratio=0.0, tier3_ratio=0.0,
        budget_utilization=0.5, graph_overlap_ratio=1.0,
        citation_precision=1.0, citation_coverage=1.0, citation_hallucination_rate=0.0, semantic_alignment_score=1.0,
        evidence_overlap=1.0, citation_support=1.0, semantic_grounding=1.0,
        completeness=1.0, groundedness=1.0, llm_faithfulness=1.0, hybrid_faithfulness=1.0,
        robustness_semantic_consistency=1.0, robustness_entity_consistency=1.0, robustness_score=1.0
    )
    
    # Compute bootstrap CIs
    target_ms = 8000
    score_w = {"retrieval": 0.20, "citation": 0.20, "faithfulness": 0.25, "groundedness": 0.15, "abstention": 0.10, "latency": 0.10}
    
    # Setup resample simulation
    num_resamples = 50
    confidence_level = 0.95
    random_seed = 42
    n_queries = 5
    query_eval_results = [mock_res] * n_queries
    
    indices = np.arange(n_queries)
    rng = np.random.default_rng(random_seed)
    
    resample_overall_scores = []
    for _ in range(num_resamples):
        res_idx = rng.choice(indices, size=n_queries, replace=True)
        r_ret = float(np.mean([query_eval_results[i].context_recall for i in res_idx]))
        r_cit = float(np.mean([query_eval_results[i].citation_precision for i in res_idx]))
        r_faith = float(np.mean([query_eval_results[i].hybrid_faithfulness for i in res_idx]))
        r_ground = float(np.mean([query_eval_results[i].groundedness for i in res_idx]))
        r_abst = float(np.mean([query_eval_results[i].abstention_accuracy for i in res_idx]))
        
        r_latencies = [query_eval_results[i].latency_ms for i in res_idx]
        r_p95 = float(np.percentile(r_latencies, 95))
        r_latency_penalty = max(0.0, min(1.0, (r_p95 - target_ms) / target_ms)) if r_p95 > target_ms else 0.0
        
        r_overall = (
            score_w["retrieval"] * r_ret +
            score_w["citation"] * r_cit +
            score_w["faithfulness"] * r_faith +
            score_w["groundedness"] * r_ground +
            score_w["abstention"] * r_abst +
            score_w["latency"] * (1.0 - r_latency_penalty)
        ) * 100.0
        resample_overall_scores.append(r_overall)
        
    # Calculate CI
    alpha = 1.0 - confidence_level
    q_low = (alpha / 2.0) * 100.0
    q_high = (1.0 - alpha / 2.0) * 100.0
    lower, upper = np.percentile(resample_overall_scores, [q_low, q_high])
    
    assert lower == 100.0
    assert upper == 100.0
    assert (upper - lower) == 0.0  # zero-width interval
    
    # Check deterministic behavior with seed
    rng1 = np.random.default_rng(random_seed)
    rng2 = np.random.default_rng(random_seed)
    choices1 = rng1.choice(indices, size=10, replace=True)
    choices2 = rng2.choice(indices, size=10, replace=True)
    assert np.array_equal(choices1, choices2)

def test_sqlite_cache_pruning_on_model_change(tmp_path):
    import sqlite3
    from src.evaluation.embedding_cache import CachedEmbeddingModel
    db_path = str(tmp_path / "cache.sqlite")
    mock_base_model = MagicMock()
    mock_base_model.encode.return_value = np.array([[0.1, 0.2]])
    
    # 1. Populate model_A entry
    cached_model_A = CachedEmbeddingModel(mock_base_model, model_name="model_A", db_path=db_path, invalidation_strategy="aggressive")
    cached_model_A.encode(["hello"], normalize_embeddings=True)
    cached_model_A.close()
    
    # Check model_A entry exists
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_A'")
    assert cursor.fetchone()[0] == 1
    conn.close()
    
    # 2. Init model_B (should prune model_A entries)
    cached_model_B = CachedEmbeddingModel(mock_base_model, model_name="model_B", db_path=db_path, invalidation_strategy="aggressive")
    cached_model_B.close()
    
    # Check model_A entry was pruned
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_A'")
    assert cursor.fetchone()[0] == 0
    conn.close()

def test_dry_run_benchmark_metrics(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from src.evaluation.benchmark_runner import BenchmarkRunner
    
    # Create a tiny gold dataset with 2 cases to keep test extremely fast
    tiny_cases = [
        {
            "id": "gold_001",
            "query": "What datasets evaluate Code2LoRA?",
            "category": "dataset",
            "expected_papers": ["2606.06492"],
            "expected_entities": ["RepoPeftBench"],
            "must_contain": ["RepoPeftBench", "604 Python repositories"],
            "must_not_contain": ["Readability"],
            "allow_abstain": False,
            "variations": ["What benchmark does Code2LoRA use?"]
        },
        {
            "id": "gold_003",
            "query": "What is the recipe for baking a chocolate cake?",
            "category": "ood",
            "expected_papers": [],
            "expected_entities": [],
            "must_contain": [],
            "must_not_contain": ["chocolate", "cake", "baking", "flour"],
            "allow_abstain": True,
            "variations": ["How to bake a chocolate cake?"]
        }
    ]
    tiny_gold_path = str(tmp_path / "tiny_gold_dataset.json")
    with open(tiny_gold_path, "w", encoding="utf-8") as f:
        json.dump(tiny_cases, f)

    # Patch SentenceTransformer and CachedEmbeddingModel to prevent downloading/loading heavy models
    with patch("sentence_transformers.SentenceTransformer"), \
         patch("src.evaluation.benchmark_runner.CachedEmbeddingModel") as mock_cached_model_class:
         
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_cached_model_class.return_value = mock_embedding_model
        
        runner = BenchmarkRunner(dry_run=True)
        # Minimize bootstrap resampling loop during tests
        runner.eval_config["bootstrap"]["num_resamples"] = 5
        
        report_file = runner.run_benchmark(gold_dataset_path=tiny_gold_path)
        report_dir = os.path.dirname(report_file)
    
    # Load aggregated json results
    json_path = os.path.join(report_dir, "benchmark_dryrun.json")
    assert os.path.exists(json_path)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Verify regression metrics bounds are satisfied (> 90%)
    assert data["avg_retrieval_recall"] > 0.90
    assert data["policy_routing_accuracy"] > 0.90

def test_retrieval_pairwise_distances_identical():
    # Setup mock embedding model that returns identical vectors
    mock_model = MagicMock()
    # 3 chunks, returns 3 identical vectors
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    
    evaluator = RetrievalEvaluator(embedding_model=mock_model, compute_pairwise_metrics=True)
    
    query_case = {"expected_papers": ["p1"], "expected_entities": []}
    retrieved = [
        MockChunk("Chunk text A", "p1", "c1"),
        MockChunk("Chunk text A", "p1", "c2"),
        MockChunk("Chunk text A", "p1", "c3")
    ]
    generator_result = {"vector_context": retrieved}
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["avg_pairwise_distance"] == pytest.approx(0.0)
    assert metrics["min_pairwise_distance"] == pytest.approx(0.0)
    assert metrics["pairwise_distance_std"] == pytest.approx(0.0)
    assert metrics["context_redundancy_score"] == pytest.approx(1.0)

def test_retrieval_pairwise_distances_orthogonal():
    # Setup mock embedding model that returns orthogonal vectors
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    evaluator = RetrievalEvaluator(embedding_model=mock_model, compute_pairwise_metrics=True)
    
    query_case = {"expected_papers": ["p1"], "expected_entities": []}
    retrieved = [
        MockChunk("Chunk text X", "p1", "c1"),
        MockChunk("Chunk text Y", "p1", "c2")
    ]
    generator_result = {"vector_context": retrieved}
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["avg_pairwise_distance"] == pytest.approx(1.0)
    assert metrics["min_pairwise_distance"] == pytest.approx(1.0)
    assert metrics["pairwise_distance_std"] == pytest.approx(0.0)
    assert metrics["context_redundancy_score"] == pytest.approx(0.0)

def test_retrieval_pairwise_distances_single():
    mock_model = MagicMock()
    evaluator = RetrievalEvaluator(embedding_model=mock_model, compute_pairwise_metrics=True)
    
    query_case = {"expected_papers": ["p1"], "expected_entities": []}
    retrieved = [
        MockChunk("Chunk text X", "p1", "c1")
    ]
    generator_result = {"vector_context": retrieved}
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["avg_pairwise_distance"] is None
    assert metrics["min_pairwise_distance"] is None
    assert metrics["pairwise_distance_std"] is None
    assert metrics["context_redundancy_score"] is None
    # Verify encode was never called
    mock_model.encode.assert_not_called()

def test_retrieval_pairwise_distances_empty():
    mock_model = MagicMock()
    evaluator = RetrievalEvaluator(embedding_model=mock_model, compute_pairwise_metrics=True)
    
    query_case = {"expected_papers": ["p1"], "expected_entities": []}
    retrieved = []
    generator_result = {"vector_context": retrieved}
    
    metrics = evaluator.evaluate(query_case, generator_result)
    assert metrics["avg_pairwise_distance"] is None
    assert metrics["min_pairwise_distance"] is None
    assert metrics["pairwise_distance_std"] is None
    assert metrics["context_redundancy_score"] is None
    # Verify encode was never called
    mock_model.encode.assert_not_called()

def test_cache_invalidation_strategies(tmp_path):
    from src.evaluation.embedding_cache import SqliteEmbeddingCache
    import sqlite3
    import time
    
    db_path = str(tmp_path / "cache.sqlite")
    
    # 1. Test Aggressive invalidation
    cache_aggressive = SqliteEmbeddingCache(
        db_path=db_path,
        invalidation_strategy="aggressive",
        max_models=3,
        max_age_days=30
    )
    # Populate cache for model_A and model_B
    cache_aggressive.set("text1", "model_A", True, [0.1, 0.2])
    cache_aggressive.set("text2", "model_B", True, [0.3, 0.4])
    # Run invalidation for model_B
    cache_aggressive.invalidate_other_models("model_B")
    
    # Check that model_A was deleted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_A'")
    assert cursor.fetchone()[0] == 0
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_B'")
    assert cursor.fetchone()[0] == 1
    conn.close()
    cache_aggressive.close()
    
    # Clear database for next strategy test
    os.remove(db_path)
    
    # 2. Test Keep All invalidation
    cache_keep_all = SqliteEmbeddingCache(
        db_path=db_path,
        invalidation_strategy="keep_all",
        max_models=3,
        max_age_days=30
    )
    cache_keep_all.set("text1", "model_A", True, [0.1, 0.2])
    cache_keep_all.set("text2", "model_B", True, [0.3, 0.4])
    cache_keep_all.invalidate_other_models("model_B")
    
    # Check that model_A is preserved
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_A'")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = 'model_B'")
    assert cursor.fetchone()[0] == 1
    conn.close()
    cache_keep_all.close()
    
    # Clear database
    os.remove(db_path)
    
    # 3. Test LRU invalidation (max_models = 2)
    cache_lru = SqliteEmbeddingCache(
        db_path=db_path,
        invalidation_strategy="lru",
        max_models=2,
        max_age_days=30
    )
    # Populate usage entries to simulate LRU order: model_A (oldest), model_B (older), model_C (active)
    conn = sqlite3.connect(db_path)
    cache_lru._init_db()
    
    now = int(time.time() * 1000)
    with conn:
        conn.execute("INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES ('model_A', ?)", (now - 100 * 1000,))
        conn.execute("INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES ('model_B', ?)", (now - 50 * 1000,))
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h1', 't1', 'model_A', 1, '[]')")
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h2', 't2', 'model_B', 1, '[]')")
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h3', 't3', 'model_C', 1, '[]')")
    conn.close()
    
    # Access/touch model_C
    cache_lru.invalidate_other_models("model_C")
    
    # Since max_models = 2, it should keep model_C (active) and model_B (last_used = now - 50 * 1000) and prune model_A (last_used = now - 100 * 1000)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT model_name FROM embeddings")
    remaining_models = {r[0] for r in cursor.fetchall()}
    assert "model_A" not in remaining_models
    assert "model_B" in remaining_models
    assert "model_C" in remaining_models
    conn.close()
    cache_lru.close()
    
    # Clear database
    os.remove(db_path)
    
    # 4. Test TTL invalidation (max_age_days = 5)
    cache_ttl = SqliteEmbeddingCache(
        db_path=db_path,
        invalidation_strategy="ttl",
        max_models=3,
        max_age_days=5
    )
    cache_ttl._init_db()
    
    conn = sqlite3.connect(db_path)
    now = int(time.time() * 1000)
    # model_A is 10 days old (stale)
    # model_B is 2 days old (fresh)
    with conn:
        conn.execute("INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES ('model_A', ?)", (now - 10 * 86400 * 1000,))
        conn.execute("INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES ('model_B', ?)", (now - 2 * 86400 * 1000,))
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h1', 't1', 'model_A', 1, '[]')")
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h2', 't2', 'model_B', 1, '[]')")
        conn.execute("INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES ('h3', 't3', 'model_C', 1, '[]')")
    conn.close()
    
    cache_ttl.invalidate_other_models("model_C")
    
    # model_A should be pruned since it is older than 5 days. model_B and model_C should remain.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT model_name FROM embeddings")
    remaining_models = {r[0] for r in cursor.fetchall()}
    assert "model_A" not in remaining_models
    assert "model_B" in remaining_models
    assert "model_C" in remaining_models
    conn.close()
    cache_ttl.close()



