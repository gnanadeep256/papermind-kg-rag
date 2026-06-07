import os
import gzip
import json
import pytest
from src.observability.trace_models import (
    TraceContext, QueryTrace, RetrievalTrace, ChunkSummary, PolicyTrace,
    CacheTrace, GenerationTrace, CitationTrace, ConfidenceTrace, ExperimentMetadata
)
from src.observability.cost_estimator import CostEstimatorFactory, GeminiCostEstimator, GroqCostEstimator
from src.observability.experiment_manager import ExperimentManager
from src.observability.trace_manager import LocalTracer
from src.observability.trace_visualizer import TraceVisualizer

def test_telemetry_models_serialization():
    # Verify that we can instantiate models and do recursive Pydantic serialization
    query_trace = QueryTrace(
        timestamp="2026-06-07T12:00:00Z",
        user_query="test query",
        detected_intent="comparison",
        selected_policy="method",
        run_mode="LIVE",
        evaluation_version="1.0.0"
    )
    
    chunk = ChunkSummary(
        chunk_id="c1",
        arxiv_id="2405.0001",
        section="Abstract",
        page_start=1,
        page_end=1,
        word_count=50,
        semantic_score=0.85,
        reranker_score=0.90,
        graph_bonus=0.05,
        combined_score=0.95
    )
    
    retrieval_trace = RetrievalTrace(
        retrieved_papers=["2405.0001"],
        retrieved_chunks=[chunk],
        graph_nodes_count=2,
        graph_edges_count=1,
        context_hash="abcd1234",
        context_tokens_estimated=65
    )
    
    policy_trace = PolicyTrace(
        policy_selected="method",
        fallback_used=False,
        tier_distribution={"tier1": 1},
        budget_limit_words=1200,
        budget_used_words=50,
        budget_utilization=50/1200,
        graph_overlap_ratio=1.0,
        semantic_weight=0.70,
        graph_weight=0.30,
        execution_time_ms=10.5
    )
    
    cache_trace = CacheTrace(
        hits=1,
        misses=1,
        hit_rate=0.5,
        lookup_latency_ms=1.2,
        insert_latency_ms=0.5,
        invalidation_strategy="lru",
        models_pruned=["old_model"]
    )
    
    gen_trace = GenerationTrace(
        provider_attempted="gemini",
        provider_used="gemini",
        fallback_used=False,
        fallback_reason=None,
        model_id="gemini-2.5-flash",
        prompt_hash="prompt_hash",
        prompt_text="Prompt text",
        system_template_hash="system_hash",
        temperature=0.0,
        max_tokens=1024,
        prompt_tokens_actual=120,
        completion_tokens_actual=35,
        total_tokens_actual=155,
        prompt_tokens_estimated=120,
        completion_tokens_estimated=35,
        total_tokens_estimated=155,
        estimated_cost=0.00001,
        latency_ms=250.0
    )
    
    citation_trace = CitationTrace(
        generated_citations_count=1,
        validated_citations=[{"paper_title": "Paper 1"}],
        rejected_citations=[],
        semantic_alignment_scores=[0.85],
        citation_precision=1.0,
        citation_coverage=1.0
    )
    
    conf_trace = ConfidenceTrace(
        semantic_score=0.85,
        graph_score=0.5,
        citation_score=1.0,
        reranker_score=0.90,
        final_confidence=0.82,
        confidence_breakdown={"semantic": 0.85},
        abstention_trigger=False,
        drift_score=0.98
    )
    
    meta = ExperimentMetadata(
        git_sha="123456",
        python_version="3.11.9",
        platform_os="Windows",
        embedding_model="bge-small",
        judge_model="gemini",
        fallback_model="groq",
        config_hash="cfg_hash",
        dependency_lock_hash="lock_hash",
        benchmark_version="1.0.0"
    )
    
    context = TraceContext(
        experiment_id="exp_test",
        run_id="run_test",
        query_id="q_test",
        trace_id="t_test",
        query=query_trace,
        retrieval=retrieval_trace,
        policy=policy_trace,
        cache=cache_trace,
        generation=gen_trace,
        citation=citation_trace,
        confidence=conf_trace,
        metadata=meta,
        latency_waterfall_ms={"total": 260.5}
    )
    
    # Recursive serialization dump
    serialized = context.model_dump_json()
    data = json.loads(serialized)
    
    # Validate nested structure
    assert data["experiment_id"] == "exp_test"
    assert data["trace_schema_version"] == "1.0.0"
    assert data["spans"] == []
    assert data["query"]["user_query"] == "test query"
    assert data["retrieval"]["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert data["policy"]["budget_used_words"] == 50
    assert data["generation"]["model_id"] == "gemini-2.5-flash"
    assert data["citation"]["validated_citations"][0]["paper_title"] == "Paper 1"
    assert data["confidence"]["final_confidence"] == 0.82
    assert data["metadata"]["git_sha"] == "123456"
    assert data["latency_waterfall_ms"]["total"] == 260.5

def test_cost_estimators():
    # Test factory resolution
    gemini_est = CostEstimatorFactory.get_estimator("gemini")
    groq_est = CostEstimatorFactory.get_estimator("groq")
    
    assert isinstance(gemini_est, GeminiCostEstimator)
    assert isinstance(groq_est, GroqCostEstimator)
    
    # Test Gemini Pro vs Flash costs
    pro_cost = gemini_est.estimate_cost("gemini-2.5-pro", 1_000_000, 1_000_000)
    flash_cost = gemini_est.estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    
    assert pro_cost == pytest.approx(1.25 + 5.0)
    assert flash_cost == pytest.approx(0.075 + 0.30)
    
    # Test Groq costs
    llama_70b_cost = groq_est.estimate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    llama_8b_cost = groq_est.estimate_cost("llama-3.1-8b-instant", 1_000_000, 1_000_000)
    
    assert llama_70b_cost == pytest.approx(0.59 + 0.79)
    assert llama_8b_cost == pytest.approx(0.05 + 0.08)

def test_local_tracer_compression(tmp_path):
    # Setup test directories
    test_dir = str(tmp_path / "experiment_run")
    
    context = TraceContext(
        experiment_id="exp_comp",
        run_id="run_comp",
        query_id="q_comp",
        trace_id="t_comp"
    )
    
    # Test raw JSONL logging
    raw_tracer = LocalTracer(test_dir, compress=False)
    raw_tracer.log_trace(context)
    raw_tracer.close()
    
    raw_file = os.path.join(test_dir, "traces.jsonl")
    assert os.path.exists(raw_file)
    with open(raw_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["experiment_id"] == "exp_comp"
        
    # Test Gzipped JSONL logging
    gz_tracer = LocalTracer(test_dir, compress=True)
    gz_tracer.log_trace(context)
    gz_tracer.close()
    
    gz_file = os.path.join(test_dir, "traces.jsonl.gz")
    assert os.path.exists(gz_file)
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["experiment_id"] == "exp_comp"

def test_trace_visualizer_markdown_summary(tmp_path):
    test_dir = str(tmp_path / "experiment_run")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create sample trace file
    trace_data = {
        "experiment_id": "exp_vis",
        "run_id": "run_vis",
        "query_id": "q_vis",
        "trace_id": "t_vis",
        "query": {
            "timestamp": "2026-06-07T12:00:00Z",
            "user_query": "What is GraphRAG?",
            "detected_intent": "research",
            "selected_policy": "research",
            "run_mode": "LIVE",
            "evaluation_version": "1.0.0"
        },
        "retrieval": {
            "retrieved_papers": ["2606.06492"],
            "retrieved_chunks": [],
            "graph_nodes_count": 5,
            "graph_edges_count": 3,
            "context_hash": "mock_hash",
            "context_tokens_estimated": 200
        },
        "policy": {
            "policy_selected": "research",
            "fallback_used": False,
            "tier_distribution": {"tier1": 3},
            "budget_limit_words": 1200,
            "budget_used_words": 150,
            "budget_utilization": 0.125,
            "graph_overlap_ratio": 0.8,
            "semantic_weight": 0.7,
            "graph_weight": 0.3,
            "execution_time_ms": 45.0
        },
        "cache": {
            "hits": 3,
            "misses": 1,
            "hit_rate": 0.75,
            "lookup_latency_ms": 4.5,
            "insert_latency_ms": 2.0,
            "invalidation_strategy": "lru",
            "models_pruned": []
        },
        "generation": {
            "provider_attempted": "gemini",
            "provider_used": "gemini",
            "fallback_used": False,
            "fallback_reason": None,
            "model_id": "gemini-2.5-flash",
            "prompt_hash": "p_hash",
            "prompt_text": None,
            "system_template_hash": "s_hash",
            "temperature": 0.0,
            "max_tokens": 1024,
            "prompt_tokens_estimated": 150,
            "completion_tokens_estimated": 45,
            "total_tokens_estimated": 195,
            "estimated_cost": 0.00002,
            "latency_ms": 320.0
        },
        "citation": {
            "generated_citations_count": 1,
            "validated_citations": [],
            "rejected_citations": [],
            "semantic_alignment_scores": [],
            "citation_precision": 1.0,
            "citation_coverage": 1.0
        },
        "confidence": {
            "semantic_score": 0.82,
            "graph_score": 0.75,
            "citation_score": 1.0,
            "reranker_score": 0.85,
            "final_confidence": 0.81,
            "confidence_breakdown": {},
            "abstention_trigger": False,
            "drift_score": 0.95
        },
        "latency_waterfall_ms": {
            "retrieval": 45.0,
            "generation": 320.0,
            "total_query": 380.0
        },
        "metadata": {
            "git_sha": "commit_sha",
            "python_version": "3.11",
            "platform_os": "Windows",
            "embedding_model": "bge-small",
            "judge_model": "gemini",
            "fallback_model": "groq",
            "config_hash": "c_hash",
            "dependency_lock_hash": "d_hash",
            "benchmark_version": "1.0.0"
        }
    }
    
    traces_file = os.path.join(test_dir, "traces.jsonl")
    with open(traces_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(trace_data) + "\n")
        
    summary_path = TraceVisualizer.visualize(test_dir)
    assert os.path.exists(summary_path)
    
    with open(summary_path, "r", encoding="utf-8") as f:
        content = f.read()
        
        # Verify section titles
        assert "# Execution Trace Summary" in content
        assert "## Experiment Metadata" in content
        assert "## Cost & Token Accounting" in content
        assert "## Latency Waterfall" in content
        assert "## Policy Routing & Cache Stats" in content
        assert "## Confidence & Abstention Summary" in content
        
        # Verify specific stats
        assert "exp_vis" in content
        assert "gemini-2.5-flash" in content
        assert "Average Confidence (All Queries)" in content
        assert "Average Confidence (Answered Queries)" in content
        assert "Average Confidence (Abstention Queries)" in content
        assert "0.810" in content  # Confidence
        assert "75.00%" in content # Cache hit rate
        assert "Research-KG-RAG" not in content # Ensure no typo
        
        # Crucial check: EMOJI BAN. Assure absolutely no raw emojis are written to markdown summaries
        assert "✅" not in content
        assert "❌" not in content
        assert "⭐" not in content
        assert "🔥" not in content
        assert "⚠️" not in content

def test_experiment_manager(tmp_path):
    test_dir = str(tmp_path / "experiments")
    config_file = str(tmp_path / "config.yaml")
    
    # Create fake config.yaml
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("test_property: hello\nevaluation:\n  version: 2.0.0\n")
        
    exp_mgr = ExperimentManager(base_dir=test_dir, config_path=config_file)
    config_dict = {
        "evaluation": {
            "version": "2.0.0",
            "judge_model_provenance": "Gemini 2.5 Pro",
            "fallback_judge_provenance": "Groq Llama"
        },
        "embeddings": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
    
    exp_dir = exp_mgr.setup_experiment(config_dict)
    
    assert os.path.exists(exp_dir)
    assert os.path.exists(os.path.join(exp_dir, "config_snapshot.yaml"))
    assert os.path.exists(os.path.join(exp_dir, "environment_metadata.json"))
    
    # Read environment metadata
    with open(os.path.join(exp_dir, "environment_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["benchmark_version"] == "2.0.0"
        assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert meta["judge_model"] == "Gemini 2.5 Pro"
