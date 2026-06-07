from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict

class LatencyPercentiles(BaseModel):
    model_config = ConfigDict(frozen=True)
    mean: float
    median: float
    p95: float
    p99: float

class CategoryBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    score: float
    faithfulness: float
    completeness: float
    citation_precision: float
    abstention_accuracy: float
    count: int

class RegressionComparison(BaseModel):
    model_config = ConfigDict(frozen=True)
    metric_name: str
    current_value: float
    previous_value: float
    differential: float
    label: str  # e.g., "Faithfulness: 96.1% (↑ +1.8%)"

class ConfidenceInterval(BaseModel):
    model_config = ConfigDict(frozen=True)
    lower: float
    upper: float
    width: float

class BenchmarkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    evaluation_version: str
    embedding_model: str
    judge_model: str
    fallback_model: str
    bootstrap_resamples: int
    confidence_level: float
    random_seed: int
    generated_at: str
    git_sha: str
    run_mode: str = "DRY-RUN"
    python_version: str = "unknown"
    sentence_transformer_version: str = "unknown"
    platform_os: str = "unknown"
    lockfile_hash: str = "unknown"



class QueryEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_id: str
    query: str
    category: str
    answer: str
    is_abstention: bool
    allow_abstain: bool
    
    # Latency breakdown (ms)
    latency_ms: float
    latency_breakdown: Dict[str, float]
    
    # Sufficiency & Abstention classification
    # values: "correct_answer", "correct_abstention", "false_abstention", "missed_abstention"
    abstention_mode: str
    abstention_accuracy: float
    
    # Retrieval Metrics
    context_precision: float
    context_recall: float
    paper_recall: float
    entity_recall: float
    paper_diversity: float
    entity_diversity: float
    
    # Expanded retrieval diversity metrics
    method_diversity: float = 1.0
    dataset_diversity: float = 1.0
    author_diversity: float = 1.0
    section_diversity: float = 1.0
    paper_entropy: float = 0.0
    method_entropy: float = 0.0
    dataset_entropy: float = 0.0
    
    # Policy Routing Metrics
    policy_selected: str
    policy_correct: bool
    policy_fallback: bool
    tier1_ratio: float
    tier2_ratio: float
    tier3_ratio: float
    budget_utilization: float
    graph_overlap_ratio: float
    
    # Citation Metrics
    citation_precision: float
    citation_coverage: float
    citation_hallucination_rate: float
    semantic_alignment_score: float
    
    # Grounding Metrics
    evidence_overlap: float
    citation_support: float
    semantic_grounding: float
    
    # Generation Metrics
    completeness: float
    groundedness: float
    llm_faithfulness: float
    hybrid_faithfulness: float
    
    # Robustness Metrics (computed by comparing variations)
    robustness_semantic_consistency: float
    robustness_entity_consistency: float
    robustness_score: float

    # Pairwise chunk distance metrics
    avg_pairwise_distance: Optional[float] = None
    min_pairwise_distance: Optional[float] = None
    pairwise_distance_std: Optional[float] = None
    context_redundancy_score: Optional[float] = None

class AggregatedEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_timestamp: str
    total_queries: int
    
    # overall scores
    overall_score: float
    micro_overall_score: float = 0.0
    macro_overall_score: float = 0.0
    weighted_macro_overall_score: float = 0.0
    
    # macro metric averages
    avg_retrieval_recall: float
    avg_citation_precision: float
    avg_hybrid_faithfulness: float
    avg_groundedness: float
    avg_abstention_accuracy: float
    avg_robustness: float
    
    # Latency Profile
    avg_latency_ms: float
    latency_percentiles: LatencyPercentiles
    
    # Calibration Diagnostics
    expected_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Diversity
    avg_paper_diversity: float
    avg_entity_diversity: float
    
    # Expanded diversity metrics
    avg_method_diversity: float = 1.0
    avg_dataset_diversity: float = 1.0
    avg_author_diversity: float = 1.0
    avg_section_diversity: float = 1.0
    avg_paper_entropy: float = 0.0
    avg_method_entropy: float = 0.0
    avg_dataset_entropy: float = 0.0
    
    # Pairwise chunk distance aggregations
    avg_pairwise_distance: Optional[float] = None
    avg_min_pairwise_distance: Optional[float] = None
    avg_pairwise_distance_std: Optional[float] = None
    avg_context_redundancy_score: Optional[float] = None
    
    # Policy performance summary
    policy_routing_accuracy: float
    policy_fallback_rate: float
    avg_budget_utilization: float
    
    # Versioning & Provenance
    evaluation_version: str = "1.0.0"
    embedding_model_provenance: str = "BAAI/bge-small-en-v1.5"
    judge_model_provenance: str = "Gemini 2.5 Flash"
    fallback_judge_provenance: str = "Llama-3.3-70B"
    evaluation_commit_hash: str = "unknown"
    
    # Bootstrap Confidence Intervals
    overall_score_ci: Optional[ConfidenceInterval] = None
    micro_overall_score_ci: Optional[ConfidenceInterval] = None
    macro_overall_score_ci: Optional[ConfidenceInterval] = None
    weighted_macro_overall_score_ci: Optional[ConfidenceInterval] = None
    
    avg_retrieval_recall_ci: Optional[ConfidenceInterval] = None
    avg_citation_precision_ci: Optional[ConfidenceInterval] = None
    avg_hybrid_faithfulness_ci: Optional[ConfidenceInterval] = None
    avg_groundedness_ci: Optional[ConfidenceInterval] = None
    avg_abstention_accuracy_ci: Optional[ConfidenceInterval] = None
    avg_robustness_ci: Optional[ConfidenceInterval] = None
    avg_pairwise_distance_ci: Optional[ConfidenceInterval] = None
    avg_context_redundancy_score_ci: Optional[ConfidenceInterval] = None
    
    # Separate Metadata block
    metadata: Optional[BenchmarkMetadata] = None
    
    # Breakdown per category (Leaderboard)
    category_breakdown: Dict[str, CategoryBreakdown]
    
    # Regression compared to last run
    regression_differentials: List[RegressionComparison] = []
