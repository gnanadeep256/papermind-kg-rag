from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict

# Flat representation of retrieved context chunks to decouple from business domain models
class ChunkSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    arxiv_id: str
    section: str
    page_start: int
    page_end: int
    word_count: int
    semantic_score: float
    reranker_score: float
    graph_bonus: float
    combined_score: float

class QueryTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    user_query: str
    detected_intent: str
    selected_policy: str
    run_mode: str
    evaluation_version: str

class RetrievalTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    retrieved_papers: List[str]
    retrieved_chunks: List[ChunkSummary]
    graph_nodes_count: int
    graph_edges_count: int
    context_hash: Optional[str]
    context_tokens_estimated: int

class PolicyTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy_selected: str
    fallback_used: bool
    tier_distribution: Dict[str, int]
    budget_limit_words: int
    budget_used_words: int
    budget_utilization: float
    graph_overlap_ratio: float
    semantic_weight: float
    graph_weight: float
    execution_time_ms: float

class CacheTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    hits: int
    misses: int
    hit_rate: float
    lookup_latency_ms: float
    insert_latency_ms: float
    invalidation_strategy: str
    models_pruned: List[str]

class GenerationTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider_attempted: str
    provider_used: str
    fallback_used: bool
    fallback_reason: Optional[str]
    model_id: str
    prompt_hash: str
    prompt_text: Optional[str]
    system_template_hash: str
    temperature: float
    max_tokens: int
    # Token count metrics
    prompt_tokens_actual: Optional[int] = None
    completion_tokens_actual: Optional[int] = None
    total_tokens_actual: Optional[int] = None
    prompt_tokens_estimated: int
    completion_tokens_estimated: int
    total_tokens_estimated: int
    # Cost & timing
    estimated_cost: float
    latency_ms: float

class CitationTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_citations_count: int
    validated_citations: List[Dict[str, Any]]  # Primitive serializations only
    rejected_citations: List[Dict[str, Any]]
    semantic_alignment_scores: List[float]
    citation_precision: Optional[float]
    citation_coverage: float

class ConfidenceTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    semantic_score: float
    graph_score: float
    citation_score: float
    reranker_score: float
    final_confidence: float
    confidence_breakdown: Dict[str, Any]
    abstention_trigger: bool
    drift_score: float

class ExperimentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    git_sha: str
    python_version: str
    platform_os: str
    embedding_model: str
    judge_model: str
    fallback_model: str
    config_hash: str
    dependency_lock_hash: str
    benchmark_version: str

class TraceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    span_id: str
    name: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = {}

# Unified TraceContext container passed through the pipeline stages
class TraceContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Trace Schema Version
    trace_schema_version: str = "1.0.0"
    
    # Identity Hierarchy
    experiment_id: str
    run_id: str
    query_id: str
    trace_id: str
    
    # Stages Telemetry
    query: Optional[QueryTrace] = None
    retrieval: Optional[RetrievalTrace] = None
    policy: Optional[PolicyTrace] = None
    cache: Optional[CacheTrace] = None
    generation: Optional[GenerationTrace] = None
    citation: Optional[CitationTrace] = None
    confidence: Optional[ConfidenceTrace] = None
    metadata: Optional[ExperimentMetadata] = None
    
    # Distributed tracing compatible spans
    spans: List[TraceSpan] = []
    
    # Latency Waterfall (Inclusive/Exclusive Stage Timings)
    latency_waterfall_ms: Dict[str, float] = {}
