from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Sequence
from pydantic import BaseModel, ConfigDict, Field
from loguru import logger

class WeightBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    semantic: float = 0.0
    graph_overlap: float = 0.0
    reranker: float = 0.0
    graph_bonus: float = 0.0

class RetrievalRanking(BaseModel):
    model_config = ConfigDict(frozen=True)
    semantic_score: float
    graph_overlap: float
    reranker_score: Optional[float] = None
    graph_bonus: float
    final_score: float
    weight_breakdown: WeightBreakdown

class RetrievalMerge(BaseModel):
    model_config = ConfigDict(frozen=True)
    merged_chunks: int
    merged_word_count: int
    merged_chunk_ids: List[str]
    provenance_sources: List[str]

class RetrievalSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    matched_entity: Optional[str] = None
    matched_dataset: Optional[str] = None
    matched_method: Optional[str] = None

class RetrievalReason(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy: str
    tier: int
    strategy: str
    ranking: RetrievalRanking
    source: RetrievalSource
    merge: RetrievalMerge

class SelectedEvidenceChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str = ""
    arxiv_id: str = ""
    title: str = ""
    section: str = ""
    page_start: int = 1
    page_end: int = 1
    chunk_word_count: int = 0
    text: str = ""
    similarity_score: float = 0.0
    reranker_score: Optional[float] = None
    graph_bonus: float = 0.0
    combined_score: float = 0.0
    context_text: str = ""
    explanations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_reason: Optional[RetrievalReason] = None

class TierDistribution(BaseModel):
    model_config = ConfigDict(frozen=True)
    tier1: int
    tier2: int
    tier3: int

class EvidenceTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)
    fallback_used: bool
    evidence_budget: int
    evidence_words: int
    utilization: float
    tier_distribution: TierDistribution

class EvidencePolicyResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    selected_chunks: Sequence[SelectedEvidenceChunk]
    telemetry: EvidenceTelemetry
    evidence_words: int
    utilization: float

class BaseEvidencePolicy(ABC):
    """Abstract base class for evidence routing, classification, scoring, and budgeting."""
    def __init__(self, retrieval_config: Dict[str, Any], entity_cache: Dict[str, Dict[str, Any]]) -> None:
        self.retrieval_config = retrieval_config
        self.entity_cache = entity_cache

    @abstractmethod
    def classify(
        self,
        unpacked_chunks: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Classifies raw chunk dictionaries into tier1, tier2, and tier3 candidates."""
        pass

    @abstractmethod
    def score(
        self,
        tier1: List[Dict[str, Any]],
        tier2: List[Dict[str, Any]],
        tier3: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculates features, scores, and assigns retrieval_reason dictionary to chunks."""
        pass

    @abstractmethod
    def rank(
        self,
        tier1: List[Dict[str, Any]],
        tier2: List[Dict[str, Any]],
        tier3: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Sorts chunks within each tier using policy configuration weights and thresholds."""
        pass

    def assemble(
        self,
        intent: str,
        tier1: List[Dict[str, Any]],
        tier2: List[Dict[str, Any]],
        tier3: List[Dict[str, Any]]
    ) -> EvidencePolicyResult:
        """Assembles budgeted chunks sequentially from tiers, converts to typed chunks, and compiles telemetry."""
        # Retrieve min words budget from configuration or fallback to default
        policy_config = self.retrieval_config.get("policies", {}).get(intent, {})
        budgeting_config = policy_config.get("budgeting", {})
        min_words = budgeting_config.get("min_words", self.retrieval_config.get("min_words_evidence_budget", 1200))

        selected_chunks_raw = []
        current_words = 0
        
        tier1_selected = 0
        tier2_selected = 0
        tier3_selected = 0
        fallback_used = False

        # 1. Add Tier 1 chunks
        for chunk in tier1:
            selected_chunks_raw.append(chunk)
            current_words += chunk.get("chunk_word_count", len(chunk["text"].split()))
            tier1_selected += 1

        # Check if budget is satisfied
        if current_words < min_words:
            # 2. Add Tier 2 chunks
            for chunk in tier2:
                selected_chunks_raw.append(chunk)
                current_words += chunk.get("chunk_word_count", len(chunk["text"].split()))
                tier2_selected += 1
                fallback_used = True
                if current_words >= min_words:
                    break

        # Check if budget is satisfied
        if current_words < min_words:
            # 3. Add Tier 3 chunks
            for chunk in tier3:
                selected_chunks_raw.append(chunk)
                current_words += chunk.get("chunk_word_count", len(chunk["text"].split()))
                tier3_selected += 1
                fallback_used = True
                if current_words >= min_words:
                    break

        # Instantiate final SelectedEvidenceChunk objects
        selected_chunks = []
        for c in selected_chunks_raw:
            reason_dict = c.get("retrieval_reason")
            reason_obj = None
            if reason_dict:
                # Build RetrievalReason object from dict
                reason_obj = RetrievalReason(
                    policy=reason_dict["policy"],
                    tier=reason_dict["tier"],
                    strategy=reason_dict["strategy"],
                    ranking=RetrievalRanking(
                        semantic_score=reason_dict["ranking"]["semantic_score"],
                        graph_overlap=reason_dict["ranking"]["graph_overlap"],
                        reranker_score=reason_dict["ranking"].get("reranker_score"),
                        graph_bonus=reason_dict["ranking"]["graph_bonus"],
                        final_score=reason_dict["ranking"]["final_score"],
                        weight_breakdown=WeightBreakdown(
                            semantic=reason_dict["ranking"]["weight_breakdown"].get("semantic", 0.0),
                            graph_overlap=reason_dict["ranking"]["weight_breakdown"].get("graph_overlap", 0.0),
                            reranker=reason_dict["ranking"]["weight_breakdown"].get("reranker", 0.0),
                            graph_bonus=reason_dict["ranking"]["weight_breakdown"].get("graph_bonus", 0.0)
                        )
                    ),
                    source=RetrievalSource(
                        matched_entity=reason_dict["source"].get("matched_entity"),
                        matched_dataset=reason_dict["source"].get("matched_dataset"),
                        matched_method=reason_dict["source"].get("matched_method")
                    ),
                    merge=RetrievalMerge(
                        merged_chunks=reason_dict["merge"]["merged_chunks"],
                        merged_word_count=reason_dict["merge"]["merged_word_count"],
                        merged_chunk_ids=reason_dict["merge"]["merged_chunk_ids"],
                        provenance_sources=reason_dict["merge"]["provenance_sources"]
                    )
                )

            selected_chunks.append(SelectedEvidenceChunk(
                chunk_id=c["chunk_id"],
                arxiv_id=c["arxiv_id"],
                title=c["title"],
                section=c.get("section", "Unknown"),
                page_start=c.get("page_start", 1),
                page_end=c.get("page_end", 1),
                chunk_word_count=c.get("chunk_word_count", len(c["text"].split())),
                text=c["text"],
                similarity_score=c["similarity_score"],
                reranker_score=c.get("reranker_score"),
                graph_bonus=c["graph_bonus"],
                combined_score=c.get("combined_score", c["combined_score"]),
                context_text=c["context_text"],
                explanations=c.get("explanations", []),
                retrieval_reason=reason_obj
            ))

        utilization = float(current_words / min_words) if min_words > 0 else 0.0

        telemetry = EvidenceTelemetry(
            fallback_used=fallback_used,
            evidence_budget=min_words,
            evidence_words=current_words,
            utilization=utilization,
            tier_distribution=TierDistribution(
                tier1=tier1_selected,
                tier2=tier2_selected,
                tier3=tier3_selected
            )
        )

        return EvidencePolicyResult(
            selected_chunks=selected_chunks,
            telemetry=telemetry,
            evidence_words=current_words,
            utilization=utilization
        )

    def execute(
        self,
        unpacked_chunks: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> EvidencePolicyResult:
        """Executes the complete evidence retrieval pipeline lifecycle."""
        # Retrieve policy name dynamically from class naming
        intent = "research"
        class_name = self.__class__.__name__.lower()
        if "dataset" in class_name:
            intent = "dataset"
        elif "method" in class_name:
            intent = "method"
        elif "paper" in class_name:
            intent = "paper"

        t1, t2, t3 = self.classify(unpacked_chunks, graph_context, query_entities)
        t1, t2, t3 = self.score(t1, t2, t3, graph_context, query_entities)
        t1, t2, t3 = self.rank(t1, t2, t3)
        return self.assemble(intent, t1, t2, t3)
