from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Sequence
from pydantic import BaseModel
from src.evidence.base_policy import RetrievalReason, SelectedEvidenceChunk

class RetrievalExplanation(BaseModel):
    type: str  # "semantic_match", "graph_neighbor", "introduced_method", "entity_link", "cross_encoder_boost", "packed_context", "neighbor_expansion", "paper_match"
    score: Optional[float] = None
    path_length: Optional[int] = None

class Citation(BaseModel):
    paper_title: str
    arxiv_id: str
    section: str
    page_start: int
    page_end: int
    chunk_id: str
    similarity_score: float
    graph_bonus: float
    combined_score: float
    selected_by: List[str] = []
    retrieval_reason: Optional[RetrievalReason] = None

class RetrievalResult(BaseModel):
    query: str
    graph_context: Dict[str, Any]
    vector_context: Sequence[SelectedEvidenceChunk]
    source_papers: List[Dict[str, Any]]
    citations: List[Citation]
    retrieval_metadata: Dict[str, Any]

class BaseRetriever(ABC):
    """
    Abstract Base Class for PaperMind retrievers (HybridRetriever, TemporaryRetriever).
    Ensures consistent retrieve() signature and loading lifecycle.
    """
    @abstractmethod
    def load(self) -> None:
        """Initializes and loads underlying resources (models, indexes, caches)."""
        pass

    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k_vector: Optional[int] = None, 
        top_k_graph: Optional[int] = None, 
        max_chunks_per_paper: int = 2, 
        category: Optional[str] = None
    ) -> RetrievalResult:
        """Executes candidate matching and context compilation returning a RetrievalResult."""
        pass
