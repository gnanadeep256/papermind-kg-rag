from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.retriever import Citation
from src.observability.trace_models import TraceContext

class EvaluationContext(BaseModel):
    vector_context: List[Any]
    graph_context: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]

class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    provenance: Dict[str, Any]
    metadata: Dict[str, Any]
    evaluation_context: Optional[EvaluationContext] = None

    @property
    def abstained(self) -> bool:
        return self.metadata.get("abstained", False) or "do not have sufficient evidence" in self.answer.lower()

class BaseGenerator(ABC):
    """
    Abstract Base Class for PaperMind grounded generators.
    """
    @abstractmethod
    def generate_answer(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        trace_context: Optional[TraceContext] = None
    ) -> GenerationResult:
        """
        Synthesizes a grounded answer from retrieved context.
        """
        pass
