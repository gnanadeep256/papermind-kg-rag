from typing import Dict, Any, List, Set, Tuple
from src.hybrid_retriever import Citation, RetrievalResult

class ConfidenceEstimator:
    """
    Computes answer confidence and coverage metrics, and evaluates if the
    retrieved evidence is sufficient to proceed with generation.
    """
    def __init__(self, min_confidence_threshold: float = 0.45, weights: Dict[str, float] = None) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.weights = dict(weights) if weights else {
            "semantic": 0.45,
            "graph": 0.20,
            "citation_coverage": 0.10,
            "citation_precision": 0.10,
            "rerank": 0.15
        }
        # Backward compatibility for old citation key
        if "citation" in self.weights:
            val = self.weights.pop("citation")
            self.weights["citation_coverage"] = val / 2.0
            self.weights["citation_precision"] = val / 2.0

    def estimate(
        self,
        retrieval_result: RetrievalResult,
        used_citations: List[Citation],
        citation_precision: float = 1.0
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Estimates retrieval metrics:
        Returns:
            - confidence: float between 0.0 and 1.0
            - coverage: float between 0.0 and 1.0
            - details: dict of individual scoring metrics and breakdown
        """
        vector_context = retrieval_result.vector_context
        graph_context = retrieval_result.graph_context
        metadata = retrieval_result.retrieval_metadata
        
        # 1. Average Semantic Similarity (FAISS score)
        similarities = []
        for c in vector_context:
            if hasattr(c, "similarity_score"):
                similarities.append(c.similarity_score)
            elif isinstance(c, dict):
                similarities.append(c.get("similarity_score", 0.0))
            else:
                similarities.append(getattr(c, "similarity_score", 0.0))
        semantic_score = sum(similarities) / len(similarities) if similarities else 0.0
        
        # 2. Graph Connectivity
        # Graph connectivity = reachable entities / matched entities in query
        query_entity_ids = metadata.get("query_entities", [])
        if not query_entity_ids:
            graph_connectivity = 1.0
        else:
            graph_node_ids = {n.get("entity_id") or n.get("id") for n in graph_context.get("nodes", []) if (n.get("entity_id") or n.get("id"))}
            matched_count = sum(1 for eid in query_entity_ids if eid in graph_node_ids)
            graph_connectivity = matched_count / len(query_entity_ids)
            
        # 3. Citation Density / Answer Coverage
        available_count = len(vector_context)
        used_count = len(used_citations)
        citation_coverage = used_count / available_count if available_count > 0 else 0.0
        coverage = citation_coverage
        
        # 4. Average Reranker Score
        reranker_scores = []
        for c in vector_context:
            if hasattr(c, "reranker_score"):
                val = c.reranker_score if c.reranker_score is not None else (c.similarity_score if hasattr(c, "similarity_score") else 0.0)
                reranker_scores.append(val)
            elif isinstance(c, dict):
                val = c.get("reranker_score")
                if val is None:
                    val = c.get("similarity_score", 0.0)
                reranker_scores.append(val)
            else:
                val = getattr(c, "reranker_score", None)
                if val is None:
                    val = getattr(c, "similarity_score", 0.0)
                reranker_scores.append(val)
        avg_reranker_score = sum(reranker_scores) / len(reranker_scores) if reranker_scores else 0.0
        
        # Weighted Confidence Score Calculation
        w = self.weights
        confidence = (
            w.get("semantic", 0.45) * semantic_score +
            w.get("graph", 0.20) * graph_connectivity +
            w.get("citation_coverage", 0.10) * citation_coverage +
            w.get("citation_precision", 0.10) * citation_precision +
            w.get("rerank", 0.15) * avg_reranker_score
        )
        
        confidence_breakdown = {
            "semantic": float(semantic_score),
            "graph": float(graph_connectivity),
            "citation_coverage": float(citation_coverage),
            "citation_precision": float(citation_precision),
            "reranker": float(avg_reranker_score),
            "final": float(confidence)
        }
        
        details = {
            "avg_similarity": float(semantic_score),
            "graph_connectivity": float(graph_connectivity),
            "citation_density": float(citation_coverage), # Keep old key for backward compatibility
            "citation_coverage": float(citation_coverage),
            "citation_precision": float(citation_precision),
            "reranker_score": float(avg_reranker_score),
            "coverage": float(coverage),
            "confidence_breakdown": confidence_breakdown
        }
        
        return float(confidence), float(coverage), details

    def should_abstain(
        self,
        retrieval_result: RetrievalResult,
        confidence: float,
        details: Dict[str, Any]
    ) -> bool:
        """
        Determines if the answer generator should abstain based on evidence thresholds.
        """
        vector_context = retrieval_result.vector_context
        avg_similarity = details.get("avg_similarity", 0.0)
        
        # Strict Abstention guards:
        # - Context is empty
        if not vector_context:
            return True
            
        # - Semantic match quality is too low (< 0.50)
        if avg_similarity < 0.50:
            return True
            
        # - Combined confidence score is below threshold
        if confidence < self.min_confidence_threshold:
            return True
            
        return False
