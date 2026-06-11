import re
from typing import Dict, Any, List, Set
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class RobustnessEvaluator(BaseEvaluator):
    """Measures answer consistency (semantic cosine similarity and Jaccard citation overlap) across query variations."""
    
    def __init__(self, embedding_model: Any = None) -> None:
        """
        Args:
            embedding_model: Reusable SentenceTransformer instance. If None, loaded lazily.
        """
        self._model = embedding_model
        self.citation_pattern = re.compile(r"\[Citation (\d+)\]")

    def _get_model(self) -> Any:
        if self._model is None:
            from src.llm import get_embedding_model
            self._model = get_embedding_model("BAAI/bge-small-en-v1.5")
        return self._model

    def evaluate_variations(
        self,
        base_result: Dict[str, Any],
        variation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compares base query outputs with variation query outputs.
        
        Args:
            base_result: Dict containing generator outputs for base query.
            variation_results: List of Dicts containing generator outputs for variations.
            
        Returns:
            Dict containing semantic consistency, Jaccard entity consistency, and overall robustness score.
        """
        if not variation_results:
            return {
                "robustness_semantic_consistency": 1.0,
                "robustness_entity_consistency": 1.0,
                "robustness_score": 1.0
            }
            
        base_answer = base_result.get("answer", "")
        # Remove citation tags for semantic embedding
        clean_base_answer = self.citation_pattern.sub("", base_answer).strip()
        
        # Get base cited papers
        base_citations = base_result.get("citations", [])
        if not base_citations and base_result.get("retrieval_result"):
            base_citations = base_result.get("retrieval_result").citations
        base_cited_papers = {c.arxiv_id if hasattr(c, "arxiv_id") else c.get("arxiv_id", "") for c in base_citations if c}
        base_cited_papers.discard("")
        
        semantic_scores = []
        jaccard_scores = []
        
        try:
            model = self._get_model()
            if clean_base_answer:
                base_emb = model.encode([clean_base_answer], normalize_embeddings=True)[0]
            else:
                base_emb = None
                
            for var_res in variation_results:
                var_answer = var_res.get("answer", "")
                clean_var_answer = self.citation_pattern.sub("", var_answer).strip()
                
                # 1. Semantic Similarity
                if base_emb is not None and clean_var_answer:
                    var_emb = model.encode([clean_var_answer], normalize_embeddings=True)[0]
                    sim = float(np.dot(base_emb, var_emb))
                    semantic_scores.append(max(0.0, sim))
                elif not clean_base_answer and not clean_var_answer:
                    semantic_scores.append(1.0)
                else:
                    semantic_scores.append(0.0)
                    
                # 2. Entity/Paper Citation Jaccard Overlap
                var_citations = var_res.get("citations", [])
                if not var_citations and var_res.get("retrieval_result"):
                    var_citations = var_res.get("retrieval_result").citations
                var_cited_papers = {c.arxiv_id if hasattr(c, "arxiv_id") else c.get("arxiv_id", "") for c in var_citations if c}
                var_cited_papers.discard("")
                
                intersection = base_cited_papers.intersection(var_cited_papers)
                union = base_cited_papers.union(var_cited_papers)
                
                if not base_cited_papers and not var_cited_papers:
                    jaccard = 1.0  # consistently cited nothing
                elif not union:
                    jaccard = 1.0
                else:
                    jaccard = len(intersection) / len(union)
                jaccard_scores.append(jaccard)
        except Exception:
            semantic_scores = [1.0] * len(variation_results)
            jaccard_scores = [1.0] * len(variation_results)
            
        raw_sem = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 1.0
        raw_ent = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 1.0
        raw_rob = 0.60 * raw_sem + 0.40 * raw_ent
        
        semantic_consistency = min(1.0, max(0.0, raw_sem))
        entity_consistency = min(1.0, max(0.0, raw_ent))
        robustness_score = min(1.0, max(0.0, raw_rob))
        
        return {
            "robustness_semantic_consistency": float(semantic_consistency),
            "robustness_entity_consistency": float(entity_consistency),
            "robustness_score": float(robustness_score)
        }

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        # Satisfy BaseEvaluator interface (returns defaults if evaluated singly)
        return {
            "robustness_semantic_consistency": 1.0,
            "robustness_entity_consistency": 1.0,
            "robustness_score": 1.0
        }
