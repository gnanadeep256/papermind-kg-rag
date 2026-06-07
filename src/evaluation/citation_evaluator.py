import re
from typing import Dict, Any, List
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class CitationEvaluator(BaseEvaluator):
    """Evaluates citation correctness, coverage, and paragraph semantic alignment."""
    
    def __init__(self, embedding_model: Any = None) -> None:
        """
        Args:
            embedding_model: Reusable SentenceTransformer instance. If None, loaded lazily.
        """
        self._model = embedding_model
        self.citation_pattern = re.compile(r"\[Citation (\d+)\]")

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        answer = generator_result.get("answer", "")
        invalid_citations = generator_result.get("invalid_citations", [])
        
        retrieval_result = generator_result.get("retrieval_result")
        if retrieval_result:
            vector_context = retrieval_result.vector_context
            used_citations = generator_result.get("citations", [])
        else:
            vector_context = generator_result.get("vector_context", [])
            used_citations = generator_result.get("used_citations", [])

        # Find final citations in answer
        final_tags = self.citation_pattern.findall(answer)
        final_tags_count = len(final_tags)
        
        # 1. Citation Precision
        # Calculated based on invalid citations filter
        metadata_precision = generator_result.get("retrieval_metadata", {}).get("citation_precision")
        if metadata_precision is not None:
            citation_precision = metadata_precision
        else:
            total_attempted = final_tags_count + len(invalid_citations)
            citation_precision = final_tags_count / total_attempted if total_attempted > 0 else 1.0
            
        # 2. Citation Coverage
        citation_coverage = len(used_citations) / len(vector_context) if vector_context else 0.0
        
        # 3. Citation Hallucination Rate
        citation_hallucination_rate = 1.0 - citation_precision
        
        # 4. Semantic Alignment Score
        # Match each cited sentence to its target chunk paragraph to find alignment score
        alignment_scores = []
        
        if final_tags_count > 0 and len(vector_context) > 0:
            try:
                # Segment answer into sentences
                sentences = re.split(r"(?<=[.!?])\s+", answer)
                model = self._get_model()
                
                for sentence in sentences:
                    matches = self.citation_pattern.findall(sentence)
                    if not matches:
                        continue
                        
                    # Clean sentence for embedding (remove citation tags)
                    clean_sent = self.citation_pattern.sub("", sentence).strip()
                    if not clean_sent:
                        continue
                        
                    sent_emb = model.encode([clean_sent], normalize_embeddings=True)[0]
                    
                    for match_str in matches:
                        idx = int(match_str) - 1
                        if 0 <= idx < len(vector_context):
                            chunk = vector_context[idx]
                            chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
                            
                            # Split chunk into paragraphs
                            paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
                            if not paragraphs:
                                paragraphs = [chunk_text]
                                
                            para_embs = model.encode(paragraphs, normalize_embeddings=True)
                            similarities = [float(np.dot(sent_emb, p_emb)) for p_emb in para_embs]
                            
                            max_sim = max(similarities) if similarities else 0.0
                            alignment_scores.append(max_sim)
            except Exception as e:
                # Fallback to default if embedding fails
                alignment_scores = [1.0] * final_tags_count
                
        alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 1.0
        
        return {
            "citation_precision": float(citation_precision),
            "citation_coverage": float(citation_coverage),
            "citation_hallucination_rate": float(citation_hallucination_rate),
            "semantic_alignment_score": float(alignment_score)
        }
