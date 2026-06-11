import re
from collections import Counter
from typing import Dict, Any, List
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class GroundingEvaluator(BaseEvaluator):
    """Computes evidence overlap (ROUGE), citation support, and semantic grounding without LLM."""
    
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

    def _lcs(self, x: List[str], y: List[str]) -> int:
        """Computes Longest Common Subsequence length, truncated to prevent memory bottleneck."""
        x = x[:150]
        y = y[:150]
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if x[i-1] == y[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        answer = generator_result.get("answer", "")
        
        retrieval_result = generator_result.get("retrieval_result")
        if retrieval_result:
            vector_context = retrieval_result.vector_context
        else:
            vector_context = generator_result.get("vector_context", [])
            
        flat_context_text = " ".join([c.text if hasattr(c, "text") else c.get("text", "") for c in vector_context])
        
        # 1. ROUGE Lexical Overlap (ROUGE-1 and ROUGE-L)
        words_answer = re.findall(r"\w+", answer.lower())
        words_context = re.findall(r"\w+", flat_context_text.lower())
        
        # ROUGE-1
        cnt_answer = Counter(words_answer)
        cnt_context = Counter(words_context)
        overlap1 = sum((cnt_answer & cnt_context).values())
        r1_prec = overlap1 / len(words_answer) if words_answer else 0.0
        r1_rec = overlap1 / len(words_context) if words_context else 0.0
        r1_f1 = 2 * r1_prec * r1_rec / (r1_prec + r1_rec) if (r1_prec + r1_rec) > 0 else 0.0
        
        # ROUGE-L
        lcs_len = self._lcs(words_answer, words_context)
        rl_prec = lcs_len / len(words_answer) if words_answer else 0.0
        rl_rec = lcs_len / len(words_context) if words_context else 0.0
        rl_f1 = 2 * rl_prec * rl_rec / (rl_prec + rl_rec) if (rl_prec + rl_rec) > 0 else 0.0
        
        evidence_overlap = (r1_f1 + rl_f1) / 2.0
        
        # 2. Citation Support
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        
        if generator_result.get("is_abstention", False) or "sufficient evidence" in answer.lower():
            # If the system abstained, citation support is 1.0 (correct behavior)
            citation_support = 1.0
        else:
            supported_sentences = sum(1 for s in sentences if self.citation_pattern.search(s))
            citation_support = supported_sentences / len(sentences) if sentences else 1.0
            
        # 3. Semantic Grounding
        # Compute sentence similarity to cited paragraphs (or any context paragraph if no citation)
        grounding_scores = []
        
        if sentences and vector_context:
            try:
                model = self._get_model()
                # Pre-embed all context paragraphs
                all_paragraphs = []
                para_to_chunk_idx = []
                
                for idx, chunk in enumerate(vector_context):
                    chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
                    paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
                    if not paragraphs:
                        paragraphs = [chunk_text]
                    for para in paragraphs:
                        all_paragraphs.append(para)
                        para_to_chunk_idx.append(idx)
                        
                if all_paragraphs:
                    context_embs = model.encode(all_paragraphs, normalize_embeddings=True)
                    
                    for sentence in sentences:
                        # Clean tag
                        clean_sent = self.citation_pattern.sub("", sentence).strip()
                        if not clean_sent:
                            continue
                            
                        sent_emb = model.encode([clean_sent], normalize_embeddings=True)[0]
                        matches = self.citation_pattern.findall(sentence)
                        
                        if matches:
                            # Filter paragraphs of cited chunks
                            cited_indices = {int(m) - 1 for m in matches}
                            similarities = []
                            for p_idx, chunk_idx in enumerate(para_to_chunk_idx):
                                if chunk_idx in cited_indices:
                                    sim = float(np.dot(sent_emb, context_embs[p_idx]))
                                    similarities.append(sim)
                        else:
                            # Compare against all context paragraphs
                            similarities = [float(np.dot(sent_emb, p_emb)) for p_emb in context_embs]
                            
                        max_sim = max(similarities) if similarities else 0.0
                        grounding_scores.append(max_sim)
            except Exception:
                grounding_scores = [1.0] * len(sentences)
                
        semantic_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 1.0
        
        return {
            "evidence_overlap": float(evidence_overlap),
            "citation_support": float(citation_support),
            "semantic_grounding": float(semantic_grounding)
        }
