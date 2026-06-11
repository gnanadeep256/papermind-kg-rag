import time
from typing import Optional, List, Dict, Any
import numpy as np
import faiss
from src.retriever import BaseRetriever
from src.hybrid_retriever import RetrievalResult, Citation
from src.evidence.base_policy import SelectedEvidenceChunk
from src.utils.config import load_config
from loguru import logger

class TemporaryRetriever(BaseRetriever):
    """
    Temporary RAG retriever for session-only PDFs.
    Queries an in-memory FAISS index and conforms to the BaseRetriever interface.
    """
    def __init__(self, index: faiss.Index, chunks: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> None:
        self.index = index
        self.chunks = chunks
        self.metadata = metadata or {}
        self.config = load_config()
        self.vector_retriever = self # Satisfies citation_validator structure checks
        self.model = None # sentence_transformer model
        
    @property
    def model_name(self) -> str:
        return self.config.get("retrieval", {}).get("embedding_model", "BAAI/bge-small-en-v1.5")

    def load(self) -> None:
        """Loads embedding model dynamically."""
        from src.llm import get_embedding_model
        self.model = get_embedding_model(self.model_name)

    def retrieve(
        self, 
        query: str, 
        top_k_vector: Optional[int] = None, 
        top_k_graph: Optional[int] = None, 
        max_chunks_per_paper: int = 2, 
        category: Optional[str] = None
    ) -> RetrievalResult:
        start_time = time.time()
        
        # Adaptive Token Budgets (capped to prevent 413/429 TPM limits on free tier API keys)
        budgets = {
            "summary": 5000,
            "methodology": 5000,
            "comparison": 5000,
            "dataset": 4000,
            "evaluation": 4000,
            "definition": 4000,
            "why": 4000,
            "future_work": 5000,
            "limitations": 5000,
            "research": 5000
        }
        token_budget = budgets.get(category, 5000) if category else 5000
        
        if self.model is None:
            self.load()
            
        # 1. Encode query
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')
        
        # 2. Vector search on in-memory FAISS
        k_val = min(30, len(self.chunks))
        if k_val == 0:
            return RetrievalResult(
                query=query,
                graph_context={"nodes": [], "relationships": []},
                vector_context=[],
                source_papers=[],
                citations=[],
                retrieval_metadata={"intent": "research", "fusion_time_ms": 0.0}
            )
            
        scores, indices = self.index.search(q_emb, k=k_val)
        
        # 3. Create SelectedEvidenceChunk candidates
        candidates = []
        for rank_pos, rank_idx in enumerate(indices[0]):
            if rank_idx < 0 or rank_idx >= len(self.chunks):
                continue
            c_data = self.chunks[rank_idx]
            score_val = float(scores[0][rank_pos])
            
            # Extract section & page info
            section = c_data.get("section", "Unknown")
            page_start = c_data.get("page_start", 1)
            page_end = c_data.get("page_end", 1)
            title = c_data.get("title", self.metadata.get("title", "Uploaded Paper"))
            arxiv_id = c_data.get("arxiv_id", self.metadata.get("arxiv_id", "temp_paper"))
            
            text_val = c_data.get("text", "")
            word_count = len(text_val.split())
            
            explanations = [{"type": "semantic_match", "score": score_val}]
            
            candidates.append(SelectedEvidenceChunk(
                chunk_id=f"temp_chunk_{rank_idx}",
                arxiv_id=arxiv_id,
                title=title,
                section=section,
                page_start=page_start,
                page_end=page_end,
                chunk_word_count=word_count,
                text=text_val,
                similarity_score=score_val,
                combined_score=score_val,
                context_text=text_val,
                explanations=explanations
            ))
            
        # 4. Filter by token budget
        final_context = []
        total_tokens_used = 0
        for chunk in candidates:
            est_tokens = int(chunk.chunk_word_count * 1.3)
            if not final_context or (total_tokens_used + est_tokens <= token_budget):
                final_context.append(chunk)
                total_tokens_used += est_tokens
            else:
                break
                
        # Deduplicate source papers
        source_papers = []
        if final_context:
            arxiv_id = final_context[0].arxiv_id
            title = final_context[0].title
            source_papers.append({"arxiv_id": arxiv_id, "title": title})
            
        # 5. Build citations
        citations = []
        for chunk in final_context:
            citations.append(Citation(
                paper_title=chunk.title,
                arxiv_id=chunk.arxiv_id,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_id=chunk.chunk_id,
                similarity_score=chunk.similarity_score,
                graph_bonus=0.0,
                combined_score=chunk.combined_score,
                selected_by=["semantic_match"]
            ))
            
        fusion_time_ms = (time.time() - start_time) * 1000
        
        policy_stats = {
            "name": "research",
            "fallback_used": False,
            "tier_distribution": {
                "tier1": len(final_context),
                "tier2": 0,
                "tier3": 0
            },
            "evidence": {
                "words": total_tokens_used,
                "budget": token_budget,
                "utilization": float(total_tokens_used / token_budget) if token_budget > 0 else 0.0
            }
        }
        
        retrieval_metadata = {
            "intent": "research",
            "routing_strategy": "temporary_in_memory",
            "token_budget": token_budget,
            "token_used": total_tokens_used,
            "vector_candidates": len(candidates),
            "graph_nodes": 0,
            "graph_relationships": 0,
            "policy": policy_stats,
            "fusion_time_ms": fusion_time_ms
        }
        
        return RetrievalResult(
            query=query,
            graph_context={"nodes": [], "relationships": []},
            vector_context=final_context,
            source_papers=source_papers,
            citations=citations,
            retrieval_metadata=retrieval_metadata
        )
