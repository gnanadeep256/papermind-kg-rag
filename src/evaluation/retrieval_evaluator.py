import math
from collections import Counter
from typing import Dict, Any, List
from src.evaluation.base_evaluator import BaseEvaluator

def compute_shannon_entropy(items: List[str]) -> float:
    """Computes Shannon entropy (base 2) of a list of items to measure distribution spread."""
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)

def get_chunk_field(c: Any, field: str, default: Any = "") -> Any:
    """Defensively gets a field from a chunk which can be a Pydantic model, dictionary, or mock object."""
    if hasattr(c, field):
        val = getattr(c, field)
        if val is not None:
            return val
    if isinstance(c, dict):
        return c.get(field, default)
    if hasattr(c, "get"):
        try:
            return c.get(field, default)
        except Exception:
            pass
    return default

class RetrievalEvaluator(BaseEvaluator):
    """Evaluates vector context and graph context retrieval quality and diversity."""
    
    def __init__(self, embedding_model: Any = None, compute_pairwise_metrics: bool = True, max_pairwise_chunks: int = 10) -> None:
        self.embedding_model = embedding_model
        self.compute_pairwise_metrics = compute_pairwise_metrics
        self.max_pairwise_chunks = max_pairwise_chunks
        
    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        expected_papers = query_case.get("expected_papers", [])
        expected_entities = query_case.get("expected_entities", [])
        
        # Unpack retrieved vector chunks
        retrieval_result = generator_result.get("retrieval_result")
        if retrieval_result:
            retrieved_chunks = retrieval_result.vector_context
            graph_context = retrieval_result.graph_context
        else:
            retrieved_chunks = generator_result.get("vector_context", [])
            graph_context = generator_result.get("graph_context", {"nodes": [], "relationships": []})
            
        num_chunks = len(retrieved_chunks)
        
        # Calculate new diversity metrics from graph nodes
        nodes = graph_context.get("nodes", [])
        
        methods = [n.get("entity_id") or n.get("id") for n in nodes if n.get("entity_type") == "Method"]
        datasets = [n.get("entity_id") or n.get("id") for n in nodes if n.get("entity_type") == "Dataset"]
        authors = [n.get("entity_id") or n.get("id") for n in nodes if n.get("entity_type") == "Author"]
        
        methods = [m for m in methods if m]
        datasets = [d for d in datasets if d]
        authors = [a for a in authors if a]
        
        method_diversity = len(set(methods)) / len(methods) if methods else 1.0
        dataset_diversity = len(set(datasets)) / len(datasets) if datasets else 1.0
        author_diversity = len(set(authors)) / len(authors) if authors else 1.0
        
        # Shannon entropy
        method_entropy = compute_shannon_entropy(methods)
        dataset_entropy = compute_shannon_entropy(datasets)
        
        if num_chunks == 0:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0 if (expected_papers or expected_entities) else 1.0,
                "paper_recall": 0.0 if expected_papers else 1.0,
                "entity_recall": 0.0 if expected_entities else 1.0,
                "paper_diversity": 1.0,
                "entity_diversity": 1.0,
                "method_diversity": float(method_diversity),
                "dataset_diversity": float(dataset_diversity),
                "author_diversity": float(author_diversity),
                "section_diversity": 1.0,
                "paper_entropy": 0.0,
                "method_entropy": float(method_entropy),
                "dataset_entropy": float(dataset_entropy),
                "avg_pairwise_distance": None,
                "min_pairwise_distance": None,
                "pairwise_distance_std": None,
                "context_redundancy_score": None
            }
            
        # 1. Context Precision
        # A chunk is relevant if its arxiv_id matches or if it contains any expected entity
        relevant_count = 0
        retrieved_papers = []
        retrieved_text_flat = []
        sections = []
        
        for c in retrieved_chunks:
            c_arxiv_id = get_chunk_field(c, "arxiv_id", "")
            c_text = get_chunk_field(c, "text", "")
            c_title = get_chunk_field(c, "title", "")
            c_section = get_chunk_field(c, "section", "")
            
            if c_arxiv_id:
                retrieved_papers.append(c_arxiv_id)
            if c_section:
                sections.append(c_section)
                
            retrieved_text_flat.append(c_text.lower())
            
            is_relevant = False
            if c_arxiv_id in expected_papers:
                is_relevant = True
            else:
                for ent in expected_entities:
                    if ent.lower() in c_text.lower() or ent.lower() in c_title.lower():
                        is_relevant = True
                        break
            if is_relevant:
                relevant_count += 1
                
        context_precision = relevant_count / num_chunks
        
        # 2. Context Recall & Paper/Entity Recall
        flat_text = " ".join(retrieved_text_flat)
        unique_retrieved_papers = set(retrieved_papers)
        papers_recalled = sum(1 for p in expected_papers if p in unique_retrieved_papers)
        entities_recalled = sum(1 for ent in expected_entities if ent.lower() in flat_text)
        
        total_expected_count = len(expected_papers) + len(expected_entities)
        if total_expected_count > 0:
            context_recall = (papers_recalled + entities_recalled) / total_expected_count
        else:
            context_recall = 1.0
            
        paper_recall = 1.0 if not expected_papers else (1.0 if papers_recalled > 0 else 0.0)
        entity_recall = 1.0 if not expected_entities else (entities_recalled / len(expected_entities))
        
        # 3. Paper Diversity & Entropy
        paper_diversity = len(unique_retrieved_papers) / num_chunks
        paper_entropy = compute_shannon_entropy(retrieved_papers)
        
        # 4. Section Diversity
        section_diversity = len(set(sections)) / len(sections) if sections else 1.0
        
        # 5. Entity Diversity in Graph Context
        unique_nodes = {n.get("entity_id") or n.get("id") for n in nodes if (n.get("entity_id") or n.get("id"))}
        entity_diversity = len(unique_nodes) / len(nodes) if nodes else 1.0
        
        # Calculate pairwise chunk distances
        avg_pairwise_distance = None
        min_pairwise_distance = None
        pairwise_distance_std = None
        context_redundancy_score = None
        
        if self.compute_pairwise_metrics and self.embedding_model is not None and num_chunks >= 2:
            try:
                import numpy as np
                sliced_chunks = retrieved_chunks[:self.max_pairwise_chunks]
                texts = [get_chunk_field(c, "text", "") for c in sliced_chunks]
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
                
                dists = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        dot_prod = np.dot(embeddings[i], embeddings[j])
                        cos_dist = float(max(0.0, min(2.0, 1.0 - dot_prod)))
                        dists.append(cos_dist)
                        
                if dists:
                    avg_pairwise_distance = float(np.mean(dists))
                    min_pairwise_distance = float(np.min(dists))
                    pairwise_distance_std = float(np.std(dists))
                    context_redundancy_score = float(max(0.0, min(1.0, 1.0 - avg_pairwise_distance)))
            except Exception:
                pass
        
        return {
            "context_precision": float(context_precision),
            "context_recall": float(context_recall),
            "paper_recall": float(paper_recall),
            "entity_recall": float(entity_recall),
            "paper_diversity": float(paper_diversity),
            "entity_diversity": float(entity_diversity),
            "method_diversity": float(method_diversity),
            "dataset_diversity": float(dataset_diversity),
            "author_diversity": float(author_diversity),
            "section_diversity": float(section_diversity),
            "paper_entropy": float(paper_entropy),
            "method_entropy": float(method_entropy),
            "dataset_entropy": float(dataset_entropy),
            "avg_pairwise_distance": avg_pairwise_distance,
            "min_pairwise_distance": min_pairwise_distance,
            "pairwise_distance_std": pairwise_distance_std,
            "context_redundancy_score": context_redundancy_score
        }
