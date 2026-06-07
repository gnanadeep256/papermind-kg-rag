from typing import Dict, Any, List, Tuple
from loguru import logger
from src.evidence.base_policy import BaseEvidencePolicy

class ResearchEvidencePolicy(BaseEvidencePolicy):
    """Evidence Policy for general research and default intent queries."""

    def classify(
        self,
        unpacked_chunks: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tier1_chunks = []
        tier2_chunks = []
        tier3_chunks = []

        query_entity_ids = {ent["entity_id"] for ent in query_entities}

        for chunk in unpacked_chunks:
            P = chunk["arxiv_id"]
            similarity_score = chunk.get("similarity_score", 0.0)
            reranker_score = chunk.get("reranker_score", similarity_score)

            is_connected = False
            if query_entity_ids:
                for rel in graph_context.get("relationships", []):
                    if (rel["source"] == P and rel["target"] in query_entity_ids) or \
                       (rel["target"] == P and rel["source"] in query_entity_ids):
                        is_connected = True
                        break

            if is_connected:
                tier1_chunks.append(chunk)
            elif reranker_score >= 0.60:
                tier2_chunks.append(chunk)
            else:
                tier3_chunks.append(chunk)

        return tier1_chunks, tier2_chunks, tier3_chunks

    def score(
        self,
        tier1: List[Dict[str, Any]],
        tier2: List[Dict[str, Any]],
        tier3: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Load weights and thresholds from configuration
        policy_config = self.retrieval_config.get("policies", {}).get("research", {})
        ranking_config = policy_config.get("ranking", {})
        semantic_weight = ranking_config.get("semantic", 0.70)
        graph_overlap_weight = ranking_config.get("graph_overlap", 0.30)

        thresholds = policy_config.get("thresholds", {})
        semantic_tier2_threshold = thresholds.get("semantic", 0.60)
        semantic_tier3_threshold = thresholds.get("semantic", 0.50)

        first_entity_name = query_entities[0].get("name") if query_entities else None

        # 1. Score Tier 1
        scored_tier1 = []
        for chunk in tier1:
            similarity_score = chunk.get("similarity_score", 0.0)
            reranker_score = chunk.get("reranker_score", similarity_score)
            graph_bonus = chunk.get("graph_bonus", 0.0)

            graph_overlap = 0.5
            final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(final_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "research",
                "tier": 1,
                "strategy": "graph_expansion",
                "ranking": {
                    "semantic_score": float(similarity_score),
                    "graph_overlap": float(graph_overlap),
                    "reranker_score": float(reranker_score),
                    "graph_bonus": float(graph_bonus),
                    "final_score": float(final_score),
                    "weight_breakdown": {
                        "semantic": float(semantic_weight),
                        "graph_overlap": float(graph_overlap_weight)
                    }
                },
                "source": {
                    "matched_entity": first_entity_name
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["graph_expansion"]
                }
            }
            scored_tier1.append(chunk_copy)

        # 2. Score Tier 2
        scored_tier2 = []
        for chunk in tier2:
            similarity_score = chunk.get("similarity_score", 0.0)
            reranker_score = chunk.get("reranker_score", similarity_score)
            graph_bonus = chunk.get("graph_bonus", 0.0)

            if reranker_score < semantic_tier2_threshold:
                continue

            graph_overlap = 0.0
            final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(final_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "research",
                "tier": 2,
                "strategy": "generic_semantic",
                "ranking": {
                    "semantic_score": float(similarity_score),
                    "graph_overlap": float(graph_overlap),
                    "reranker_score": float(reranker_score),
                    "graph_bonus": float(graph_bonus),
                    "final_score": float(final_score),
                    "weight_breakdown": {
                        "semantic": float(semantic_weight),
                        "graph_overlap": float(graph_overlap_weight)
                    }
                },
                "source": {
                    "matched_entity": None
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["generic_semantic"]
                }
            }
            scored_tier2.append(chunk_copy)

        # 3. Score Tier 3
        scored_tier3 = []
        for chunk in tier3:
            similarity_score = chunk.get("similarity_score", 0.0)
            reranker_score = chunk.get("reranker_score", similarity_score)
            graph_bonus = chunk.get("graph_bonus", 0.0)

            if reranker_score < semantic_tier3_threshold:
                continue

            graph_overlap = 0.0
            final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(final_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "research",
                "tier": 3,
                "strategy": "low_relevance_semantic",
                "ranking": {
                    "semantic_score": float(similarity_score),
                    "graph_overlap": float(graph_overlap),
                    "reranker_score": float(reranker_score),
                    "graph_bonus": float(graph_bonus),
                    "final_score": float(final_score),
                    "weight_breakdown": {
                        "semantic": float(semantic_weight),
                        "graph_overlap": float(graph_overlap_weight)
                    }
                },
                "source": {
                    "matched_entity": None
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["low_relevance_semantic"]
                }
            }
            scored_tier3.append(chunk_copy)

        return scored_tier1, scored_tier2, scored_tier3

    def rank(
        self,
        tier1: List[Dict[str, Any]],
        tier2: List[Dict[str, Any]],
        tier3: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tier1.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        tier2.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        tier3.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return tier1, tier2, tier3
