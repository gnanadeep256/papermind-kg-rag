from typing import Dict, Any, List, Tuple
from loguru import logger
from src.evidence.base_policy import BaseEvidencePolicy

class MethodEvidencePolicy(BaseEvidencePolicy):
    """Evidence Policy for method/algorithm intent queries."""

    def classify(
        self,
        unpacked_chunks: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
        query_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tier1_chunks = []
        tier2_chunks = []
        tier3_chunks = []

        matched_method_ids = {ent["entity_id"] for ent in query_entities if ent.get("type") == "Method" or ent.get("entity_type") == "Method"}

        for chunk in unpacked_chunks:
            P = chunk["arxiv_id"]
            chunk_text = chunk.get("text", "")

            is_intro_or_solve = False
            if matched_method_ids:
                for rel in graph_context.get("relationships", []):
                    if rel["source"] == P and rel["target"] in matched_method_ids and rel["relation"] in ["INTRODUCES", "SOLVES"]:
                        is_intro_or_solve = True
                        break

            if is_intro_or_solve:
                tier1_chunks.append(chunk)
            else:
                has_mention = False
                if matched_method_ids:
                    matched_method_names = {
                        n.get("name") or n.get("title") 
                        for n in graph_context.get("nodes", []) 
                        if n.get("entity_id") in matched_method_ids
                    }
                    has_mention = any(mname and mname.lower() in chunk_text.lower() for mname in matched_method_names)

                if has_mention:
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
        policy_config = self.retrieval_config.get("policies", {}).get("method", {})
        ranking_config = policy_config.get("ranking", {})
        semantic_weight = ranking_config.get("semantic", 0.70)
        graph_overlap_weight = ranking_config.get("graph_overlap", 0.30)

        thresholds = policy_config.get("thresholds", {})
        semantic_tier2_threshold = thresholds.get("semantic", 0.70)
        semantic_tier3_threshold = thresholds.get("semantic", 0.70)

        matched_method_ids = {ent["entity_id"] for ent in query_entities if ent.get("type") == "Method" or ent.get("entity_type") == "Method"}
        matched_method_names = {
            n.get("entity_id"): (n.get("name") or n.get("title")) 
            for n in graph_context.get("nodes", []) 
            if n.get("entity_id") in matched_method_ids
        }
        first_method_name = next(iter(matched_method_names.values()), None) if matched_method_names else None

        # 1. Score Tier 1
        scored_tier1 = []
        for chunk in tier1:
            similarity_score = chunk.get("similarity_score", 0.0)
            reranker_score = chunk.get("reranker_score", similarity_score)
            graph_bonus = chunk.get("graph_bonus", 0.0)
            
            graph_overlap = 1.0
            final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(final_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "method",
                "tier": 1,
                "strategy": "introduced_or_solved_method_graph",
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
                    "matched_entity": first_method_name,
                    "matched_method": first_method_name
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["introduced_or_solved_method_graph"]
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

            graph_overlap = 0.5
            final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(final_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "method",
                "tier": 2,
                "strategy": "semantic_method_mention",
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
                    "matched_entity": first_method_name,
                    "matched_method": first_method_name
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["semantic_method_mention"]
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
                "policy": "method",
                "tier": 3,
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
                    "matched_entity": None,
                    "matched_method": None
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["generic_semantic"]
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
