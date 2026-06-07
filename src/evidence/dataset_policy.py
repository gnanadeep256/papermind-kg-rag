from typing import Dict, Any, List, Tuple
from loguru import logger
from src.evidence.base_policy import BaseEvidencePolicy

class DatasetEvidencePolicy(BaseEvidencePolicy):
    """Evidence Policy for dataset/evaluation intent queries."""
    
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
        all_datasets = {ent["name"] for ent in self.entity_cache.values() if ent.get("type") == "Dataset"}

        if matched_method_ids:
            # Find nodes connected to matched methods in graph context
            method_graph_ids = set(matched_method_ids)
            for rel in graph_context.get("relationships", []):
                src = rel["source"]
                tgt = rel["target"]
                if src in matched_method_ids:
                    method_graph_ids.add(tgt)
                if tgt in matched_method_ids:
                    method_graph_ids.add(src)

            # Datasets connected to matched methods in graph context
            method_datasets = {
                n.get("name") or n.get("title") 
                for n in graph_context.get("nodes", []) 
                if (n.get("entity_type") == "Dataset" or n.get("type") == "Dataset") 
                and n.get("entity_id") in method_graph_ids
            }
            all_datasets_fallback = all_datasets - method_datasets

            for chunk in unpacked_chunks:
                P = chunk["arxiv_id"]
                
                # Compute chunk graph neighbors
                chunk_graph_ids = {P}
                for rel in graph_context.get("relationships", []):
                    src = rel["source"]
                    tgt = rel["target"]
                    if src == P:
                        chunk_graph_ids.add(tgt)
                    if tgt == P:
                        chunk_graph_ids.add(src)
                        
                intersection = chunk_graph_ids.intersection(method_graph_ids)

                if len(intersection) > 0:
                    tier1_chunks.append(chunk)
                else:
                    chunk_text = chunk.get("text", "")
                    fallback_overlap = sum(1 for dname in all_datasets_fallback if dname and dname.lower() in chunk_text.lower())
                    if fallback_overlap > 0:
                        tier2_chunks.append(chunk)
                    else:
                        tier3_chunks.append(chunk)
        else:
            # Fallback dataset logic when no methods are matched in query
            graph_datasets = {
                n.get("name") or n.get("title") 
                for n in graph_context.get("nodes", []) 
                if (n.get("entity_type") == "Dataset" or n.get("type") == "Dataset")
            }
            all_datasets_fallback = all_datasets - graph_datasets

            for chunk in unpacked_chunks:
                chunk_text = chunk.get("text", "")
                overlap_count = sum(1 for dname in graph_datasets if dname and dname.lower() in chunk_text.lower())
                
                if overlap_count > 0:
                    tier1_chunks.append(chunk)
                else:
                    fallback_overlap = sum(1 for dname in all_datasets_fallback if dname and dname.lower() in chunk_text.lower())
                    if fallback_overlap > 0:
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
        policy_config = self.retrieval_config.get("policies", {}).get("dataset", {})
        ranking_config = policy_config.get("ranking", {})
        semantic_weight = ranking_config.get("semantic", 0.70)
        graph_overlap_weight = ranking_config.get("graph_overlap", 0.30)

        thresholds = policy_config.get("thresholds", {})
        semantic_tier2_threshold = thresholds.get("semantic", 0.68)
        semantic_tier3_threshold = thresholds.get("semantic", 0.72)

        matched_method_ids = {ent["entity_id"] for ent in query_entities if ent.get("type") == "Method" or ent.get("entity_type") == "Method"}
        matched_method_names = {
            n.get("entity_id"): (n.get("name") or n.get("title")) 
            for n in graph_context.get("nodes", []) 
            if n.get("entity_id") in matched_method_ids
        }
        
        # 1. Score Tier 1
        scored_tier1 = []
        if matched_method_ids:
            method_graph_ids = set(matched_method_ids)
            for rel in graph_context.get("relationships", []):
                src = rel["source"]
                tgt = rel["target"]
                if src in matched_method_ids:
                    method_graph_ids.add(tgt)
                if tgt in matched_method_ids:
                    method_graph_ids.add(src)

            for chunk in tier1:
                P = chunk["arxiv_id"]
                similarity_score = chunk.get("similarity_score", 0.0)
                reranker_score = chunk.get("reranker_score", similarity_score)
                graph_bonus = chunk.get("graph_bonus", 0.0)

                chunk_graph_ids = {P}
                for rel in graph_context.get("relationships", []):
                    src = rel["source"]
                    tgt = rel["target"]
                    if src == P:
                        chunk_graph_ids.add(tgt)
                    if tgt == P:
                        chunk_graph_ids.add(src)

                intersection = chunk_graph_ids.intersection(method_graph_ids)
                graph_overlap = len(intersection) / len(chunk_graph_ids) if len(chunk_graph_ids) > 0 else 0.0
                final_score = semantic_weight * reranker_score + graph_overlap_weight * graph_overlap

                # Find source matched entities
                matched_ent_name = next(iter(matched_method_names.values()), None) if matched_method_names else None
                matched_ds_name = None
                for node in graph_context.get("nodes", []):
                    if (node.get("entity_type") == "Dataset" or node.get("type") == "Dataset") and node.get("entity_id") in intersection:
                        matched_ds_name = node.get("name") or node.get("title")
                        break

                chunk_copy = dict(chunk)
                chunk_copy["combined_score"] = float(final_score)
                chunk_copy["retrieval_reason"] = {
                    "policy": "dataset",
                    "tier": 1,
                    "strategy": "graph_grounded_dataset",
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
                        "matched_entity": matched_ent_name,
                        "matched_dataset": matched_ds_name
                    },
                    "merge": {
                        "merged_chunks": 1,
                        "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                        "merged_chunk_ids": [chunk["chunk_id"]],
                        "provenance_sources": ["graph_grounded_dataset"]
                    }
                }
                scored_tier1.append(chunk_copy)
        else:
            # Fallback scoring when no matched methods exist in query
            for chunk in tier1:
                similarity_score = chunk.get("similarity_score", 0.0)
                reranker_score = chunk.get("reranker_score", similarity_score)
                graph_bonus = chunk.get("graph_bonus", 0.0)
                final_score = semantic_weight * reranker_score + graph_overlap_weight * 1.0

                chunk_copy = dict(chunk)
                chunk_copy["combined_score"] = float(final_score)
                chunk_copy["retrieval_reason"] = {
                    "policy": "dataset",
                    "tier": 1,
                    "strategy": "graph_grounded_dataset",
                    "ranking": {
                        "semantic_score": float(similarity_score),
                        "graph_overlap": 1.0,
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
                        "matched_dataset": None
                    },
                    "merge": {
                        "merged_chunks": 1,
                        "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                        "merged_chunk_ids": [chunk["chunk_id"]],
                        "provenance_sources": ["graph_grounded_dataset"]
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

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(reranker_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "dataset",
                "tier": 2,
                "strategy": "semantic_dataset_fallback",
                "ranking": {
                    "semantic_score": float(similarity_score),
                    "graph_overlap": 0.0,
                    "reranker_score": float(reranker_score),
                    "graph_bonus": float(graph_bonus),
                    "final_score": float(reranker_score),
                    "weight_breakdown": {
                        "semantic": 1.0,
                        "graph_overlap": 0.0
                    }
                },
                "source": {
                    "matched_entity": None,
                    "matched_dataset": None
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["semantic_dataset_fallback"]
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

            chunk_copy = dict(chunk)
            chunk_copy["combined_score"] = float(reranker_score)
            chunk_copy["retrieval_reason"] = {
                "policy": "dataset",
                "tier": 3,
                "strategy": "generic_semantic_fallback",
                "ranking": {
                    "semantic_score": float(similarity_score),
                    "graph_overlap": 0.0,
                    "reranker_score": float(reranker_score),
                    "graph_bonus": float(graph_bonus),
                    "final_score": float(reranker_score),
                    "weight_breakdown": {
                        "semantic": 1.0,
                        "graph_overlap": 0.0
                    }
                },
                "source": {
                    "matched_entity": None,
                    "matched_dataset": None
                },
                "merge": {
                    "merged_chunks": 1,
                    "merged_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                    "merged_chunk_ids": [chunk["chunk_id"]],
                    "provenance_sources": ["generic_semantic_fallback"]
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
