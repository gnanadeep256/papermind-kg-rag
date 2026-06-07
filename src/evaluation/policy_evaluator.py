from typing import Dict, Any, List
from src.evaluation.base_evaluator import BaseEvaluator
from src.evidence.policy_factory import POLICY_CATEGORY_MAP

class PolicyEvaluator(BaseEvaluator):
    """Evaluates the strategy pattern routing, fallback behaviors, and budgeting stats."""
    
    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        retrieval_result = generator_result.get("retrieval_result")
        metadata = generator_result.get("retrieval_metadata", {})
        
        if retrieval_result:
            metadata = retrieval_result.retrieval_metadata
            vector_context = retrieval_result.vector_context
        else:
            vector_context = generator_result.get("vector_context", [])
            
        policy_stats = metadata.get("policy", {})
        
        policy_selected = policy_stats.get("name", metadata.get("intent", "research"))
        gold_category = query_case.get("category", "research")
        
        # Policy correctness (some categories map to research)
        mapped_intent = POLICY_CATEGORY_MAP.get(gold_category, gold_category)
        policy_correct = (policy_selected == mapped_intent)

        
        policy_fallback = policy_stats.get("fallback_used", False)
        
        # Tier ratios
        tier_distribution = policy_stats.get("tier_distribution", {})
        t1 = tier_distribution.get("tier1", 0)
        t2 = tier_distribution.get("tier2", 0)
        t3 = tier_distribution.get("tier3", 0)
        total_tiers = t1 + t2 + t3
        
        tier1_ratio = t1 / total_tiers if total_tiers > 0 else 0.0
        tier2_ratio = t2 / total_tiers if total_tiers > 0 else 0.0
        tier3_ratio = t3 / total_tiers if total_tiers > 0 else 0.0
        
        # Budget utilization
        evidence_stats = policy_stats.get("evidence", {})
        budget_utilization = evidence_stats.get("utilization", 0.0)
        
        # Graph overlap ratio in selected chunks
        overlaps = []
        for c in vector_context:
            reason = c.retrieval_reason if hasattr(c, "retrieval_reason") else c.get("retrieval_reason")
            if reason:
                ranking = reason.ranking if hasattr(reason, "ranking") else reason.get("ranking")
                if ranking:
                    overlap = ranking.graph_overlap if hasattr(ranking, "graph_overlap") else ranking.get("graph_overlap")
                    if overlap is not None:
                        overlaps.append(overlap)
                        
        graph_overlap_ratio = sum(overlaps) / len(overlaps) if overlaps else 0.0
        
        return {
            "policy_selected": policy_selected,
            "policy_correct": float(policy_correct),
            "policy_fallback": float(policy_fallback),
            "tier1_ratio": float(tier1_ratio),
            "tier2_ratio": float(tier2_ratio),
            "tier3_ratio": float(tier3_ratio),
            "budget_utilization": float(budget_utilization),
            "graph_overlap_ratio": float(graph_overlap_ratio)
        }
