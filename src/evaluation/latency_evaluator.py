from typing import Dict, Any, List
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class LatencyEvaluator(BaseEvaluator):
    """Parses sub-stage timings and computes latency statistics (percentiles, averages)."""

    def evaluate_run(self, total_latencies_ms: List[float]) -> Dict[str, float]:
        """
        Computes run percentiles for total latency.
        """
        if not total_latencies_ms:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
            
        arr = np.array(total_latencies_ms)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99))
        }

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        provenance = generator_result.get("provenance", {})
        metadata = generator_result.get("metadata", {})
        
        # Access properties defensively supporting both objects and dicts
        retrieval_time = provenance.get("retrieval_time_ms", 0.0)
        generation_time = provenance.get("generation_time_ms", 0.0)
        
        total_latency = metadata.get("total_execution_time_ms")
        if total_latency is None:
            total_latency = provenance.get("total_execution_time_ms", retrieval_time + generation_time)
            
        # Unpack retrieval sub-stages if available
        retrieval_result = generator_result.get("retrieval_result")
        ret_metadata = {}
        if retrieval_result:
            ret_metadata = getattr(retrieval_result, "retrieval_metadata", {})
        else:
            # Fallback check inside metadata
            ret_metadata = metadata.get("retrieval_metrics", {})
            if not isinstance(ret_metadata, dict):
                ret_metadata = {}
                
        vector_time = ret_metadata.get("vector_time_ms", 0.0)
        graph_time = ret_metadata.get("graph_time_ms", 0.0)
        rerank_time = ret_metadata.get("rerank_time_ms", 0.0)
        fusion_time = ret_metadata.get("fusion_time_ms", 0.0)
        
        latency_breakdown = {
            "vector_retrieval": float(vector_time),
            "graph_retrieval": float(graph_time),
            "policy_execution": float(max(0.0, fusion_time - (vector_time + graph_time))),
            "generation": float(generation_time),
            "total": float(total_latency)
        }
        
        return {
            "latency_ms": float(total_latency),
            "latency_breakdown": latency_breakdown
        }
