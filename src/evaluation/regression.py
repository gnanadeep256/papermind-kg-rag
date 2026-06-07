import os
import json
import re
from typing import Dict, Any, List, Optional
from src.evaluation.metrics import RegressionComparison, AggregatedEvaluationResult

class RegressionEngine:
    """Compares current evaluation metrics against historical runs to identify performance changes."""
    
    def __init__(self, reports_dir: str = "reports/evaluation") -> None:
        self.reports_dir = reports_dir

    def get_latest_previous_run(self, current_timestamp: str) -> Optional[AggregatedEvaluationResult]:
        """
        Finds the most recent folder before current_timestamp and loads its benchmark.json.
        """
        if not os.path.exists(self.reports_dir):
            return None
            
        # List all run directories
        folders = []
        for name in os.listdir(self.reports_dir):
            path = os.path.join(self.reports_dir, name)
            if os.path.isdir(path) and name.startswith("run_"):
                folders.append(name)
                
        folders.sort()  # Sorts chronologically if formatted like run_YYYYMMDD_HHMMSS
        
        # Find latest folder before current_timestamp
        previous_folder = None
        for f in folders:
            if f < f"run_{current_timestamp}":
                previous_folder = f
            else:
                break
                
        if not previous_folder:
            # Fallback to the last one if current_timestamp isn't matching structure
            if len(folders) > 1 and folders[-1] != f"run_{current_timestamp}":
                previous_folder = folders[-2] if folders[-1] == f"run_{current_timestamp}" else folders[-1]
            else:
                return None
                
        json_path = os.path.join(self.reports_dir, previous_folder, "benchmark.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(self.reports_dir, previous_folder, "benchmark_dryrun.json")
        if not os.path.exists(json_path):
            return None
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return AggregatedEvaluationResult.model_validate(data)
        except Exception:
            return None

    def compare_runs(
        self,
        current: AggregatedEvaluationResult,
        previous: Optional[AggregatedEvaluationResult]
    ) -> List[RegressionComparison]:
        """
        Compares two runs and returns delta metrics.
        """
        if not previous:
            return []
            
        comparisons = []
        
        # Metrics mapping: (metric_name, current_getter, previous_getter, higher_is_better, is_percentage)
        metrics_to_compare = [
            (
                "Overall Score", 
                lambda r: r.overall_score, 
                lambda r: r.overall_score, 
                True, 
                False
            ),
            (
                "Micro Score",
                lambda r: r.micro_overall_score / 100.0 if hasattr(r, "micro_overall_score") else None,
                lambda r: r.micro_overall_score / 100.0 if hasattr(r, "micro_overall_score") else None,
                True,
                True
            ),
            (
                "Macro Score",
                lambda r: r.macro_overall_score if hasattr(r, "macro_overall_score") else None,
                lambda r: r.macro_overall_score if hasattr(r, "macro_overall_score") else None,
                True,
                False
            ),
            (
                "Weighted Macro Score",
                lambda r: r.weighted_macro_overall_score if hasattr(r, "weighted_macro_overall_score") else None,
                lambda r: r.weighted_macro_overall_score if hasattr(r, "weighted_macro_overall_score") else None,
                True,
                False
            ),
            (
                "Retrieval Recall", 
                lambda r: r.avg_retrieval_recall, 
                lambda r: r.avg_retrieval_recall, 
                True, 
                True
            ),
            (
                "Citation Precision", 
                lambda r: r.avg_citation_precision, 
                lambda r: r.avg_citation_precision, 
                True, 
                True
            ),
            (
                "Hybrid Faithfulness", 
                lambda r: r.avg_hybrid_faithfulness, 
                lambda r: r.avg_hybrid_faithfulness, 
                True, 
                True
            ),
            (
                "Groundedness", 
                lambda r: r.avg_groundedness, 
                lambda r: r.avg_groundedness, 
                True, 
                True
            ),
            (
                "Abstention Accuracy", 
                lambda r: r.avg_abstention_accuracy, 
                lambda r: r.avg_abstention_accuracy, 
                True, 
                True
            ),
            (
                "Robustness Score", 
                lambda r: r.avg_robustness, 
                lambda r: r.avg_robustness, 
                True, 
                True
            ),
            (
                "Latency P95", 
                lambda r: r.latency_percentiles.p95 / 1000.0,  # convert to seconds
                lambda r: r.latency_percentiles.p95 / 1000.0, 
                False, 
                False
            ),
            (
                "Pairwise Distance",
                lambda r: r.avg_pairwise_distance if hasattr(r, "avg_pairwise_distance") else None,
                lambda r: r.avg_pairwise_distance if hasattr(r, "avg_pairwise_distance") else None,
                True,
                False
            ),
            (
                "Context Redundancy",
                lambda r: r.avg_context_redundancy_score if hasattr(r, "avg_context_redundancy_score") else None,
                lambda r: r.avg_context_redundancy_score if hasattr(r, "avg_context_redundancy_score") else None,
                False,
                False
            ),
            (
                "Expected Calibration Error",
                lambda r: r.expected_calibration_error if hasattr(r, "expected_calibration_error") else None,
                lambda r: r.expected_calibration_error if hasattr(r, "expected_calibration_error") else None,
                False,
                False
            ),
            (
                "Brier Score",
                lambda r: r.brier_score if hasattr(r, "brier_score") else None,
                lambda r: r.brier_score if hasattr(r, "brier_score") else None,
                False,
                False
            ),
            (
                "Policy Routing Accuracy",
                lambda r: r.policy_routing_accuracy if hasattr(r, "policy_routing_accuracy") else None,
                lambda r: r.policy_routing_accuracy if hasattr(r, "policy_routing_accuracy") else None,
                True,
                True
            )
        ]
        
        for name, curr_get, prev_get, higher_is_better, is_pct in metrics_to_compare:
            try:
                curr_val = curr_get(current)
                prev_val = prev_get(previous)
                if curr_val is None or prev_val is None:
                    continue
                diff = curr_val - prev_val
                if abs(diff) < 1e-6:
                    diff = 0.0
                
                # Format sign and label
                sign = "+" if diff >= 0 else ""
                unit = "%" if is_pct else ("s" if "Latency" in name else "")
                
                # Multiply by 100 for display if percentage
                disp_curr = curr_val * 100.0 if is_pct else curr_val
                disp_prev = prev_val * 100.0 if is_pct else prev_val
                disp_diff = diff * 100.0 if is_pct else diff
                
                fmt = ".3f" if any(x in name for x in ["Error", "Brier", "Distance", "Redundancy"]) else ".1f"
                
                if diff == 0:
                    label = f"{name}: {disp_curr:{fmt}}{unit} (no change)"
                else:
                    arrow = "↑" if diff > 0 else "↓"
                    # Determine if change is positive or negative
                    improvement = (diff > 0) == higher_is_better
                    change_type = "improvement" if improvement else "regression"
                    label = f"{name}: {disp_curr:{fmt}}{unit} ({arrow} {sign}{disp_diff:{fmt}}{unit})"
                    
                comparisons.append(RegressionComparison(
                    metric_name=name,
                    current_value=float(curr_val),
                    previous_value=float(prev_val),
                    differential=float(diff),
                    label=label
                ))
            except Exception:
                pass
                
        return comparisons
