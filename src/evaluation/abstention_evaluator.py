from typing import Dict, Any, List
from src.evaluation.base_evaluator import BaseEvaluator

class AbstentionEvaluator(BaseEvaluator):
    """Evaluates sufficiencies, correctness of abstention behavior on OOD/traps."""

    def evaluate_run(self, should_abstain_list: List[bool], did_abstain_list: List[bool]) -> Dict[str, float]:
        """
        Computes aggregated accuracy, precision, recall, and F1 for the abstention class.
        """
        N = len(should_abstain_list)
        if N == 0:
            return {
                "abstention_accuracy": 1.0,
                "abstention_precision": 1.0,
                "abstention_recall": 1.0,
                "abstention_f1": 1.0
            }
            
        tp = 0  # Should abstain, did abstain
        fp = 0  # Should answer, did abstain
        fn = 0  # Should abstain, did answer
        tn = 0  # Should answer, did answer
        
        for sa, da in zip(should_abstain_list, did_abstain_list):
            if sa and da:
                tp += 1
            elif not sa and da:
                fp += 1
            elif sa and not da:
                fn += 1
            else:
                tn += 1
                
        accuracy = (tp + tn) / N
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
        
        return {
            "abstention_accuracy": float(accuracy),
            "abstention_precision": float(precision),
            "abstention_recall": float(recall),
            "abstention_f1": float(f1)
        }

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        should_abstain = query_case.get("allow_abstain", False) or query_case.get("category") in ["ood", "hallucination_trap"]
        did_abstain = generator_result.get("is_abstention", False) or "sufficient evidence" in generator_result.get("answer", "").lower()
        
        abstention_accuracy = 1.0 if should_abstain == did_abstain else 0.0
        
        # Determine classification mode:
        if should_abstain and did_abstain:
            mode = "correct_abstention"
        elif not should_abstain and not did_abstain:
            mode = "correct_answer"
        elif not should_abstain and did_abstain:
            mode = "false_abstention"
        else:
            mode = "missed_abstention"
            
        return {
            "abstention_mode": mode,
            "abstention_accuracy": float(abstention_accuracy)
        }
