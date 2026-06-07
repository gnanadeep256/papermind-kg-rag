from typing import Dict, Any, List
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class CalibrationEvaluator(BaseEvaluator):
    """Evaluates the calibration of confidence estimates (ECE and Brier score)."""
    
    def __init__(self, num_bins: int = 5) -> None:
        self.num_bins = num_bins

    def evaluate_run(self, confidences: List[float], correctness_scores: List[float]) -> Dict[str, float]:
        """
        Computes calibration diagnostics on a full run.
        
        Args:
            confidences: List of floats between 0.0 and 1.0 representing predicted confidence.
            correctness_scores: List of floats (0.0 or 1.0) representing actual correctness.
            
        Returns:
            Dict containing ECE and Brier Score.
        """
        N = len(confidences)
        if N == 0:
            return {"expected_calibration_error": 0.0, "brier_score": 0.0}
            
        conf_arr = np.array(confidences)
        corr_arr = np.array(correctness_scores)
        
        # 1. Brier Score (Mean Squared Error)
        brier_score = float(np.mean((conf_arr - corr_arr) ** 2))
        
        # 2. Expected Calibration Error (ECE)
        ece = 0.0
        # Define bin boundaries
        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        
        for m in range(self.num_bins):
            bin_lower = bin_boundaries[m]
            bin_upper = bin_boundaries[m + 1]
            
            # Select samples in current bin
            if m == self.num_bins - 1:
                in_bin = (conf_arr >= bin_lower) & (conf_arr <= bin_upper)
            else:
                in_bin = (conf_arr >= bin_lower) & (conf_arr < bin_upper)
                
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = np.mean(corr_arr[in_bin])
                bin_conf = np.mean(conf_arr[in_bin])
                ece += (bin_size / N) * np.abs(bin_acc - bin_conf)
                
        return {
            "expected_calibration_error": float(ece),
            "brier_score": float(brier_score)
        }

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        # Satisfy BaseEvaluator interface (returns defaults if evaluated singly)
        return {
            "expected_calibration_error": 0.0,
            "brier_score": 0.0
        }
