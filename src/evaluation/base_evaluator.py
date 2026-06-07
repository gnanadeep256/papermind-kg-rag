from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Abstract Base Class for all benchmark metrics evaluators."""
    
    @abstractmethod
    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single query case against the answer generation outcome.
        
        Args:
            query_case: Dict containing gold benchmark case information (query, category, expected fields).
            generator_result: Dict containing generator pipeline outputs (answer, context, metadata).
            
        Returns:
            Dict containing evaluated metric names and float scores.
        """
        pass
