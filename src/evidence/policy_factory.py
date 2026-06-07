from typing import Dict, Any, Type
from src.evidence.base_policy import BaseEvidencePolicy
from src.evidence.dataset_policy import DatasetEvidencePolicy
from src.evidence.method_policy import MethodEvidencePolicy
from src.evidence.paper_policy import PaperEvidencePolicy
from src.evidence.research_policy import ResearchEvidencePolicy

POLICY_REGISTRY: Dict[str, Type[BaseEvidencePolicy]] = {
    "dataset": DatasetEvidencePolicy,
    "method": MethodEvidencePolicy,
    "paper": PaperEvidencePolicy,
    "research": ResearchEvidencePolicy
}

class PolicyFactory:
    _registry: Dict[str, Type[BaseEvidencePolicy]] = dict(POLICY_REGISTRY)

    @classmethod
    def register(cls, intent: str, policy_class: Type[BaseEvidencePolicy]) -> None:
        """Allows runtime registration of new intent policies."""
        cls._registry[intent] = policy_class

    @classmethod
    def create(cls, intent: str, retrieval_config: Dict[str, Any], entity_cache: Dict[str, Dict[str, Any]]) -> BaseEvidencePolicy:
        """Factory method to instantiate a policy class from registry."""
        policy_class = cls._registry.get(intent, ResearchEvidencePolicy)
        return policy_class(retrieval_config, entity_cache)
