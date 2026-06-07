import re
from typing import Dict, Any, List

class QuestionClassifier:
    """
    Classifies natural language research queries into specific categories
    to dynamically adapt output formatting styles.
    """
    def __init__(self) -> None:
        # Rules defined in order of precedence
        self.rules = [
            (r"\b(compare|comparison|versus|vs|difference|contrast|trade-off|similarities and differences)\b", "comparison"),
            (r"\b(summarize|summary|abstract|overview|synopsis|brief explanation of)\b", "summary"),
            (r"\b(advantages?|benefits?|strengths?|improvements?|gains?|pro|pros|positives?|upsides?)\b", "advantages"),
            (r"\b(limitations?|weaknesses?|bottlenecks?|drawbacks?|con|cons|negatives?|downsides?|failure modes?)\b", "limitations"),
            (r"\b(future work|future directions?|next steps?|proposed extensions?|planned improvements?)\b", "future_work"),
            (r"\b(evaluations?|protocols?|baselines?|comparison models?|metrics?|precision|recall|f1)\b", "evaluation"),
            (r"\b(experimental results?|ablation|performance|accuracy|f1-score|recall score|precision score|numbers|charts|results table)\b", "experimental_results"),
            (r"\b(workflow|steps?|process|pipeline|algorithm|procedure|stage|phase|step-by-step)\b", "workflow"),
            (r"\b(survey|literature review|thematic|overview of papers|state of the art|sota)\b", "survey"),
            (r"\b(what is|define|definition|meaning of)\b", "definition"),
            (r"\b(why|reason|rationale|cause|motivation|purpose)\b", "why"),
            (r"\b(how do|how does|how is|how can|mechanism|manner|mode)\b", "how"),
            (r"\b(datasets?|benchmarks?|corpora|corpus|data split|eval split)\b", "dataset"),
            (r"\b(papers?|arxiv|authors?|published|written by|title)\b", "paper"),
            (r"\b(methods?|frameworks?|approaches?|architectures?|techniques?|systems?)\b", "method"),
            (r"\b(implementation|code|repo|github|libraries|install|compile|run|execution)\b", "implementation"),
        ]

    def classify(self, query: str) -> str:
        """Classifies query into one of the 16 categories using rule precedence."""
        query_lower = query.lower()
        
        for pattern, category in self.rules:
            if re.search(pattern, query_lower):
                return category
                
        return "default"

    def get_template_path(self, category: str) -> str:
        """Maps category to corresponding prompt XML path."""
        # Normalize category string
        cat = category.replace(" ", "_")
        return f"src/prompts/{cat}.xml"
