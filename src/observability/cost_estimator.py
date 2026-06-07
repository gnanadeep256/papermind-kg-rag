from abc import ABC, abstractmethod

class BaseCostEstimator(ABC):
    @abstractmethod
    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Returns the estimated cost of the LLM call in USD."""
        pass

class GeminiCostEstimator(BaseCostEstimator):
    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Gemini 1.5/2.5 Pro pricing: $1.25 input / $5.00 output per million tokens
        # Gemini 1.5/2.5 Flash pricing: $0.075 input / $0.30 output per million tokens
        model_lower = model_id.lower()
        if "pro" in model_lower:
            return (prompt_tokens * 1.25 + completion_tokens * 5.0) / 1_000_000.0
        return (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000.0

class GroqCostEstimator(BaseCostEstimator):
    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Groq Llama-3-8b/Llama-3.1-8b: $0.05 input / $0.08 output per million tokens
        # Groq Llama-3-70b/Llama-3.3-70b: $0.59 input / $0.79 output per million tokens
        model_lower = model_id.lower()
        if "8b" in model_lower:
            return (prompt_tokens * 0.05 + completion_tokens * 0.08) / 1_000_000.0
        return (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000.0

class OpenAICostEstimator(BaseCostEstimator):
    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        # GPT-4o: $2.50 input / $10.00 output per million tokens
        # GPT-4o-mini: $0.150 input / $0.600 output per million tokens
        model_lower = model_id.lower()
        if "mini" in model_lower:
            return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0
        return (prompt_tokens * 2.50 + completion_tokens * 10.0) / 1_000_000.0

class AnthropicCostEstimator(BaseCostEstimator):
    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Claude 3.5 Sonnet: $3.00 input / $15.00 output per million tokens
        # Claude 3.5 Haiku: $0.80 input / $4.00 output per million tokens
        model_lower = model_id.lower()
        if "haiku" in model_lower:
            return (prompt_tokens * 0.80 + completion_tokens * 4.0) / 1_000_000.0
        return (prompt_tokens * 3.00 + completion_tokens * 15.0) / 1_000_000.0

class CostEstimatorFactory:
    @staticmethod
    def get_estimator(provider: str) -> BaseCostEstimator:
        p_lower = provider.lower()
        if p_lower == "gemini":
            return GeminiCostEstimator()
        elif p_lower == "groq":
            return GroqCostEstimator()
        elif p_lower == "openai":
            return OpenAICostEstimator()
        elif p_lower == "anthropic":
            return AnthropicCostEstimator()
        else:
            # Fallback default estimator (use Gemini Flash pricing)
            return GeminiCostEstimator()
