from abc import ABC, abstractmethod
from src.observability.trace_models import TraceContext

class BaseTracer(ABC):
    @abstractmethod
    def log_trace(self, context: TraceContext) -> None:
        """Publishes the collected TraceContext telemetry."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Closes and flushes tracer resources."""
        pass
