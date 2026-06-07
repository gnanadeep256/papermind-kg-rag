import os
import gzip
from src.observability.base_tracer import BaseTracer
from src.observability.trace_models import TraceContext

class LocalTracer(BaseTracer):
    def __init__(self, experiment_dir: str, compress: bool = True):
        self.experiment_dir = experiment_dir
        self.compress = compress
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        if self.compress:
            self.trace_file = os.path.join(self.experiment_dir, "traces.jsonl.gz")
        else:
            self.trace_file = os.path.join(self.experiment_dir, "traces.jsonl")
            
    def log_trace(self, context: TraceContext) -> None:
        # Pydantic v2 model_dump_json() serializes recursively automatically
        trace_json = context.model_dump_json()
        
        if self.compress:
            # Gzip file-like objects support text mode ("at") in Python
            with gzip.open(self.trace_file, "at", encoding="utf-8") as f:
                f.write(trace_json + "\n")
        else:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(trace_json + "\n")
                
    def close(self) -> None:
        # Flush or clean up if needed
        pass
