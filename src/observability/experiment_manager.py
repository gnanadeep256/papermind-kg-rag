import os
import sys
import platform
import shutil
import hashlib
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from src.observability.trace_models import ExperimentMetadata

def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"

def compute_file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "none"
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "error"

def compute_string_hash(text: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()

class ExperimentManager:
    def __init__(self, base_dir: str = "reports/experiments", config_path: str = "configs/config.yaml"):
        self.base_dir = base_dir
        self.config_path = config_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"experiment_{self.timestamp}"
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_id)
        self.metadata: Optional[ExperimentMetadata] = None

    def setup_experiment(self, config_dict: Dict[str, Any]) -> str:
        """Sets up the experiment directory layout and writes metadata/config snapshots."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 1. Snapshot the configuration file
        snapshot_path = os.path.join(self.experiment_dir, "config_snapshot.yaml")
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, snapshot_path)
            
        # 2. Gather metadata
        git_sha = get_git_sha()
        py_version = sys.version.replace("\n", " ")
        plat_os = f"{platform.system()} {platform.release()} {platform.machine()}"
        
        # Extract metadata from config dict
        eval_cfg = config_dict.get("evaluation", {})
        emb_cfg = config_dict.get("embeddings", {})
        llm_cfg = config_dict.get("llm", {})
        
        embedding_model = emb_cfg.get("model_name", "unknown")
        judge_model = eval_cfg.get("judge_model_provenance", "unknown")
        fallback_model = eval_cfg.get("fallback_judge_provenance", "unknown")
        benchmark_version = str(eval_cfg.get("version", "1.0.0"))
        
        config_hash = compute_file_hash(self.config_path)
        
        # Dependency lock file hash (e.g., uv.lock or pyproject.toml)
        dep_lock_hash = "none"
        for lockfile in ["uv.lock", "poetry.lock", "pyproject.toml"]:
            if os.path.exists(lockfile):
                dep_lock_hash = compute_file_hash(lockfile)
                break
                
        self.metadata = ExperimentMetadata(
            git_sha=git_sha,
            python_version=py_version,
            platform_os=plat_os,
            embedding_model=embedding_model,
            judge_model=judge_model,
            fallback_model=fallback_model,
            config_hash=config_hash,
            dependency_lock_hash=dep_lock_hash,
            benchmark_version=benchmark_version
        )
        
        # 3. Write metadata file
        meta_filepath = os.path.join(self.experiment_dir, "environment_metadata.json")
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(self.metadata.model_dump(), f, indent=2)
            
        return self.experiment_dir

    def get_metadata(self) -> Optional[ExperimentMetadata]:
        return self.metadata
