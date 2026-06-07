import os
import json
import sqlite3
import hashlib
import numpy as np
from typing import Any, Dict, List, Optional

class SqliteEmbeddingCache:
    """SQLite-backed cache for text embeddings to prevent redundant LLM/SentenceTransformer API encoding calls."""
    
    def __init__(self, db_path: str = "data/evaluation/embedding_cache.sqlite",
                 invalidation_strategy: Optional[str] = None,
                 max_models: Optional[int] = None,
                 max_age_days: Optional[int] = None) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        
        # Load from config if not explicitly provided
        if invalidation_strategy is None or max_models is None or max_age_days is None:
            try:
                from src.utils.config import load_config
                config = load_config()
                eval_cache = config.get("evaluation", {}).get("cache", {})
                if invalidation_strategy is None:
                    invalidation_strategy = eval_cache.get("invalidation_strategy", "aggressive")
                if max_models is None:
                    max_models = eval_cache.get("max_models", 3)
                if max_age_days is None:
                    max_age_days = eval_cache.get("max_age_days", 30)
            except Exception:
                if invalidation_strategy is None:
                    invalidation_strategy = "aggressive"
                if max_models is None:
                    max_models = 3
                if max_age_days is None:
                    max_age_days = 30
                    
        self.invalidation_strategy = invalidation_strategy
        self.max_models = max_models
        self.max_age_days = max_age_days
        
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                text TEXT,
                model_name TEXT,
                normalized INTEGER,
                embedding TEXT
            )
            """)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_usage (
                model_name TEXT PRIMARY KEY,
                last_used INTEGER
            )
            """)

    def get(self, text: str, model_name: str, normalized: bool) -> Optional[List[float]]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ? AND model_name = ? AND normalized = ?",
            (text_hash, model_name, int(normalized))
        )
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def set(self, text: str, model_name: str, normalized: bool, embedding: List[float]) -> None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings (text_hash, text, model_name, normalized, embedding) VALUES (?, ?, ?, ?, ?)",
                    (text_hash, text, model_name, int(normalized), json.dumps(embedding))
                )
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def update_model_usage(self, model_name: str) -> None:
        """Updates the last used timestamp for a model with millisecond precision."""
        import time
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES (?, ?)",
                    (model_name, int(time.time() * 1000))
                )
        except Exception:
            pass

    def invalidate_other_models(self, current_model_name: str) -> None:
        """Applies the configured invalidation strategy to clean up cached entries."""
        import time
        now = int(time.time() * 1000)
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO model_usage (model_name, last_used) VALUES (?, ?)",
                    (current_model_name, now)
                )
                
                # Proactively ensure all model names in embeddings have a usage entry
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT model_name FROM embeddings WHERE model_name NOT IN (SELECT model_name FROM model_usage)"
                )
                missing_models = [r[0] for r in cursor.fetchall()]
                for m in missing_models:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO model_usage (model_name, last_used) VALUES (?, ?)",
                        (m, now)
                    )
        except Exception:
            pass

        if self.invalidation_strategy == "aggressive":
            try:
                with self.conn:
                    self.conn.execute("DELETE FROM embeddings WHERE model_name != ?", (current_model_name,))
                    self.conn.execute("DELETE FROM model_usage WHERE model_name != ?", (current_model_name,))
            except Exception:
                pass
        elif self.invalidation_strategy == "lru":
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT model_name FROM model_usage ORDER BY last_used DESC")
                rows = cursor.fetchall()
                all_models = [r[0] for r in rows]
                if len(all_models) > self.max_models:
                    models_to_delete = all_models[self.max_models:]
                    with self.conn:
                        for model in models_to_delete:
                            self.conn.execute("DELETE FROM embeddings WHERE model_name = ?", (model,))
                            self.conn.execute("DELETE FROM model_usage WHERE model_name = ?", (model,))
            except Exception:
                pass
        elif self.invalidation_strategy == "ttl":
            try:
                cutoff = now - (self.max_age_days * 86400 * 1000)
                cursor = self.conn.cursor()
                cursor.execute("SELECT model_name FROM model_usage WHERE last_used < ?", (cutoff,))
                rows = cursor.fetchall()
                models_to_delete = [r[0] for r in rows]
                with self.conn:
                    for model in models_to_delete:
                        # Never delete the active model
                        if model == current_model_name:
                            continue
                        self.conn.execute("DELETE FROM embeddings WHERE model_name = ?", (model,))
                        self.conn.execute("DELETE FROM model_usage WHERE model_name = ?", (model,))
            except Exception:
                pass
        # keep_all strategy does nothing

class CachedEmbeddingModel:
    """Wrapper around SentenceTransformer models implementing automatic caching in SQLite database."""
    
    def __init__(self, model: Any, model_name: str = "BAAI/bge-small-en-v1.5",
                 db_path: str = "data/evaluation/embedding_cache.sqlite",
                 invalidation_strategy: Optional[str] = None,
                 max_models: Optional[int] = None,
                 max_age_days: Optional[int] = None) -> None:
        self.model = model
        self.model_name = model_name
        self.db_path = db_path
        self.cache = SqliteEmbeddingCache(
            db_path=db_path,
            invalidation_strategy=invalidation_strategy,
            max_models=max_models,
            max_age_days=max_age_days
        )
        self.cache.invalidate_other_models(self.model_name)
        # Observability stats
        self.hits = 0
        self.misses = 0
        self.lookup_latencies_ms = []
        self.insert_latencies_ms = []

    def reset_stats(self) -> None:
        """Resets the cache performance tracking stats."""
        self.hits = 0
        self.misses = 0
        self.lookup_latencies_ms = []
        self.insert_latencies_ms = []

    def encode(self, texts: List[str], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
            
        import time
        results = []
        to_encode = []
        to_encode_indices = []
        
        for idx, text in enumerate(texts):
            t0 = time.time()
            cached = self.cache.get(text, self.model_name, normalize_embeddings)
            t1 = time.time()
            self.lookup_latencies_ms.append((t1 - t0) * 1000.0)
            
            if cached is not None:
                self.hits += 1
                results.append((idx, np.array(cached)))
            else:
                self.misses += 1
                to_encode.append(text)
                to_encode_indices.append(idx)
                
        if to_encode:
            # Call actual underlying SentenceTransformer model
            encoded = self.model.encode(to_encode, normalize_embeddings=normalize_embeddings, **kwargs)
            # If encoded is a single vector, wrap in list/numpy array
            if len(to_encode) == 1 and len(encoded.shape) == 1:
                encoded = np.array([encoded])
                
            for text_val, emb, original_idx in zip(to_encode, encoded, to_encode_indices):
                emb_list = [float(x) for x in emb]
                t0 = time.time()
                self.cache.set(text_val, self.model_name, normalize_embeddings, emb_list)
                t1 = time.time()
                self.insert_latencies_ms.append((t1 - t0) * 1000.0)
                results.append((original_idx, np.array(emb_list)))
                
        # Re-sort to maintain original input ordering
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])

    def close(self) -> None:
        self.cache.close()

