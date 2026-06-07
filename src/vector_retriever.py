import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Dict, Any, List
from src.utils.config import load_config

class VectorRetriever:
    """
    Retrieves relevant text chunks from the local FAISS index using semantic similarity.
    """
    def __init__(self, vectorstore_dir: str = "data/vectorstore") -> None:
        self.vectorstore_dir = vectorstore_dir
        self.index_path = os.path.join(vectorstore_dir, "faiss.index")
        self.metadata_path = os.path.join(vectorstore_dir, "chunk_metadata.json")
        
        try:
            config = load_config()
            self.model_name = config.get("embeddings", {}).get("model_name", "BAAI/bge-small-en-v1.5")
        except Exception:
            self.model_name = "BAAI/bge-small-en-v1.5"
            
        self.model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
    def load(self) -> None:
        """Loads the embedding model, FAISS index, and chunk metadata into memory."""
        logger.info(f"Loading Vector Retriever resources from {self.vectorstore_dir}...")
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Vector store files not found in {self.vectorstore_dir}. Please run the construction pipeline first."
            )
            
        # Load embedding model
        logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
        from src.evaluation.embedding_cache import CachedEmbeddingModel
        base_model = SentenceTransformer(self.model_name)
        self.model = CachedEmbeddingModel(base_model, model_name=self.model_name)
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata mappings
        logger.info(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        logger.info("Vector Retriever resources loaded successfully.")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the query (prefixed with the official BGE instruction) and performs 
        an Inner Product search on the FAISS index.
        Returns a list of top-k matches with similarity scores and metadata.
        """
        if self.model is None or self.index is None or not self.metadata:
            self.load()
            
        # Add BGE query instruction prefix for retrieval tasks
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        
        # Encode and normalize query embedding
        query_vector = self.model.encode([prefixed_query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype('float32')
        
        # Perform FAISS search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 if no match is found
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            chunk_info = self.metadata[idx]
            results.append({
                "score": float(score),
                "chunk_id": chunk_info.get("chunk_id"),
                "arxiv_id": chunk_info.get("arxiv_id"),
                "title": chunk_info.get("title"),
                "section": chunk_info.get("section", "Unknown"),
                "page_start": chunk_info.get("page_start"),
                "page_end": chunk_info.get("page_end"),
                "chunk_word_count": chunk_info.get("chunk_word_count"),
                "text": chunk_info.get("text")
            })
            
        return results
