import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

def main() -> None:
    logger.info("Initializing vector store builder...")
    
    chunks_path = "data/processed/chunks.json"
    vectorstore_dir = "data/vectorstore"
    os.makedirs(vectorstore_dir, exist_ok=True)
    
    index_path = os.path.join(vectorstore_dir, "faiss.index")
    metadata_path = os.path.join(vectorstore_dir, "chunk_metadata.json")
    provenance_path = os.path.join(vectorstore_dir, "index_metadata.json")
    
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found at {chunks_path}. Run chunk_documents first.")
        return
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    if not chunks:
        logger.warning("No chunks found to embed.")
        return
        
    logger.info(f"Loaded {len(chunks)} chunks. Loading embedding model BAAI/bge-small-en-v1.5...")
    
    # Load SentenceTransformer model
    model_name = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_name)
    dimension = 384
    
    logger.info("Generating embeddings for chunks...")
    chunk_texts = [c["text"] for c in chunks]
    
    start_time = time.time()
    # Generate normalized embeddings to allow Inner Product to compute Cosine Similarity
    embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    embedding_duration = time.time() - start_time
    logger.info(f"Generated embeddings in {embedding_duration:.2f} seconds.")
    
    # Build FAISS Index Flat Inner Product
    logger.info(f"Building FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save FAISS Index
    logger.info(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Save chunk metadata aligned with the index order
    logger.info(f"Saving chunk metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        
    # Save index model provenance tracking metadata
    logger.info(f"Saving index metadata to {provenance_path}...")
    provenance_metadata = {
        "embedding_model": model_name,
        "dimension": dimension,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(chunks)
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_metadata, f, indent=2)
        
    logger.info("Vector store construction completed successfully.")

if __name__ == "__main__":
    main()
