import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
from src.utils.config import load_config

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
        
    # Check config for incremental option
    incremental = False
    try:
        config = load_config()
        incremental = config.get("corpus", {}).get("incremental", True)
    except Exception as e:
        logger.warning(f"Could not load config for incremental check: {e}. Defaulting to full rebuild.")

    model_name = "BAAI/bge-small-en-v1.5"
    dimension = 384
    
    # Check if index exists and we can append incrementally
    if incremental and os.path.exists(index_path) and os.path.exists(metadata_path):
        logger.info("Incremental build enabled and existing vectorstore found. Loading existing data...")
        try:
            # Load existing index
            index = faiss.read_index(index_path)
            
            # Load existing metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
                
            existing_ids = {c["chunk_id"] for c in existing_metadata if "chunk_id" in c}
            
            # Filter new chunks
            new_chunks = [c for c in chunks if c.get("chunk_id") not in existing_ids]
            
            if not new_chunks:
                logger.info("No new chunks to index. Vector store is up to date.")
                return
                
            logger.info(f"Found {len(new_chunks)} new chunks out of {len(chunks)} total. Loading embedding model...")
            model = SentenceTransformer(model_name)
            
            logger.info("Generating embeddings for new chunks...")
            new_texts = [c["text"] for c in new_chunks]
            
            start_time = time.time()
            embeddings = model.encode(new_texts, show_progress_bar=True, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype('float32')
            embedding_duration = time.time() - start_time
            logger.info(f"Generated embeddings in {embedding_duration:.2f} seconds.")
            
            # Add to index
            logger.info("Appending new embeddings to FAISS index...")
            index.add(embeddings)
            
            # Save updated index
            logger.info(f"Saving updated FAISS index to {index_path}...")
            faiss.write_index(index, index_path)
            
            # Merge and save metadata
            merged_metadata = existing_metadata + new_chunks
            logger.info(f"Saving merged chunk metadata ({len(merged_metadata)} items) to {metadata_path}...")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(merged_metadata, f, indent=2)
                
            # Update provenance metadata
            logger.info(f"Saving index metadata to {provenance_path}...")
            provenance_metadata = {
                "embedding_model": model_name,
                "dimension": dimension,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_chunks": len(merged_metadata),
                "incremental_update": True
            }
            with open(provenance_path, "w", encoding="utf-8") as f:
                json.dump(provenance_metadata, f, indent=2)
                
            logger.info("Vector store incremental update completed successfully.")
            return
            
        except Exception as e:
            logger.error(f"Error during incremental vectorstore update: {e}. Falling back to full rebuild.")
            
    # Full rebuild fallback
    logger.info(f"Performing full rebuild. Loaded {len(chunks)} chunks. Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating embeddings for all chunks...")
    chunk_texts = [c["text"] for c in chunks]
    
    start_time = time.time()
    embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    embedding_duration = time.time() - start_time
    logger.info(f"Generated embeddings in {embedding_duration:.2f} seconds.")
    
    logger.info(f"Building FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    logger.info(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    logger.info(f"Saving chunk metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        
    logger.info(f"Saving index metadata to {provenance_path}...")
    provenance_metadata = {
        "embedding_model": model_name,
        "dimension": dimension,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(chunks)
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_metadata, f, indent=2)
        
    logger.info("Vector store full rebuild completed successfully.")

if __name__ == "__main__":
    main()
