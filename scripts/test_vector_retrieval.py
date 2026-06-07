import sys
import time
from src.vector_retriever import VectorRetriever
from loguru import logger

# Reconfigure stdout to support UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main() -> None:
    logger.info("Initializing vector retrieval validation CLI...")
    
    try:
        retriever = VectorRetriever()
        retriever.load()
    except Exception as e:
        logger.error(f"Failed to load Vector Retriever. Has the index been constructed? Error: {e}")
        return

    queries = [
        "Explain TempoVLA",
        "What is Code2LoRA?",
        "How does HANDOFF work?"
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{q}'")
        print("=" * 80)
        
        start = time.time()
        results = retriever.search(q, k=3)
        duration = time.time() - start
        
        print(f"Retrieval took {duration:.4f}s. Top 3 results:")
        print("-" * 80)
        
        for idx, r in enumerate(results, 1):
            print(f"Result {idx} [Score: {r['score']:.4f}]")
            print(f"Paper: {r['title']} (arXiv: {r['arxiv_id']})")
            print(f"Section: {r['section']}")
            page_str = f"{r['page_start']}-{r['page_end']}" if r['page_start'] != r['page_end'] else f"{r['page_start']}"
            print(f"Pages: {page_str} | Words: {r['chunk_word_count']}")
            print(f"Chunk ID: {r['chunk_id']}")
            print("-" * 40)
            
            # Print a snippet of the text
            snippet = r['text'][:400] + "..." if len(r['text']) > 400 else r['text']
            print(f"Snippet:\n{snippet}")
            print("-" * 80)

if __name__ == "__main__":
    main()
