import sys
import time
from src.hybrid_retriever import HybridRetriever
from loguru import logger

# Reconfigure stdout to support UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main() -> None:
    logger.info("Initializing Hybrid Retrieval validation CLI...")
    
    try:
        retriever = HybridRetriever()
        retriever.load()
    except Exception as e:
        logger.error(f"Failed to load Hybrid Retriever: {e}")
        return

    queries = [
        "Explain TempoVLA",
        "How does HANDOFF work?",
        "What is Code2LoRA?",
        "How do humanoid robots learn?",
        "What datasets are used for evaluation?"
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{q}'")
        print("=" * 80)
        
        start_time = time.time()
        result = retriever.retrieve(q, top_k_vector=4, top_k_graph=8, max_chunks_per_paper=2)
        total_time_ms = (time.time() - start_time) * 1000
        
        meta = result.retrieval_metadata
        
        print(f"Detected Intent: {meta['intent'].upper()} | Routing Strategy: {meta['routing_strategy'].upper()}")
        print(f"Performance Profile:")
        print(f"  - Vector retrieval : {meta['vector_time_ms']:.2f} ms ({meta['vector_candidates']} candidates)")
        print(f"  - Graph retrieval  : {meta['graph_time_ms']:.2f} ms ({meta['entity_matches']} query matched entities)")
        print(f"  - Reranking time   : {meta['rerank_time_ms']:.2f} ms")
        print(f"  - Total Fusion time: {meta['fusion_time_ms']:.2f} ms")
        print(f"  - Script wrapper   : {total_time_ms:.2f} ms")
        print(f"Token Budget Profile:")
        print(f"  - Token Budget     : {meta['token_budget']} tokens")
        print(f"  - Tokens Used      : {meta['token_used']} tokens")
        print(f"  - Packed Chunks    : {meta['packed_chunks']} blocks")
        print("-" * 80)
        
        print(f"Deduplicated Source Papers ({len(result.source_papers)}):")
        for p in result.source_papers:
            print(f"  * {p['title']} (arXiv: {p['arxiv_id']})")
        print("-" * 80)
        
        print(f"Graph Context Topology:")
        print(f"  - Nodes         : {meta['graph_nodes']}")
        print(f"  - Relationships : {meta['graph_relationships']}")
        
        # Print coverage metrics
        cov = meta.get("coverage", {})
        print(f"  - Coverage      : {cov.get('papers', 0)} papers, {cov.get('methods', 0)} methods, {cov.get('datasets', 0)} datasets, {cov.get('concepts', 0)} concepts")
        
        # Print first few nodes/rels as sample
        if result.graph_context["nodes"]:
            sample_nodes = [n["name"] for n in result.graph_context["nodes"][:5] if n.get("name")]
            print(f"  - Sample Entities: {', '.join(sample_nodes)}")
        if result.graph_context["relationships"]:
            sample_rels = [f"{r['source']}--{r['relation']}-->{r['target']}" for r in result.graph_context["relationships"][:3]]
            print(f"  - Sample Edges   : {', '.join(sample_rels)}")
        print("-" * 80)
        
        print(f"Vector Context Citations & Context Formatting ({len(result.vector_context)} chunks):")
        print("-" * 80)
        for idx, (chunk, citation) in enumerate(zip(result.vector_context, result.citations), 1):
            print(f"Result {idx} [Score: {chunk['combined_score']:.4f} | Sim: {chunk['similarity_score']:.4f} | Graph Bonus: {chunk['graph_bonus']:.4f}]")
            print(f"Citation: {citation.paper_title} (arXiv: {citation.arxiv_id}) | Section: {citation.section} | Pages: {citation.page_start}-{citation.page_end} | Words: {chunk['chunk_word_count']}")
            print(f"Chunk ID: {citation.chunk_id}")
            
            # Print structured explanations
            exps = chunk.get("explanations", [])
            exp_strs = []
            for exp in exps:
                if exp["type"] == "semantic_match":
                    exp_strs.append(f"Semantic match (score: {exp['score']:.4f})")
                elif exp["type"] == "graph_neighbor":
                    exp_strs.append(f"Graph neighbor ({exp['path_length']} hops)")
                elif exp["type"] == "introduced_method":
                    exp_strs.append("Introduced method")
            print(f"Explanations: {', '.join(exp_strs)}")
            
            print("Formatted Context text:")
            # Indent the context text slightly for neatness
            indented_context = "    " + chunk["context_text"].replace("\n", "\n    ")
            print(indented_context[:300] + "...\n")
            print("." * 60)
            
    # Close retriever driver connection
    retriever.close()

if __name__ == "__main__":
    main()
