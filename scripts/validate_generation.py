import sys
from loguru import logger

from src.hybrid_retriever import HybridRetriever
from src.answer_generator import GroundedAnswerGenerator

# Reconfigure stdout to support UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main() -> None:
    logger.info("Initializing Grounded Answer Generation validation CLI...")
    
    try:
        retriever = HybridRetriever()
        retriever.load()
        generator = GroundedAnswerGenerator(retriever)
    except Exception as e:
        logger.error(f"Failed to initialize Grounded Generator components: {e}")
        return

    queries = [
        "Summarize HANDOFF.",
        "How does TempoVLA differ from HANDOFF?",
        "What datasets evaluate Code2LoRA?",
        "What is the recipe for baking a chocolate cake?"  # Expected to trigger abstention
    ]

    for q in queries:
        print("\n" + "=" * 90)
        print(f"QUERY: '{q}'")
        print("=" * 90)
        
        try:
            result = generator.generate_answer(q)
            
            print(f"Detected Category: {result.metadata['category'].upper()}")
            print(f"LLM Provider Used: {result.metadata['provider_used'].upper()} (Fallback used: {result.metadata['fallback_used']})")
            print(f"Confidence Score : {result.confidence * 100:.1f}%")
            prec = result.metadata.get('citation_precision')
            prec_str = f"{prec * 100:.1f}%" if prec is not None else "N/A (No citations generated)"
            print(f"Citation Precision: {prec_str}")
            print(f"Citation Coverage : {result.metadata['answer_coverage'] * 100:.1f}%")
            print("-" * 90)
            
            print("ANSWER:")
            print(result.answer)
            print("-" * 90)
            
            print(f"Validated Citations ({len(result.citations)}):")
            for c in result.citations:
                sel_by = getattr(c, "selected_by", [])
                print(f"  * {c.paper_title} (arXiv: {c.arxiv_id}) | Section: {c.section} | Pages: {c.page_start}-{c.page_end} | Selected By: {sel_by}")
                
            if result.metadata.get("invalid_citations"):
                print(f"Invalid (Hallucinated) Citations Logged: {result.metadata['invalid_citations']}")
            print("-" * 90)
            
            prov = result.provenance
            print("Provenance Stats:")
            print(f"  - Papers Used      : {prov['papers_used']}")
            print(f"  - Chunks Used      : {prov['chunks_used']}")
            print(f"  - Graph Nodes Used : {prov['graph_nodes_used']}")
            print(f"  - Graph Edges Used : {prov['graph_edges_used']}")
            if "features" in prov:
                print(f"  - Features Active  : {prov['features']}")
            print(f"  - Retrieval Latency: {prov['retrieval_time_ms']:.2f} ms")
            print(f"  - Gen Latency      : {prov['generation_time_ms']:.2f} ms")
            print(f"  - Total Latency    : {result.metadata['total_execution_time_ms']:.2f} ms")
            print("." * 90)
            
        except Exception as e:
            logger.error(f"Failed to generate answer for query '{q}': {e}")
            
    # Close retriever driver connection
    retriever.close()

if __name__ == "__main__":
    main()
