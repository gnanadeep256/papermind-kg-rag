import os
import time
from typing import Any, Dict, List
from loguru import logger
from src.kg_retriever import Neo4jKGRetriever

def log_section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f" {title.upper()} ")
    logger.info("=" * 60)

def main() -> None:
    # Initialize retriever
    retriever = Neo4jKGRetriever()
    try:
        retriever.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j. Is the server running? Error: {e}")
        return

    try:
        # 1. Fetch Top Centrality Nodes
        log_section("Top Degree Centrality Listings")
        
        start = time.time()
        top_papers = retriever.get_top_papers(limit=5)
        logger.info(f"get_top_papers(limit=5) took {time.time() - start:.4f}s. Results: {len(top_papers)}")
        for i, p in enumerate(top_papers, 1):
            logger.info(f"  {i}. Title: {p.get('title')} | ID: {p.get('arxiv_id')} | Degree: {p.get('degree')}")

        start = time.time()
        top_methods = retriever.get_top_methods(limit=5)
        logger.info(f"get_top_methods(limit=5) took {time.time() - start:.4f}s. Results: {len(top_methods)}")
        for i, m in enumerate(top_methods, 1):
            logger.info(f"  {i}. Method: {m.get('name')} | ID: {m.get('entity_id')} | Degree: {m.get('degree')}")

        start = time.time()
        top_datasets = retriever.get_top_datasets(limit=5)
        logger.info(f"get_top_datasets(limit=5) took {time.time() - start:.4f}s. Results: {len(top_datasets)}")
        for i, d in enumerate(top_datasets, 1):
            logger.info(f"  {i}. Dataset: {d.get('name')} | ID: {d.get('entity_id')} | Degree: {d.get('degree')}")

        if not top_papers or not top_methods:
            logger.warning("No papers or methods found in the database. Cannot run further tests.")
            return

        # Choose inputs for subsequent tests
        test_arxiv_id = top_papers[0]["arxiv_id"]
        test_title = top_papers[0]["title"]
        test_method_name = top_methods[0]["name"]
        test_entity_id = top_methods[0]["entity_id"]

        logger.info("-" * 60)
        logger.info(f"Selected Test Input Paper ID   : {test_arxiv_id}")
        logger.info(f"Selected Test Input Paper Title: {test_title}")
        logger.info(f"Selected Test Input Method Name: {test_method_name}")
        logger.info(f"Selected Test Input Entity ID  : {test_entity_id}")
        logger.info("-" * 60)

        # 2. Core Retrieval Functions
        log_section("Core Node and Neighbor Retrieval")

        start = time.time()
        paper = retriever.get_paper_by_arxiv_id(test_arxiv_id)
        logger.info(f"get_paper_by_arxiv_id('{test_arxiv_id}') took {time.time() - start:.4f}s")
        if paper:
            logger.info(f"  Found Paper: {paper.get('name')} (Published: {paper.get('published')})")

        start = time.time()
        paper_by_title = retriever.get_paper_by_title(test_title)
        logger.info(f"get_paper_by_title('{test_title[:40]}...') took {time.time() - start:.4f}s")
        if paper_by_title:
            logger.info(f"  Found Paper by Title: {paper_by_title.get('entity_id')}")

        start = time.time()
        authors = retriever.get_authors_of_paper(test_arxiv_id)
        logger.info(f"get_authors_of_paper('{test_arxiv_id}') took {time.time() - start:.4f}s. Authors: {authors}")

        start = time.time()
        methods = retriever.get_methods_for_paper(test_arxiv_id)
        logger.info(f"get_methods_for_paper('{test_arxiv_id}') took {time.time() - start:.4f}s. Methods: {methods}")

        start = time.time()
        datasets = retriever.get_datasets_for_method(test_method_name)
        logger.info(f"get_datasets_for_method('{test_method_name}') took {time.time() - start:.4f}s. Datasets: {datasets}")

        start = time.time()
        tasks = retriever.get_tasks_for_method(test_method_name)
        logger.info(f"get_tasks_for_method('{test_method_name}') took {time.time() - start:.4f}s. Tasks: {tasks}")

        start = time.time()
        papers_about = retriever.get_papers_about_method(test_method_name)
        logger.info(f"get_papers_about_method('{test_method_name}') took {time.time() - start:.4f}s. Papers: {len(papers_about)}")

        start = time.time()
        related_methods = retriever.get_related_methods(test_method_name)
        logger.info(f"get_related_methods('{test_method_name}') took {time.time() - start:.4f}s. Related Methods: {related_methods}")

        search_query = test_method_name[:5]
        start = time.time()
        search_results = retriever.search_entities_by_name(search_query, limit=5)
        logger.info(f"search_entities_by_name('{search_query}') took {time.time() - start:.4f}s. Results: {len(search_results)}")
        for r in search_results:
            logger.info(f"  - [{r.get('entity_type')}] {r.get('name')} (ID: {r.get('entity_id')})")

        # 3. Multi-Hop Graph Retrieval Functions
        log_section("Multi-Hop Graph & Subgraph Retrieval")

        start = time.time()
        method_context = retriever.get_method_context(test_method_name)
        logger.info(f"get_method_context('{test_method_name}') took {time.time() - start:.4f}s")
        logger.info(f"  Nodes count: {len(method_context['nodes'])}")
        logger.info(f"  Relationships count: {len(method_context['relationships'])}")

        start = time.time()
        paper_subgraph = retriever.get_paper_subgraph(test_arxiv_id)
        logger.info(f"get_paper_subgraph('{test_arxiv_id}') took {time.time() - start:.4f}s")
        logger.info(f"  Nodes count: {len(paper_subgraph['nodes'])}")
        logger.info(f"  Relationships count: {len(paper_subgraph['relationships'])}")

        start = time.time()
        neighborhood_1hop = retriever.get_entity_neighborhood(test_entity_id, hops=1)
        logger.info(f"get_entity_neighborhood('{test_entity_id}', hops=1) took {time.time() - start:.4f}s")
        logger.info(f"  Nodes count: {len(neighborhood_1hop['nodes'])}")
        logger.info(f"  Relationships count: {len(neighborhood_1hop['relationships'])}")

        start = time.time()
        neighborhood_2hop = retriever.get_entity_neighborhood(test_entity_id, hops=2)
        logger.info(f"get_entity_neighborhood('{test_entity_id}', hops=2) took {time.time() - start:.4f}s")
        logger.info(f"  Nodes count: {len(neighborhood_2hop['nodes'])}")
        logger.info(f"  Relationships count: {len(neighborhood_2hop['relationships'])}")

        # Connect top 2 papers if we have them
        if len(top_papers) > 1:
            source_p = top_papers[0]["arxiv_id"]
            target_p = top_papers[1]["arxiv_id"]
            start = time.time()
            conn = retriever.find_connection(source_p, target_p)
            logger.info(f"find_connection('{source_p}', '{target_p}') took {time.time() - start:.4f}s")
            logger.info(f"  Nodes count: {len(conn['nodes'])}")
            logger.info(f"  Relationships count: {len(conn['relationships'])}")

    finally:
        retriever.close()

if __name__ == "__main__":
    main()
