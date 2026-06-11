import os
import json
import re
from datetime import datetime
from collections import defaultdict

def extract_year(date_str):
    """Extracts year from standard ISO format date string."""
    if not date_str:
        return 2026  # Default fallback
    try:
        # Match YYYY in the date string
        match = re.match(r"^(\d{4})", date_str)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 2026

def main():
    graph_path = "data/processed/graph_data.json"
    stats_path = "data/processed/graph_stats.json"
    chunks_path = "data/vectorstore/chunk_metadata.json"
    ui_cache_dir = "data/ui_cache"
    
    os.makedirs(ui_cache_dir, exist_ok=True)
    
    # Check if dependencies exist
    if not os.path.exists(graph_path) or not os.path.exists(stats_path) or not os.path.exists(chunks_path):
        print("Error: Missing required files to build UI cache. Please run ingestion first.")
        return
        
    print("Loading graph data, stats, and chunk metadata...")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        graph_stats = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
        
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])
    node_metrics = graph_stats.get("node_metrics", [])
    
    # Create metric maps
    pagerank_map = {m["entity_id"]: m["pagerank"] for m in node_metrics}
    betweenness_map = {m["entity_id"]: m["betweenness"] for m in node_metrics}
    degree_map = {m["entity_id"]: m["raw_degree"] for m in node_metrics}
    desc_map = {m["entity_id"]: m["description"] for m in node_metrics}
    
    # 1. TIMELINE CACHE
    print("Generating Timeline Cache...")
    timeline_items = []
    paper_nodes = [e for e in entities if e["entity_type"] == "Paper"]
    
    for paper in paper_nodes:
        arxiv_id = paper["entity_id"]
        published = paper.get("published", "")
        year = extract_year(published)
        
        timeline_items.append({
            "arxiv_id": arxiv_id,
            "title": paper["name"],
            "published": published,
            "year": year,
            "primary_category": paper.get("primary_category", "cs.AI")
        })
        
    # Sort timeline by year, then by date
    timeline_items.sort(key=lambda x: (x["year"], x["published"]))
    
    with open(os.path.join(ui_cache_dir, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump(timeline_items, f, indent=2)
        
    # Group chunks by paper
    chunks_by_paper = defaultdict(list)
    for chunk in chunk_metadata:
        arxiv_id = chunk.get("arxiv_id")
        if arxiv_id:
            chunks_by_paper[arxiv_id].append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk.get("section", "Unknown"),
                "page_start": chunk.get("page_start", 1),
                "page_end": chunk.get("page_end", 1),
                "word_count": chunk.get("chunk_word_count", 0),
                "text": chunk["text"]
            })
            
    # Sort chunks for each paper by chunk_index
    for aid in chunks_by_paper:
        chunks_by_paper[aid].sort(key=lambda x: int(x["chunk_id"].split("_")[-1]) if "_" in x["chunk_id"] else 0)

    # 2. PAPERS CACHE
    print("Generating Papers Cache...")
    papers_cache = {}
    
    # Gather relations per paper
    paper_relationships = defaultdict(list)
    for rel in relationships:
        src = rel["source"]
        tgt = rel["target"]
        rel_type = rel["relation"]
        
        # Check if src is paper
        if src in [p["entity_id"] for p in paper_nodes]:
            paper_relationships[src].append({
                "target": tgt,
                "target_name": next((e["name"] for e in entities if e["entity_id"] == tgt), tgt),
                "target_type": next((e["entity_type"] for e in entities if e["entity_id"] == tgt), "Unknown"),
                "relation": rel_type,
                "description": rel.get("description", "")
            })
        # Check if tgt is paper (e.g. Author -WRITES-> Paper)
        if tgt in [p["entity_id"] for p in paper_nodes]:
            paper_relationships[tgt].append({
                "source": src,
                "source_name": next((e["name"] for e in entities if e["entity_id"] == src), src),
                "source_type": next((e["entity_type"] for e in entities if e["entity_id"] == src), "Unknown"),
                "relation": rel_type,
                "description": rel.get("description", "")
            })
            
    for paper in paper_nodes:
        aid = paper["entity_id"]
        papers_cache[aid] = {
            "arxiv_id": aid,
            "title": paper["name"],
            "abstract": paper.get("description", ""),
            "published": paper.get("published", ""),
            "primary_category": paper.get("primary_category", "cs.AI"),
            "pdf_url": paper.get("pdf_url", ""),
            "pagerank": pagerank_map.get(aid, 0.0),
            "betweenness": betweenness_map.get(aid, 0.0),
            "degree": degree_map.get(aid, 0),
            "chunks": chunks_by_paper.get(aid, []),
            "relationships": paper_relationships.get(aid, [])
        }
        
    with open(os.path.join(ui_cache_dir, "papers_cache.json"), "w", encoding="utf-8") as f:
        json.dump(papers_cache, f, indent=2)

    # 3. METHODS CACHE
    print("Generating Methods Cache...")
    methods_cache = {}
    method_nodes = [e for e in entities if e["entity_type"] == "Method"]
    
    for method in method_nodes:
        mid = method["entity_id"]
        
        # Gather relations involving this method
        method_rels = []
        introduced_by = []
        using_papers = []
        eval_datasets = []
        related_methods = []
        
        for rel in relationships:
            src = rel["source"]
            tgt = rel["target"]
            rel_type = rel["relation"]
            
            if src == mid:
                method_rels.append({
                    "target": tgt,
                    "target_name": next((e["name"] for e in entities if e["entity_id"] == tgt), tgt),
                    "target_type": next((e["entity_type"] for e in entities if e["entity_id"] == tgt), "Unknown"),
                    "relation": rel_type,
                    "description": rel.get("description", "")
                })
                if rel_type == "EVALUATED_ON" or rel_type == "USES":
                    target_type = next((e["entity_type"] for e in entities if e["entity_id"] == tgt), "")
                    if target_type == "Dataset":
                        eval_datasets.append(next((e["name"] for e in entities if e["entity_id"] == tgt), tgt))
                if rel_type == "COMPARED_WITH" or rel_type == "EXTENDS":
                    target_type = next((e["entity_type"] for e in entities if e["entity_id"] == tgt), "")
                    if target_type == "Method":
                        related_methods.append({
                            "method_name": next((e["name"] for e in entities if e["entity_id"] == tgt), tgt),
                            "relation": rel_type
                        })
            elif tgt == mid:
                method_rels.append({
                    "source": src,
                    "source_name": next((e["name"] for e in entities if e["entity_id"] == src), src),
                    "source_type": next((e["entity_type"] for e in entities if e["entity_id"] == src), "Unknown"),
                    "relation": rel_type,
                    "description": rel.get("description", "")
                })
                if rel_type == "INTRODUCES":
                    source_type = next((e["entity_type"] for e in entities if e["entity_id"] == src), "")
                    if source_type == "Paper":
                        introduced_by.append({
                            "arxiv_id": src,
                            "title": next((e["name"] for e in entities if e["entity_id"] == src), src)
                        })
                elif rel_type == "MENTIONS" or rel_type == "USES":
                    source_type = next((e["entity_type"] for e in entities if e["entity_id"] == src), "")
                    if source_type == "Paper":
                        using_papers.append({
                            "arxiv_id": src,
                            "title": next((e["name"] for e in entities if e["entity_id"] == src), src)
                        })
                        
        methods_cache[method["name"]] = {
            "entity_id": mid,
            "name": method["name"],
            "description": desc_map.get(mid, ""),
            "pagerank": pagerank_map.get(mid, 0.0),
            "degree": degree_map.get(mid, 0),
            "introduced_by": introduced_by,
            "using_papers": using_papers,
            "datasets": eval_datasets,
            "related_methods": related_methods,
            "relationships": method_rels
        }
        
    with open(os.path.join(ui_cache_dir, "methods_cache.json"), "w", encoding="utf-8") as f:
        json.dump(methods_cache, f, indent=2)

    # 4. DATASETS CACHE
    print("Generating Datasets Cache...")
    datasets_cache = {}
    dataset_nodes = [e for e in entities if e["entity_type"] == "Dataset"]
    
    for dataset in dataset_nodes:
        did = dataset["entity_id"]
        
        dataset_rels = []
        evaluating_papers = []
        using_methods = []
        
        for rel in relationships:
            src = rel["source"]
            tgt = rel["target"]
            rel_type = rel["relation"]
            
            if src == did:
                dataset_rels.append({
                    "target": tgt,
                    "target_name": next((e["name"] for e in entities if e["entity_id"] == tgt), tgt),
                    "target_type": next((e["entity_type"] for e in entities if e["entity_id"] == tgt), "Unknown"),
                    "relation": rel_type,
                    "description": rel.get("description", "")
                })
            elif tgt == did:
                dataset_rels.append({
                    "source": src,
                    "source_name": next((e["name"] for e in entities if e["entity_id"] == src), src),
                    "source_type": next((e["entity_type"] for e in entities if e["entity_id"] == src), "Unknown"),
                    "relation": rel_type,
                    "description": rel.get("description", "")
                })
                source_type = next((e["entity_type"] for e in entities if e["entity_id"] == src), "")
                if source_type == "Paper":
                    evaluating_papers.append({
                        "arxiv_id": src,
                        "title": next((e["name"] for e in entities if e["entity_id"] == src), src)
                    })
                elif source_type == "Method":
                    using_methods.append(next((e["name"] for e in entities if e["entity_id"] == src), src))
                    
        datasets_cache[dataset["name"]] = {
            "entity_id": did,
            "name": dataset["name"],
            "description": desc_map.get(did, ""),
            "pagerank": pagerank_map.get(did, 0.0),
            "degree": degree_map.get(did, 0),
            "evaluating_papers": evaluating_papers,
            "methods": using_methods,
            "relationships": dataset_rels
        }
        
    with open(os.path.join(ui_cache_dir, "datasets_cache.json"), "w", encoding="utf-8") as f:
        json.dump(datasets_cache, f, indent=2)

    # 5. GLOBAL PYVIS GRAPH CACHE
    print("Generating Global Pyvis Graph Cache...")
    
    # Custom node styling by label type
    colors = {
        "Paper": "#1f77b4",        # Blue
        "Method": "#2ca02c",       # Green
        "Dataset": "#ff7f0e",      # Orange
        "Author": "#9467bd",       # Purple
        "Concept": "#d62728",      # Red
        "Category": "#bcbd22",     # Yellow-Green
        "Metric": "#17becf",       # Cyan
        "Task": "#8c564b",         # Brown
        "Organization": "#e377c2"  # Pink
    }
    
    pyvis_nodes = []
    for e in entities:
        eid = e["entity_id"]
        etype = e["entity_type"]
        name = e["name"]
        
        pr = pagerank_map.get(eid, 0.0)
        # Size scale: min size 10, max size 50 based on pagerank
        val = 10 + int(pr * 500)
        
        pyvis_nodes.append({
            "id": eid,
            "label": name if len(name) < 25 else name[:22] + "...",
            "title": f"Name: {name}<br>Type: {etype}<br>Degree: {degree_map.get(eid, 0)}<br>PageRank: {pr:.5f}",
            "color": colors.get(etype, "#7f7f7f"),
            "value": val,
            "group": etype
        })
        
    pyvis_edges = []
    for r in relationships:
        src = r["source"]
        tgt = r["target"]
        rel = r["relation"]
        
        pyvis_edges.append({
            "from": src,
            "to": tgt,
            "label": rel,
            "title": r.get("description", "")
        })
        
    global_graph = {
        "nodes": pyvis_nodes,
        "edges": pyvis_edges
    }
    
    with open(os.path.join(ui_cache_dir, "global_graph_pyvis.json"), "w", encoding="utf-8") as f:
        json.dump(global_graph, f, indent=2)

    # 6. PROJECT INSIGHTS CACHE
    print("Generating Project Insights Cache...")
    
    # Most influential papers (highest PageRank Paper nodes)
    influential_papers = [n for n in node_metrics if n["entity_type"] == "Paper"]
    influential_papers.sort(key=lambda x: x["pagerank"], reverse=True)
    
    # Most influential methods
    influential_methods = [n for n in node_metrics if n["entity_type"] == "Method"]
    influential_methods.sort(key=lambda x: x["pagerank"], reverse=True)
    
    # Most connected datasets
    influential_datasets = [n for n in node_metrics if n["entity_type"] == "Dataset"]
    influential_datasets.sort(key=lambda x: x["pagerank"], reverse=True)
    
    # Most connected concepts
    influential_concepts = [n for n in node_metrics if n["entity_type"] == "Concept"]
    influential_concepts.sort(key=lambda x: x["pagerank"], reverse=True)
    
    # Oldest / Newest papers
    valid_papers = [p for p in paper_nodes if p.get("published")]
    valid_papers.sort(key=lambda x: x["published"])
    
    oldest_paper = {
        "title": valid_papers[0]["name"],
        "published": valid_papers[0]["published"],
        "arxiv_id": valid_papers[0]["entity_id"],
        "year": extract_year(valid_papers[0]["published"])
    } if valid_papers else {"title": "N/A", "published": "", "arxiv_id": "", "year": 2026}
    
    newest_paper = {
        "title": valid_papers[-1]["name"],
        "published": valid_papers[-1]["published"],
        "arxiv_id": valid_papers[-1]["entity_id"],
        "year": extract_year(valid_papers[-1]["published"])
    } if valid_papers else {"title": "N/A", "published": "", "arxiv_id": "", "year": 2026}
    
    # Average publication year
    pub_years = [extract_year(p.get("published")) for p in paper_nodes if p.get("published")]
    avg_pub_year = sum(pub_years) / len(pub_years) if pub_years else 2026
    
    # Category research topics counts
    categories_list = [e for e in entities if e["entity_type"] == "Category"]
    categories_list.sort(key=lambda x: degree_map.get(x["entity_id"], 0), reverse=True)
    most_common_topic = categories_list[0]["name"] if categories_list else "Unknown"
    
    # Average chunks per paper
    total_papers = len(paper_nodes)
    total_chunks = len(chunk_metadata)
    avg_chunks_per_paper = total_chunks / total_papers if total_papers > 0 else 0
    
    insights = {
        "most_influential_paper": {
            "title": influential_papers[0]["name"] if influential_papers else "N/A",
            "arxiv_id": influential_papers[0]["entity_id"] if influential_papers else "",
            "pagerank": influential_papers[0]["pagerank"] if influential_papers else 0.0
        },
        "most_connected_method": {
            "name": influential_methods[0]["name"] if influential_methods else "N/A",
            "pagerank": influential_methods[0]["pagerank"] if influential_methods else 0.0
        },
        "most_reused_dataset": {
            "name": influential_datasets[0]["name"] if influential_datasets else "N/A",
            "pagerank": influential_datasets[0]["pagerank"] if influential_datasets else 0.0
        },
        "most_connected_concept": {
            "name": influential_concepts[0]["name"] if influential_concepts else "N/A",
            "pagerank": influential_concepts[0]["pagerank"] if influential_concepts else 0.0
        },
        "oldest_paper": oldest_paper,
        "newest_paper": newest_paper,
        "average_publication_year": avg_pub_year,
        "most_common_research_topic": most_common_topic,
        "largest_connected_component_size": graph_stats.get("metadata", {}).get("largest_component_size", 0),
        "average_degree": graph_stats.get("metadata", {}).get("average_degree", 0.0),
        "average_chunks_per_paper": avg_chunks_per_paper
    }
    
    with open(os.path.join(ui_cache_dir, "project_insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
        
    print("UI caches generation completed successfully.")

if __name__ == "__main__":
    main()
