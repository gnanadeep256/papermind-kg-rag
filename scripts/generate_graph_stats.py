import os
import json
import time
from collections import Counter, defaultdict
import networkx as nx

def main():
    graph_path = "data/processed/graph_data.json"
    stats_out_path = "data/processed/graph_stats.json"
    
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found. Please run the extraction pipeline first.")
        return

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    print("====================================================")
    print("      PAPERMIND KNOWLEDGE GRAPH STATISTICS REPORT    ")
    print("====================================================")
    print(f"Generated At:       {metadata.get('generated_at', 'N/A')}")
    print(f"LLM Provider:       {metadata.get('llm_provider', 'N/A')}")
    print(f"Papers Processed:   {metadata.get('papers_processed', 0)}")
    print(f"Papers Failed:      {metadata.get('papers_failed', 0)}")
    print("----------------------------------------------------")

    # 1. Entity Type Counts
    entity_counts = Counter(e.get("entity_type") for e in entities)
    print("ENTITY COUNTS BY TYPE:")
    for etype in ["Paper", "Author", "Category", "Method", "Concept", "Dataset", "Metric", "Task", "Organization"]:
        print(f"  - {etype:<15}: {entity_counts.get(etype, 0)}")
    print(f"  * Total Entities  : {len(entities)}")
    print("----------------------------------------------------")

    # 2. Relationship Type Counts
    relation_counts = Counter(r.get("relation") for r in relationships)
    print("RELATIONSHIP COUNTS BY TYPE:")
    for rel in sorted(list(relation_counts.keys())):
        print(f"  - {rel:<15}: {relation_counts.get(rel, 0)}")
    print(f"  * Total Edges     : {len(relationships)}")
    print("----------------------------------------------------")

    # Construct NetworkX Graphs
    print("Building NetworkX Graphs...")
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()

    # Map entity ID to details
    id_to_name = {}
    id_to_type = {}
    id_to_desc = {}
    for e in entities:
        eid = e.get("entity_id")
        id_to_name[eid] = e.get("name")
        id_to_type[eid] = e.get("entity_type")
        id_to_desc[eid] = e.get("description", "")
        
        G_directed.add_node(eid, type=e.get("entity_type"), name=e.get("name"))
        G_undirected.add_node(eid, type=e.get("entity_type"), name=e.get("name"))

    for r in relationships:
        src = r.get("source")
        tgt = r.get("target")
        rel_type = r.get("relation")
        
        G_directed.add_edge(src, tgt, type=rel_type)
        G_undirected.add_edge(src, tgt, type=rel_type)

    total_nodes = G_directed.number_of_nodes()
    total_edges = G_directed.number_of_edges()

    print(f"NetworkX Graph: {total_nodes} nodes, {total_edges} edges.")

    # Calculate Centralities
    print("Computing NetworkX graph analytics metrics...")
    pagerank_scores = {}
    betweenness_scores = {}
    closeness_scores = {}
    degree_scores = {}

    if total_nodes > 0:
        try:
            pagerank_scores = nx.pagerank(G_directed, alpha=0.85)
        except Exception as e:
            print(f"  Warning: PageRank computation failed: {e}")
            pagerank_scores = {n: 0.0 for n in G_directed.nodes()}

        try:
            betweenness_scores = nx.betweenness_centrality(G_undirected)
        except Exception as e:
            print(f"  Warning: Betweenness Centrality failed: {e}")
            betweenness_scores = {n: 0.0 for n in G_undirected.nodes()}

        try:
            closeness_scores = nx.closeness_centrality(G_directed)
        except Exception as e:
            print(f"  Warning: Closeness Centrality failed: {e}")
            closeness_scores = {n: 0.0 for n in G_directed.nodes()}

        try:
            degree_scores = nx.degree_centrality(G_undirected)
        except Exception as e:
            print(f"  Warning: Degree Centrality failed: {e}")
            degree_scores = {n: 0.0 for n in G_undirected.nodes()}

    # Connected Components (Undirected)
    connected_components = []
    if total_nodes > 0:
        components = sorted(nx.connected_components(G_undirected), key=len, reverse=True)
        connected_components = [list(c) for c in components]
        print(f"Connected Components: {len(components)} components. Largest component size: {len(components[0]) if components else 0}")
    
    # Community Detection (Louvain Modularity)
    communities = []
    if total_nodes > 0:
        try:
            from networkx.algorithms.community import louvain_communities
            communities_sets = louvain_communities(G_undirected, seed=42)
            communities = [list(c) for c in communities_sets]
            print(f"Louvain Modularity Communities Detected: {len(communities)}")
        except Exception as e:
            print(f"  Warning: Louvain community detection failed: {e}. Falling back to Label Propagation.")
            try:
                from networkx.algorithms.community import label_propagation_communities
                communities_sets = label_propagation_communities(G_undirected)
                communities = [list(c) for c in communities_sets]
                print(f"Label Propagation Communities Detected: {len(communities)}")
            except Exception as e2:
                print(f"  Warning: Community detection failed: {e2}")

    # Map node stats to lists for serialization
    node_analytics = []
    for node_id in G_directed.nodes():
        node_analytics.append({
            "entity_id": node_id,
            "name": id_to_name.get(node_id, node_id),
            "entity_type": id_to_type.get(node_id, "Unknown"),
            "description": id_to_desc.get(node_id, ""),
            "pagerank": pagerank_scores.get(node_id, 0.0),
            "betweenness": betweenness_scores.get(node_id, 0.0),
            "closeness": closeness_scores.get(node_id, 0.0),
            "degree_centrality": degree_scores.get(node_id, 0.0),
            "raw_degree": G_undirected.degree(node_id)
        })

    # Sort node lists for reports
    pagerank_sorted = sorted(node_analytics, key=lambda x: x["pagerank"], reverse=True)
    betweenness_sorted = sorted(node_analytics, key=lambda x: x["betweenness"], reverse=True)
    degree_sorted = sorted(node_analytics, key=lambda x: x["raw_degree"], reverse=True)

    # Output top PageRank nodes
    print("\nTOP 20 NODES BY PAGERANK CENTRALITY:")
    print(f"  {'Rank':<5} | {'Node Name':<35} | {'Type':<12} | {'PageRank':<10}")
    print("  " + "-" * 70)
    for rank, node in enumerate(pagerank_sorted[:20], 1):
        name = node["name"]
        if len(name) > 32:
            name = name[:29] + "..."
        print(f"  {rank:<5} | {name:<35} | {node['entity_type']:<12} | {node['pagerank']:.5f}")

    # Output top Betweenness nodes
    print("\nTOP 20 NODES BY BETWEENNESS CENTRALITY:")
    print(f"  {'Rank':<5} | {'Node Name':<35} | {'Type':<12} | {'Betweenness':<10}")
    print("  " + "-" * 72)
    for rank, node in enumerate(betweenness_sorted[:20], 1):
        name = node["name"]
        if len(name) > 32:
            name = name[:29] + "..."
        print(f"  {rank:<5} | {name:<35} | {node['entity_type']:<12} | {node['betweenness']:.5f}")

    # Serialize analytics payload
    stats_payload = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": nx.density(G_undirected) if total_nodes > 0 else 0.0,
            "average_degree": sum(dict(G_undirected.degree()).values()) / total_nodes if total_nodes > 0 else 0.0,
            "connected_components_count": len(connected_components),
            "largest_component_size": len(connected_components[0]) if connected_components else 0,
            "communities_count": len(communities)
        },
        "node_metrics": node_analytics,
        "connected_components": connected_components,
        "communities": communities
    }

    # Save graph stats
    print(f"Saving precomputed graph statistics to {stats_out_path}...")
    with open(stats_out_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
        
    print("Graph analytics precomputation completed successfully.")
    print("====================================================")

if __name__ == "__main__":
    main()
