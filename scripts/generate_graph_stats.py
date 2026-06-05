import os
import json
from collections import Counter, defaultdict

def main():
    graph_path = "data/processed/graph_data.json"
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
    all_relations = sorted(list(relation_counts.keys()))
    for rel in all_relations:
        print(f"  - {rel:<15}: {relation_counts.get(rel, 0)}")
    print(f"  * Total Edges     : {len(relationships)}")
    print("----------------------------------------------------")

    # 3. Density & Connectivity Averages
    papers_processed = metadata.get("papers_processed", 0)
    if papers_processed > 0:
        avg_entities = len(entities) / papers_processed
        avg_relationships = len(relationships) / papers_processed
        print(f"AVERAGE GRAPH DENSITY PER PAPER:")
        print(f"  - Avg Entities / Paper     : {avg_entities:.2f}")
        print(f"  - Avg Relationships / Paper: {avg_relationships:.2f}")
    print("----------------------------------------------------")

    # 4. Top 20 Connected Nodes (In/Out/Total Degree Centrality)
    in_degree = Counter()
    out_degree = Counter()
    degree_counts = Counter()
    
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        out_degree[source] += 1
        in_degree[target] += 1
        degree_counts[source] += 1
        degree_counts[target] += 1

    # Map normalized ID back to display name
    id_to_name = {e.get("entity_id"): e.get("name") for e in entities}
    id_to_type = {e.get("entity_id"): e.get("entity_type") for e in entities}

    print("TOP 20 MOST CONNECTED NODES:")
    top_nodes = degree_counts.most_common(20)
    print(f"  {'Rank':<5} | {'Node Name':<35} | {'Type':<12} | {'In':<4} | {'Out':<4} | {'Total':<5}")
    print("  " + "-" * 75)
    for rank, (node_id, degree) in enumerate(top_nodes, 1):
        name = id_to_name.get(node_id, node_id)
        ntype = id_to_type.get(node_id, "Unknown")
        ind = in_degree.get(node_id, 0)
        outd = out_degree.get(node_id, 0)
        # Truncate name if too long
        if len(name) > 32:
            name = name[:29] + "..."
        print(f"  {rank:<5} | {name:<35} | {ntype:<12} | {ind:<4} | {outd:<4} | {degree:<5}")
    print("====================================================")

if __name__ == "__main__":
    main()
