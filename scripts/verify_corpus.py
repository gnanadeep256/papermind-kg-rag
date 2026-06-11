import os
import sys
import json
import time
from collections import Counter

def main():
    chunks_path = "data/vectorstore/chunk_metadata.json"
    graph_path = "data/processed/graph_data.json"
    papers_path = "data/raw/papers.json"
    report_path = "corpus_validation.md"
    
    print("====================================================")
    print("        PAPERMIND CORPUS INTEGRITY VERIFIER         ")
    print("====================================================")
    
    validation_passed = True
    errors = []
    warnings = []
    
    # Check files exist
    for p in [chunks_path, graph_path, papers_path]:
        if not os.path.exists(p):
            print(f"Error: Required file not found: {p}")
            sys.exit(1)
            
    # Load files
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    with open(papers_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)
        
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])
    papers = papers_data.get("papers", [])
    
    # 1. Verify no duplicate chunk IDs
    chunk_ids = [c["chunk_id"] for c in chunks if "chunk_id" in c]
    unique_chunk_ids = set(chunk_ids)
    if len(chunk_ids) != len(unique_chunk_ids):
        dup_chunks = [cid for cid, count in Counter(chunk_ids).items() if count > 1]
        errors.append(f"Duplicate Chunk IDs found: {len(chunk_ids) - len(unique_chunk_ids)} duplicates. Sample: {dup_chunks[:5]}")
        validation_passed = False
    else:
        print("  - Duplicate Chunk IDs check: PASS")
        
    # 2. Verify no duplicate papers in metadata
    paper_ids = [p["arxiv_id"] for p in papers if "arxiv_id" in p]
    unique_paper_ids = set(paper_ids)
    if len(paper_ids) != len(unique_paper_ids):
        warnings.append(f"Duplicate papers in raw metadata papers.json: {len(paper_ids) - len(unique_paper_ids)} duplicates.")
    else:
        print("  - Duplicate raw papers check: PASS")
        
    # 3. Verify every chunk maps to a valid paper
    graph_paper_ids = {e["entity_id"] for e in entities if e["entity_type"] == "Paper"}
    missing_papers_for_chunks = []
    for chunk in chunks:
        aid = chunk.get("arxiv_id")
        if not aid:
            errors.append(f"Chunk {chunk.get('chunk_id')} has missing or null arxiv_id field.")
            validation_passed = False
        elif aid not in graph_paper_ids:
            missing_papers_for_chunks.append((chunk.get("chunk_id"), aid))
            
    if missing_papers_for_chunks:
        errors.append(f"{len(missing_papers_for_chunks)} chunks map to papers not present in graph data. Sample: {missing_papers_for_chunks[:5]}")
        validation_passed = False
    else:
        print("  - Chunk-to-Paper mapping check: PASS")
        
    # 4. Verify every paper in graph has chunks
    chunk_paper_ids = {c.get("arxiv_id") for c in chunks if c.get("arxiv_id")}
    missing_chunks_for_papers = []
    for pid in graph_paper_ids:
        # Since not all papers are chunked (we have 100 chunked, but only 50 in Neo4j, wait!
        # In Phase 5C, we expand Neo4j to contain all papers, but in the previous phase only 50 were in Neo4j.
        # Now, we are expanding, so all papers should have chunks.)
        if pid not in chunk_paper_ids:
            # Check if this paper was downloaded and processed
            missing_chunks_for_papers.append(pid)
            
    if missing_chunks_for_papers:
        warnings.append(f"Graph papers missing chunk representation: {len(missing_chunks_for_papers)} papers. IDs: {missing_chunks_for_papers[:5]}")
    else:
        print("  - Paper-to-Chunk reverse mapping check: PASS")
        
    # 5. Verify no orphan graph edges (endpoints must exist in entities)
    entity_ids = {e["entity_id"] for e in entities}
    dangling_edges = []
    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        if src not in entity_ids or tgt not in entity_ids:
            dangling_edges.append((src, rel.get("relation"), tgt))
            
    if dangling_edges:
        errors.append(f"Orphan graph edges found (missing endpoints): {len(dangling_edges)} edges. Sample: {dangling_edges[:5]}")
        validation_passed = False
    else:
        print("  - Orphan graph edges check: PASS")
        
    # 6. Verify no orphan graph nodes (degree == 0, excluding Category/Author if they are leaf nodes)
    # Let's count connections
    connected_nodes = set()
    for rel in relationships:
        connected_nodes.add(rel.get("source"))
        connected_nodes.add(rel.get("target"))
        
    orphan_nodes = []
    for ent in entities:
        eid = ent["entity_id"]
        etype = ent["entity_type"]
        if eid not in connected_nodes:
            orphan_nodes.append((eid, etype))
            
    if orphan_nodes:
        # Filter orphan methods or papers which are more critical
        critical_orphans = [(eid, etype) for eid, etype in orphan_nodes if etype in ["Paper", "Method", "Dataset"]]
        if critical_orphans:
            warnings.append(f"Orphan core graph nodes found (degree=0): {len(critical_orphans)} nodes. Sample: {critical_orphans[:5]}")
        else:
            print("  - Orphan graph nodes check: PASS (non-critical leaf nodes only)")
    else:
        print("  - Orphan graph nodes check: PASS")
        
    status_str = "PASS" if validation_passed else "FAIL"
    print(f"\nVerification status: {status_str}")
    print(f"Total Errors: {len(errors)} | Total Warnings: {len(warnings)}")
    
    # Write report
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("# PaperMind Corpus Integrity Validation Report\n\n")
        out.write(f"- **Validation Status**: {status_str}\n")
        out.write(f"- **Checked At**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        out.write(f"- **Total Papers Checked**: {len(graph_paper_ids)}\n")
        out.write(f"- **Total Chunks Checked**: {len(chunks)}\n")
        out.write(f"- **Total Entities**: {len(entities)}\n")
        out.write(f"- **Total Relationships**: {len(relationships)}\n\n")
        
        out.write("## Checks Summary\n\n")
        out.write(f"| Check Name | Status | Details |\n")
        out.write(f"|---|---|---|\n")
        out.write(f"| Duplicate Chunk IDs | {'PASS' if not any('Duplicate Chunk IDs' in e for e in errors) else 'FAIL'} | Checks for redundant chunk primary keys |\n")
        out.write(f"| Chunk-to-Paper Mapping | {'PASS' if not any('chunks map to papers' in e for e in errors) else 'FAIL'} | Verifies that all chunks belong to registered papers |\n")
        out.write(f"| Dangling Graph Edges | {'PASS' if not dangling_edges else 'FAIL'} | Verifies that all relationship endpoints exist in the entity cache |\n")
        out.write(f"| Orphan Core Nodes | {'PASS' if not orphan_nodes else 'WARNING'} | Verifies that all papers, methods, and datasets have connections |\n")
        out.write(f"| Raw Paper Duplicates | {'PASS' if not any('raw metadata' in w for w in warnings) else 'WARNING'} | Verifies that papers.json contains unique records |\n\n")
        
        if errors:
            out.write("## Integrity Errors\n\n")
            for err in errors:
                out.write(f"- **[ERROR]** {err}\n")
            out.write("\n")
            
        if warnings:
            out.write("## Integrity Warnings\n\n")
            for warn in warnings:
                out.write(f"- **[WARNING]** {warn}\n")
            out.write("\n")
            
        out.write("---\n")
        out.write("Report generated by the automated verification suite.\n")
        
    print(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    main()
