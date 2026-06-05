import os
import json
from collections import defaultdict

def main():
    graph_path = "data/processed/graph_data.json"
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found. Please run the extraction pipeline first.")
        return

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    print("====================================================")
    # No emojis in output, keeping design sleek
    print("         PAPERMIND DATA INTEGRITY AUDIT REPORT      ")
    print("====================================================")

    # 1. Entity Integrity Check
    print("1. ENTITY INTEGRITY CHECK:")
    # Group entities by type to check for duplicates
    entities_by_type = defaultdict(list)
    for ent in entities:
        entities_by_type[ent.get("entity_type")].append(ent)

    entity_integrity_passed = True
    for etype, type_entities in entities_by_type.items():
        seen_ids = set()
        seen_case_insensitive = set()
        duplicates = []
        for ent in type_entities:
            eid = ent.get("entity_id")
            normalized = eid.lower()
            if eid in seen_ids or normalized in seen_case_insensitive:
                duplicates.append(eid)
            seen_ids.add(eid)
            seen_case_insensitive.add(normalized)

        if duplicates:
            print(f"  [FAIL] Entity type '{etype}' has duplicate IDs: {duplicates}")
            entity_integrity_passed = False
        else:
            print(f"  [PASS] Entity type '{etype}': all {len(type_entities)} IDs are unique.")

    if entity_integrity_passed:
        print("  * Overall Entity Integrity: PASSED")
    else:
        print("  * Overall Entity Integrity: FAILED")
    print("----------------------------------------------------")

    # 2. Relationship Integrity Check
    print("2. RELATIONSHIP INTEGRITY CHECK:")
    entity_ids = {ent.get("entity_id") for ent in entities}
    missing_endpoints = []
    
    for idx, rel in enumerate(relationships, 1):
        source = rel.get("source")
        target = rel.get("target")
        missing_source = source not in entity_ids
        missing_target = target not in entity_ids
        
        if missing_source or missing_target:
            missing_endpoints.append({
                "index": idx,
                "relation": rel.get("relation"),
                "source": source,
                "target": target,
                "missing_source": missing_source,
                "missing_target": missing_target
            })

    if missing_endpoints:
        print(f"  [FAIL] Found {len(missing_endpoints)} relationships with missing endpoints:")
        for item in missing_endpoints[:10]:
            detail = []
            if item["missing_source"]:
                detail.append(f"source '{item['source']}' missing")
            if item["missing_target"]:
                detail.append(f"target '{item['target']}' missing")
            print(f"    - Rel #{item['index']} ({item['relation']}): {', '.join(detail)}")
        if len(missing_endpoints) > 10:
            print(f"    ... and {len(missing_endpoints) - 10} more.")
    else:
        print(f"  [PASS] All {len(relationships)} relationships have valid source and target endpoints.")
    print("----------------------------------------------------")

    # 3. Orphan Nodes Check (degree == 0)
    print("3. ORPHAN NODES CHECK:")
    orphan_nodes = []
    for ent in entities:
        in_deg = ent.get("in_degree", 0)
        out_deg = ent.get("out_degree", 0)
        total_deg = in_deg + out_deg
        if total_deg == 0:
            orphan_nodes.append(ent)

    if orphan_nodes:
        print(f"  [INFO] Found {len(orphan_nodes)} orphan nodes (degree == 0):")
        for ent in orphan_nodes[:10]:
            print(f"    - '{ent.get('name')}' ({ent.get('entity_type')})")
        if len(orphan_nodes) > 10:
            print(f"    ... and {len(orphan_nodes) - 10} more.")
    else:
        print("  [PASS] No orphan nodes found. All nodes have at least one connection.")
    print("----------------------------------------------------")

    # 4. Duplicate Paper Titles Check
    print("4. DUPLICATE PAPER TITLES CHECK:")
    papers = [ent for ent in entities if ent.get("entity_type") == "Paper"]
    
    paper_ids = [p.get("entity_id") for p in papers]
    paper_titles = [p.get("title") for p in papers if p.get("title")]

    duplicate_paper_ids = [pid for pid in set(paper_ids) if paper_ids.count(pid) > 1]
    duplicate_paper_titles = [title for title in set(paper_titles) if paper_titles.count(title) > 1]

    paper_check_passed = True
    if duplicate_paper_ids:
        print(f"  [FAIL] Duplicate Paper entity IDs found: {duplicate_paper_ids}")
        paper_check_passed = False
    else:
        print("  [PASS] All Paper entity IDs are unique.")

    if duplicate_paper_titles:
        print(f"  [FAIL] Duplicate Paper titles found: {duplicate_paper_titles}")
        paper_check_passed = False
    else:
        print("  [PASS] All Paper titles are unique.")

    if paper_check_passed:
        print(f"  * Paper Node Integrity: PASSED (Processed count: {len(papers)})")
    else:
        print("  * Paper Node Integrity: FAILED")
    print("====================================================")

if __name__ == "__main__":
    main()
