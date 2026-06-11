import os
import json
import re

def main():
    readme_path = "README.md"
    stats_path = "data/processed/graph_stats.json"
    chunks_path = "data/vectorstore/chunk_metadata.json"
    papers_path = "data/raw/papers.json"
    
    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}")
        return
        
    total_papers = 0
    total_chunks = 0
    total_nodes = 0
    total_edges = 0
    
    # Load actual statistics if they exist
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_papers = len(data.get("papers", []))
        except Exception as e:
            print(f"Warning: Could not read papers.json: {e}")
            
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_chunks = len(data)
        except Exception as e:
            print(f"Warning: Could not read chunk_metadata.json: {e}")
            
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata = data.get("metadata", {})
                total_nodes = metadata.get("total_nodes", 0)
                total_edges = metadata.get("total_edges", 0)
        except Exception as e:
            print(f"Warning: Could not read graph_stats.json: {e}")
            
    print(f"Loaded statistics:")
    print(f"  - Total Papers: {total_papers}")
    print(f"  - Total Chunks: {total_chunks}")
    print(f"  - Graph Nodes : {total_nodes}")
    print(f"  - Graph Edges : {total_edges}")
    
    # Read README.md
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check if markers exist
    start_marker = "<!-- START_STATS -->"
    end_marker = "<!-- END_STATS -->"
    
    stats_block = f"""{start_marker}
### Knowledge Base Statistics
* **Total Papers**: {total_papers} (Foundational + Recent research)
* **Total Chunks**: {total_chunks} paragraph/sentence-aware blocks
* **Total Graph Nodes**: {total_nodes} entities (Methods, Datasets, Authors, Concepts, etc.)
* **Total Graph Edges**: {total_edges} semantic relationships
* **Total Benchmarks**: 200 evaluation queries
* **Total Evaluation Queries**: 200 gold-standard test cases
{end_marker}"""

    if start_marker in content and end_marker in content:
        # Replace existing stats block
        pattern = re.compile(rf"{start_marker}.*?{end_marker}", re.DOTALL)
        updated_content = pattern.sub(stats_block, content)
        print("Markers found. Replacing stats block...")
    else:
        # Append stats block to project overview section or at the top
        print("Markers not found. Injecting stats block after the overview header...")
        overview_header = "## Project Overview"
        if overview_header in content:
            updated_content = content.replace(
                overview_header, 
                f"{overview_header}\n\n{stats_block}"
            )
        else:
            updated_content = content + "\n\n" + stats_block
            
    # Save README.md
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
        
    print("README.md updated successfully.")

if __name__ == "__main__":
    main()
