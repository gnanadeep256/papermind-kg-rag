import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Terms that are typically Concepts rather than distinct Methods
GENERIC_BLACKLIST = {
    "llm", "large language model", "large language models",
    "transformer", "bert", "gpt", "lora", "rag",
    "reinforcement learning", "rl", "in-context learner", "in-context learning",
    "deep learning", "machine learning", "neural network", "neural networks"
}

def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    degree_threshold = 5

    print("====================================================")
    print("         KNOWLEDGE GRAPH TAXONOMY AUDIT             ")
    print("====================================================")
    print(f"Connection URI:     {uri}")
    print(f"Degree Threshold:   {degree_threshold}")
    print("----------------------------------------------------")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return

    try:
        with driver.session() as session:
            # Query all Method nodes with their degree and name
            query = """
            MATCH (m:Method)
            RETURN m.name AS name, m.entity_id AS entity_id, COUNT { (m)--() } AS degree
            ORDER BY degree DESC
            """
            res = session.run(query)
            records = list(res)

            flagged_blacklist = []
            flagged_high_degree = []

            for r in records:
                name = r["name"]
                entity_id = r["entity_id"]
                degree = r["degree"]
                norm_name = name.lower().strip()

                is_in_blacklist = norm_name in GENERIC_BLACKLIST or entity_id.lower().strip() in GENERIC_BLACKLIST
                
                # Flag category 1: Matches generic blacklist
                if is_in_blacklist:
                    flagged_blacklist.append((name, entity_id, degree))
                
                # Flag category 2: Highly connected (degree > threshold) but not in blacklist
                elif degree > degree_threshold:
                    flagged_high_degree.append((name, entity_id, degree))

            # 1. Output Generic Blacklist matches
            print("FLAGGED: METHODS MATCHING GENERIC BLACKLIST (Should be 'Concept')")
            print(f"  {'Method Name':<45} | {'Entity ID':<25} | {'Degree':<6}")
            print("  " + "-" * 82)
            if flagged_blacklist:
                for name, ent_id, deg in flagged_blacklist:
                    print(f"  {name[:45]:<45} | {ent_id[:25]:<25} | {deg:<6}")
            else:
                print("  (No blacklist matches found)")
            
            print("\n----------------------------------------------------")
            
            # 2. Output High Degree Methods for inspection
            print(f"FLAGGED: HIGH DEGREE METHODS (Degree > {degree_threshold}, check if Concept)")
            print(f"  {'Method Name':<45} | {'Entity ID':<25} | {'Degree':<6}")
            print("  " + "-" * 82)
            if flagged_high_degree:
                for name, ent_id, deg in flagged_high_degree:
                    print(f"  {name[:45]:<45} | {ent_id[:25]:<25} | {deg:<6}")
            else:
                print(f"  (No methods with degree > {degree_threshold} found)")

            print("====================================================")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
