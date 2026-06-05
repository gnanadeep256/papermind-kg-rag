import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    entities_path = "data/processed/entities.json"
    relationships_path = "data/processed/relationships.json"

    if not os.path.exists(entities_path) or not os.path.exists(relationships_path):
        print("Error: processed data files not found. Please run extraction pipeline first.")
        return

    with open(entities_path, "r", encoding="utf-8") as f:
        expected_nodes = len(json.load(f))
    with open(relationships_path, "r", encoding="utf-8") as f:
        expected_relationships = len(json.load(f))

    print("Connecting to Neo4j to run validation queries...")
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return

    try:
        with driver.session() as session:
            # 1. Node Count
            res_nodes = session.run("MATCH (n) RETURN count(n) AS cnt")
            actual_nodes = res_nodes.single()["cnt"]

            # 2. Relationship Count
            res_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            actual_relationships = res_rels.single()["cnt"]

            # 3. Paper Count
            res_papers = session.run("MATCH (p:Paper) RETURN count(p) AS cnt")
            actual_papers = res_papers.single()["cnt"]

            # 4. Method Count
            res_methods = session.run("MATCH (m:Method) RETURN count(m) AS cnt")
            actual_methods = res_methods.single()["cnt"]

            print("\n================================================")
            print("         NEO4J GRAPH VALIDATION SUMMARY         ")
            print("================================================")
            print(f"Nodes in DB:             {actual_nodes} (Expected: {expected_nodes})")
            print(f"Relationships in DB:     {actual_relationships} (Expected: {expected_relationships})")
            print(f"Paper nodes:             {actual_papers}")
            print(f"Method nodes:            {actual_methods}")
            print("------------------------------------------------")

            # 5. Introduced Methods Sample (LIMIT 25)
            print("SAMPLE: INTRODUCED METHODS (Paper -> INTRODUCES -> Method):")
            res_intro = session.run("""
                MATCH (p:Paper)-[:INTRODUCES]->(m:Method)
                RETURN p.name AS paper_name, m.name AS method_name
                LIMIT 25
            """)
            intro_records = list(res_intro)
            if intro_records:
                for r in intro_records:
                    p_name = r["paper_name"]
                    m_name = r["method_name"]
                    if len(p_name) > 30:
                        p_name = p_name[:27] + "..."
                    print(f"  - {p_name:<30} -> INTRODUCES -> {m_name}")
            else:
                print("  (No INTRODUCES relationships found)")
            print("------------------------------------------------")

            # 6. Author Writes Paper Sample (LIMIT 20)
            print("SAMPLE: AUTHOR WRITES PAPER (Author -> WRITES -> Paper):")
            res_writes = session.run("""
                MATCH (a:Author)-[:WRITES]->(p:Paper)
                RETURN a.name AS author_name, p.name AS paper_name
                LIMIT 20
            """)
            writes_records = list(res_writes)
            if writes_records:
                for r in writes_records:
                    a_name = r["author_name"]
                    p_name = r["paper_name"]
                    if len(p_name) > 30:
                        p_name = p_name[:27] + "..."
                    print(f"  - {a_name:<20} -> WRITES -> {p_name}")
            else:
                print("  (No WRITES relationships found)")
            print("------------------------------------------------")

            # 7. Top Connected Methods (LIMIT 20)
            print("TOP 20 CONNECTED METHODS BY DEGREE:")
            res_deg = session.run("""
                MATCH (m:Method)
                RETURN m.name AS method_name, COUNT { (m)--() } AS degree
                ORDER BY degree DESC
                LIMIT 20
            """)
            deg_records = list(res_deg)
            if deg_records:
                print(f"  {'Rank':<5} | {'Method Name':<45} | {'Degree':<6}")
                print("  " + "-" * 60)
                for rank, r in enumerate(deg_records, 1):
                    m_name = r["method_name"]
                    if len(m_name) > 42:
                        m_name = m_name[:39] + "..."
                    print(f"  {rank:<5} | {m_name:<45} | {r['degree']:<6}")
            else:
                print("  (No Method nodes found in DB)")
            print("================================================")

            # Validation result determination
            nodes_match = actual_nodes == expected_nodes
            rels_match = actual_relationships == expected_relationships
            if nodes_match and rels_match:
                print("Graph Validation Result: PASS")
            else:
                print("Graph Validation Result: FAIL (Count mismatch detected)")
            print("================================================\n")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
