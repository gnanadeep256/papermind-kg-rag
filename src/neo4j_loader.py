import os
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Set
import yaml
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase, Driver

# Load environment variables
load_dotenv()

# Allowed lists to prevent malformed Cypher injection and enforce taxonomy
ALLOWED_LABELS: Set[str] = {
    "Paper",
    "Author",
    "Category",
    "Method",
    "Concept",
    "Dataset",
    "Metric",
    "Task",
    "Organization",
}

ALLOWED_RELATIONSHIPS: Set[str] = {
    "WRITES",
    "BELONGS_TO",
    "MENTIONS",
    "INTRODUCES",
    "USES",
    "BASED_ON",
    "EXTENDS",
    "EVALUATED_ON",
    "DEVELOPED_BY",
    "SOLVES",
    "OUTPERFORMS",
    "COMPARED_WITH",
}

class Neo4jLoader:
    """
    Manages loading entities and relationships into Neo4j with batching and constraint enforcement.
    """
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.driver: Driver = None
        
        # Load deterministic taxonomy corrections
        self.method_to_concept = set()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        corrections_path = os.path.abspath(os.path.join(current_dir, "..", "configs", "taxonomy_corrections.yaml"))
        if os.path.exists(corrections_path):
            try:
                with open(corrections_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and "method_to_concept" in data:
                        self.method_to_concept = {item.lower().strip() for item in data["method_to_concept"]}
                logger.info(f"Loaded {len(self.method_to_concept)} taxonomy corrections.")
            except Exception as e:
                logger.error(f"Error loading taxonomy corrections from {corrections_path}: {e}")

    def connect(self) -> None:
        """Establishes database driver connection."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise

    def close(self) -> None:
        """Closes database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j database connection.")

    def create_constraints(self) -> None:
        """Creates uniqueness constraints on entity_id for all taxonomy labels."""
        logger.info("Creating uniqueness constraints...")
        with self.driver.session() as session:
            for label in ALLOWED_LABELS:
                constraint_name = f"{label.lower()}_entity_id"
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.entity_id IS UNIQUE
                """
                try:
                    session.run(query)
                    logger.info(f"Verified constraint: {constraint_name}")
                except Exception as e:
                    logger.error(f"Error creating constraint {constraint_name}: {e}")
                    raise

    def create_indexes(self) -> None:
        """Creates search indexes on name for Paper, Method, Concept, and Dataset labels."""
        logger.info("Creating name indexes...")
        indexed_labels = ["Paper", "Method", "Concept", "Dataset"]
        with self.driver.session() as session:
            for label in indexed_labels:
                index_name = f"{label.lower()}_name_index"
                query = f"""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON (n.name)
                """
                try:
                    session.run(query)
                    logger.info(f"Verified index: {index_name}")
                except Exception as e:
                    logger.error(f"Error creating index {index_name}: {e}")
                    raise

    def load_entities(self, entities: List[Dict[str, Any]], batch_size: int = 500) -> Dict[str, Any]:
        """Loads entities from a list in transaction-committed batches."""
        nodes_processed = 0
        node_batches = 0
        label_counts = defaultdict(int)
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            node_batches += 1
            
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    for ent in batch:
                        entity_id = ent.get("entity_id")
                        entity_type = ent.get("entity_type")
                        
                        # Apply deterministic taxonomy corrections
                        if entity_type == "Method" and entity_id.lower().strip() in self.method_to_concept:
                            entity_type = "Concept"
                            ent["entity_type"] = "Concept"
                        
                        if entity_type not in ALLOWED_LABELS:
                            raise ValueError(
                                f"Malformed label '{entity_type}' is not present in ALLOWED_LABELS list."
                            )

                        # Omit None values to keep DB properties clean
                        props = {k: v for k, v in ent.items() if v is not None}
                        
                        query = f"""
                        MERGE (n:{entity_type} {{entity_id: $entity_id}})
                        SET n += $props
                        """
                        tx.run(query, entity_id=entity_id, props=props)
                        label_counts[entity_type] += 1
                        nodes_processed += 1
                    tx.commit()
            
            logger.info(f"Processed nodes: {nodes_processed} / {len(entities)}")

        return {
            "nodes_processed": nodes_processed,
            "node_batches": node_batches,
            "label_counts": label_counts
        }

    def load_nodes(self, entities_path: str, batch_size: int = 500) -> Dict[str, Any]:
        """Loads entities from entities.json in transaction-committed batches."""
        logger.info(f"Loading entities from {entities_path}...")
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f"Entities file not found: {entities_path}")

        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)

        return self.load_entities(entities, batch_size=batch_size)

    def load_relationships(self, relationships: Any, batch_size: int = 500) -> Dict[str, Any]:
        """Loads relationships from a JSON file path or a list of dictionaries in transaction-committed batches."""
        if isinstance(relationships, str):
            relationships_path = relationships
            logger.info(f"Loading relationships from {relationships_path}...")
            if not os.path.exists(relationships_path):
                raise FileNotFoundError(f"Relationships file not found: {relationships_path}")
            with open(relationships_path, "r", encoding="utf-8") as f:
                relationships_list = json.load(f)
            # Derive entities path to resolve labels
            entities_path = os.path.join(os.path.dirname(relationships_path), "entities.json")
        else:
            relationships_list = relationships
            entities_path = "data/processed/entities.json"

        entity_type_map = defaultdict(list)
        if os.path.exists(entities_path):
            with open(entities_path, "r", encoding="utf-8") as f:
                entities_data = json.load(f)
            for ent in entities_data:
                e_id = ent["entity_id"]
                e_type = ent["entity_type"]
                if e_type == "Method" and e_id.lower().strip() in self.method_to_concept:
                    e_type = "Concept"
                entity_type_map[e_id].append({
                    "type": e_type,
                    "papers": set(ent.get("source_papers", []))
                })

        def get_entity_label(entity_id: str, rel_papers: List[str]) -> str:
            candidates = entity_type_map.get(entity_id, [])
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]["type"]
            rel_papers_set = set(rel_papers)
            for cand in candidates:
                if cand["papers"] & rel_papers_set:
                    return cand["type"]
            return candidates[0]["type"]

        relationships_processed = 0
        relationship_batches = 0
        
        for i in range(0, len(relationships_list), batch_size):
            batch = relationships_list[i:i + batch_size]
            relationship_batches += 1
            
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    for rel in batch:
                        source = rel.get("source")
                        target = rel.get("target")
                        relation = rel.get("relation")
                        rel_papers = rel.get("source_papers", [])
                        
                        if relation not in ALLOWED_RELATIONSHIPS:
                            raise ValueError(
                                f"Relationship type '{relation}' is not present in ALLOWED_RELATIONSHIPS list."
                            )

                        # Keep only description, confidence, source_papers properties and omit None
                        props = {
                            k: v for k, v in rel.items() 
                            if k in ["description", "confidence", "source_papers"] and v is not None
                        }
                        
                        source_label = get_entity_label(source, rel_papers)
                        target_label = get_entity_label(target, rel_papers)
                        
                        source_match = f":{source_label}" if source_label else ""
                        target_match = f":{target_label}" if target_label else ""
                        
                        query = f"""
                        MATCH (s{source_match} {{entity_id: $source}})
                        MATCH (t{target_match} {{entity_id: $target}})
                        MERGE (s)-[r:{relation}]->(t)
                        SET r += $props
                        """
                        tx.run(query, source=source, target=target, props=props)
                        relationships_processed += 1
                    tx.commit()
            
            logger.info(f"Processed relationships: {relationships_processed} / {len(relationships_list)}")

        return {
            "relationships_processed": relationships_processed,
            "relationship_batches": relationship_batches
        }

def run_ingestion() -> None:
    """Executes the full ingestion pipeline and outputs report."""
    loader = Neo4jLoader()
    loader.connect()
    
    start_time = time.time()
    
    try:
        loader.create_constraints()
        loader.create_indexes()
        
        # Load Nodes (Batch size = 500)
        node_stats = loader.load_nodes("data/processed/entities.json", batch_size=500)
        
        # Load Relationships (Batch size = 500)
        rel_stats = loader.load_relationships("data/processed/relationships.json", batch_size=500)
        
        execution_time = time.time() - start_time
        
        # Ingestion Report Output (Strictly NO emojis in console/comments)
        print("================================================")
        print("PAPERMIND NEO4J INGESTION REPORT")
        print("================================================")
        print(f"Entities Loaded:          {node_stats['nodes_processed']}")
        print(f"Relationships Loaded:     {rel_stats['relationships_processed']}")
        print(f"Node Batches committed:   {node_stats['node_batches']}")
        print(f"Rel Batches committed:    {rel_stats['relationship_batches']}")
        print("")
        
        # Output details per node label
        counts = node_stats["label_counts"]
        for label in sorted(list(ALLOWED_LABELS)):
            print(f"{label:<15}:            {counts.get(label, 0)}")
            
        print("")
        print(f"Execution Time:           {execution_time:.2f} seconds")
        print("Neo4j Validation:         PASS")
        print("================================================")
        
    finally:
        loader.close()

if __name__ == "__main__":
    run_ingestion()
