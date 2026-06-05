import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase, Driver

# Load environment variables
load_dotenv()

def serialize_node(node) -> Dict[str, Any]:
    """Helper to convert a Neo4j Node object to a standard Python dictionary."""
    if not node:
        return {}
    return {
        "entity_id": node.get("entity_id"),
        "name": node.get("name"),
        "entity_type": list(node.labels)[0] if node.labels else "Unknown",
        **{k: v for k, v in node.items() if k not in ["entity_id", "name"]}
    }

class Neo4jKGRetriever:
    """
    Retrieves nodes, properties, and subgraphs from Neo4j for hybrid KG-RAG search.
    """
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.driver: Driver = None

    def connect(self) -> None:
        """Establishes database connection."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info("KG Retriever successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"KG Retriever failed to connect to Neo4j at {self.uri}: {e}")
            raise

    def close(self) -> None:
        """Closes database connection."""
        if self.driver:
            self.driver.close()
            logger.info("KG Retriever closed Neo4j connection.")

    def _execute_subgraph_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs a Cypher query returning (source_node, relationship, target_node) 
        and extracts a structured {"nodes": [...], "relationships": [...]} subgraph representation.
        """
        start_time = time.time()
        nodes = {}
        relationships = []
        
        try:
            with self.driver.session() as session:
                res = session.run(query, params or {})
                for record in res:
                    s = record.get("source_node")
                    r = record.get("relationship")
                    t = record.get("target_node")
                    
                    if s:
                        s_dict = serialize_node(s)
                        nodes[s_dict["entity_id"]] = s_dict
                    if t:
                        t_dict = serialize_node(t)
                        nodes[t_dict["entity_id"]] = t_dict
                    if r and s and t:
                        rel_dict = {
                            "source": s.get("entity_id"),
                            "target": t.get("entity_id"),
                            "relation": r.type,
                            **{k: v for k, v in r.items()}
                        }
                        relationships.append(rel_dict)
                        
            duration = time.time() - start_time
            logger.debug(f"Subgraph query completed in {duration:.4f}s")
            
        except Exception as e:
            logger.error(f"Error executing subgraph query: {e}")
            # Return empty subgraph on failure rather than crashing
            
        return {
            "nodes": list(nodes.values()),
            "relationships": relationships
        }

    # --- CORE RETRIEVAL UTILITIES ---

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Fetches properties of a Paper node by its arXiv ID."""
        start = time.time()
        query = "MATCH (p:Paper {entity_id: $arxiv_id}) RETURN p"
        try:
            with self.driver.session() as session:
                res = session.run(query, arxiv_id=arxiv_id)
                record = res.single()
                if record:
                    return serialize_node(record["p"])
        except Exception as e:
            logger.error(f"Error fetching paper by arXiv ID: {e}")
        finally:
            logger.debug(f"get_paper_by_arxiv_id completed in {time.time() - start:.4f}s")
        return None

    def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Case-insensitively fetches properties of a Paper node by its title."""
        start = time.time()
        query = "MATCH (p:Paper) WHERE toLower(p.title) = toLower($title) RETURN p"
        try:
            with self.driver.session() as session:
                res = session.run(query, title=title)
                record = res.single()
                if record:
                    return serialize_node(record["p"])
        except Exception as e:
            logger.error(f"Error fetching paper by title: {e}")
        finally:
            logger.debug(f"get_paper_by_title completed in {time.time() - start:.4f}s")
        return None

    def get_authors_of_paper(self, arxiv_id: str) -> List[str]:
        """Gets the list of author names for a given paper."""
        start = time.time()
        query = "MATCH (a:Author)-[:WRITES]->(p:Paper {entity_id: $arxiv_id}) RETURN a.name AS name"
        authors = []
        try:
            with self.driver.session() as session:
                res = session.run(query, arxiv_id=arxiv_id)
                authors = [record["name"] for record in res]
        except Exception as e:
            logger.error(f"Error fetching authors of paper: {e}")
        finally:
            logger.debug(f"get_authors_of_paper completed in {time.time() - start:.4f}s")
        return authors

    def get_methods_for_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Gets methods mentioned or introduced by a paper, including the relationship type."""
        start = time.time()
        query = """
        MATCH (p:Paper {entity_id: $arxiv_id})-[r:MENTIONS|INTRODUCES]->(m:Method)
        RETURN m.name AS name, type(r) AS relation_type
        """
        methods = []
        try:
            with self.driver.session() as session:
                res = session.run(query, arxiv_id=arxiv_id)
                methods = [{"name": record["name"], "relation_type": record["relation_type"]} for record in res]
        except Exception as e:
            logger.error(f"Error fetching methods for paper: {e}")
        finally:
            logger.debug(f"get_methods_for_paper completed in {time.time() - start:.4f}s")
        return methods

    def get_datasets_for_method(self, method_name: str) -> List[str]:
        """Gets datasets evaluated on or used by a method."""
        start = time.time()
        query = "MATCH (m:Method) WHERE toLower(m.name) = toLower($method_name) MATCH (m)-[:EVALUATED_ON|USES]->(d:Dataset) RETURN d.name AS name"
        datasets = []
        try:
            with self.driver.session() as session:
                res = session.run(query, method_name=method_name)
                datasets = [record["name"] for record in res]
        except Exception as e:
            logger.error(f"Error fetching datasets for method: {e}")
        finally:
            logger.debug(f"get_datasets_for_method completed in {time.time() - start:.4f}s")
        return datasets

    def get_tasks_for_method(self, method_name: str) -> List[str]:
        """Gets tasks solved by a method."""
        start = time.time()
        query = "MATCH (m:Method) WHERE toLower(m.name) = toLower($method_name) MATCH (m)-[:SOLVES]->(t:Task) RETURN t.name AS name"
        tasks = []
        try:
            with self.driver.session() as session:
                res = session.run(query, method_name=method_name)
                tasks = [record["name"] for record in res]
        except Exception as e:
            logger.error(f"Error fetching tasks for method: {e}")
        finally:
            logger.debug(f"get_tasks_for_method completed in {time.time() - start:.4f}s")
        return tasks

    def get_papers_about_method(self, method_name: str) -> List[Dict[str, Any]]:
        """Gets papers that mention or introduce a method."""
        start = time.time()
        query = "MATCH (p:Paper)-[r:MENTIONS|INTRODUCES]->(m:Method) WHERE toLower(m.name) = toLower($method_name) RETURN p"
        papers = []
        try:
            with self.driver.session() as session:
                res = session.run(query, method_name=method_name)
                papers = [serialize_node(record["p"]) for record in res]
        except Exception as e:
            logger.error(f"Error fetching papers about method: {e}")
        finally:
            logger.debug(f"get_papers_about_method completed in {time.time() - start:.4f}s")
        return papers

    def get_related_methods(self, method_name: str) -> List[Dict[str, Any]]:
        """Gets other methods connected via COMPARED_WITH or EXTENDS, along with connection type."""
        start = time.time()
        query = """
        MATCH (m:Method) WHERE toLower(m.name) = toLower($method_name)
        MATCH (m)-[r:COMPARED_WITH|EXTENDS]-(o:Method)
        RETURN o.name AS name, type(r) AS relation_type
        """
        methods = []
        try:
            with self.driver.session() as session:
                res = session.run(query, method_name=method_name)
                methods = [{"name": record["name"], "relation_type": record["relation_type"]} for record in res]
        except Exception as e:
            logger.error(f"Error fetching related methods: {e}")
        finally:
            logger.debug(f"get_related_methods completed in {time.time() - start:.4f}s")
        return methods

    def search_entities_by_name(self, query_str: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Performs fuzzy substring search across all entity names."""
        start = time.time()
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($query_str)
        RETURN n
        LIMIT $limit
        """
        entities = []
        try:
            with self.driver.session() as session:
                res = session.run(query, query_str=query_str, limit=limit)
                entities = [serialize_node(record["n"]) for record in res]
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
        finally:
            logger.debug(f"search_entities_by_name completed in {time.time() - start:.4f}s")
        return sorted(entities, key=lambda x: x.get("entity_type", ""))

    # --- TOP ENTITY LISTINGS (Degree Centrality) ---

    def get_top_methods(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches top methods based on total degree centrality."""
        query = """
        MATCH (m:Method)
        RETURN m.name AS name, m.entity_id AS entity_id, COUNT { (m)--() } AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, limit=limit)
                return [dict(record) for record in res]
        except Exception as e:
            logger.error(f"Error fetching top methods: {e}")
            return []

    def get_top_datasets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches top datasets based on total degree centrality."""
        query = """
        MATCH (d:Dataset)
        RETURN d.name AS name, d.entity_id AS entity_id, COUNT { (d)--() } AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, limit=limit)
                return [dict(record) for record in res]
        except Exception as e:
            logger.error(f"Error fetching top datasets: {e}")
            return []

    def get_top_papers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches top papers based on total degree centrality."""
        query = """
        MATCH (p:Paper)
        RETURN p.title AS title, p.entity_id AS arxiv_id, COUNT { (p)--() } AS degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, limit=limit)
                return [dict(record) for record in res]
        except Exception as e:
            logger.error(f"Error fetching top papers: {e}")
            return []

    # --- GRAPH / MULTI-HOP RETRIEVAL ---

    def get_method_context(self, method_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves a 1-hop subgraph around a Method node."""
        query = """
        MATCH (s)-[relationship]->(t)
        WHERE (s:Method AND toLower(s.name) = toLower($method_name))
           OR (t:Method AND toLower(t.name) = toLower($method_name))
        RETURN s AS source_node, relationship, t AS target_node
        """
        return self._execute_subgraph_query(query, {"method_name": method_name})

    def get_paper_subgraph(self, arxiv_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves localized subgraph for a Paper node (paper details, authors, category, methods + method links)."""
        query = """
        MATCH (s:Paper {entity_id: $arxiv_id})-[relationship]->(t)
        RETURN s AS source_node, relationship, t AS target_node
        UNION
        MATCH (p:Paper {entity_id: $arxiv_id})-[r1:MENTIONS|INTRODUCES]->(m:Method)
        MATCH (m)-[relationship]->(t)
        RETURN m AS source_node, relationship, t AS target_node
        UNION
        MATCH (p:Paper {entity_id: $arxiv_id})-[r1:MENTIONS|INTRODUCES]->(m:Method)
        MATCH (s)-[relationship]->(m)
        RETURN s AS source_node, relationship, m AS target_node
        """
        return self._execute_subgraph_query(query, {"arxiv_id": arxiv_id})

    def get_entity_neighborhood(self, entity_id: str, hops: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves all nodes and relationships within a specified hop count from the starting node."""
        if hops not in [1, 2, 3]:
            raise ValueError("Hops count must be 1, 2, or 3.")
            
        query = f"""
        MATCH path = (n {{entity_id: $entity_id}})-[*1..{hops}]-(m)
        UNWIND relationships(path) AS r
        RETURN startNode(r) AS source_node, r AS relationship, endNode(r) AS target_node
        """
        subgraph = self._execute_subgraph_query(query, {"entity_id": entity_id})
        
        # Ensure start node itself is included in the subgraph even if it has no matches
        start_node_data = self.get_paper_by_arxiv_id(entity_id)
        if not start_node_data:
            # Try general search
            try:
                with self.driver.session() as session:
                    res = session.run("MATCH (n {entity_id: $entity_id}) RETURN n", entity_id=entity_id)
                    rec = res.single()
                    if rec:
                        start_node_data = serialize_node(rec["n"])
            except Exception:
                pass
                
        if start_node_data:
            # Check if start node is already in nodes list, if not add it
            node_ids = {n["entity_id"] for n in subgraph["nodes"]}
            if start_node_data["entity_id"] not in node_ids:
                subgraph["nodes"].append(start_node_data)
                
        return subgraph

    def find_connection(self, source_entity: str, target_entity: str) -> Dict[str, List[Dict[str, Any]]]:
        """Finds the shortest connectivity path between two entity nodes."""
        query = """
        MATCH path = shortestPath((s {entity_id: $source_entity})-[*1..10]-(t {entity_id: $target_entity}))
        UNWIND relationships(path) AS r
        RETURN startNode(r) AS source_node, r AS relationship, endNode(r) AS target_node
        """
        return self._execute_subgraph_query(query, {
            "source_entity": source_entity,
            "target_entity": target_entity
        })
