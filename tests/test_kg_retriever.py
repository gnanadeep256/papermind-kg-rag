import pytest
from unittest.mock import MagicMock, patch
from src.kg_retriever import Neo4jKGRetriever, serialize_node

class MockNode:
    def __init__(self, labels, entity_id, name, **properties):
        self.labels = set(labels)
        self._properties = {"entity_id": entity_id, "name": name, **properties}
    def get(self, key, default=None):
        return self._properties.get(key, default)
    def items(self):
        return self._properties.items()
    def __getitem__(self, key):
        return self._properties[key]

class MockRelationship:
    def __init__(self, rel_type, **properties):
        self.type = rel_type
        self._properties = properties
    def get(self, key, default=None):
        return self._properties.get(key, default)
    def items(self):
        return self._properties.items()
    def __getitem__(self, key):
        return self._properties[key]

def test_serialize_node():
    """Verify node serialization handles missing labels and extra keys correctly."""
    node = MockNode(["Paper"], "123", "Test Paper", published="2026")
    res = serialize_node(node)
    assert res["entity_id"] == "123"
    assert res["name"] == "Test Paper"
    assert res["entity_type"] == "Paper"
    assert res["published"] == "2026"

    # Edge case: no labels
    empty_labels_node = MockNode([], "456", "No Label")
    res_empty = serialize_node(empty_labels_node)
    assert res_empty["entity_type"] == "Unknown"

def test_retriever_connection_lifecycle():
    """Verify retriever driver instantiation and verify_connectivity/close calls."""
    with patch("src.kg_retriever.GraphDatabase") as mock_graph_db:
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        retriever = Neo4jKGRetriever()
        retriever.connect()

        mock_graph_db.driver.assert_called_once_with(retriever.uri, auth=(retriever.username, retriever.password))
        mock_driver.verify_connectivity.assert_called_once()

        retriever.close()
        mock_driver.close.assert_called_once()

def test_get_paper_by_arxiv_id():
    """Test get_paper_by_arxiv_id retrieves and serializes a single Paper node."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_record = {"p": MockNode(["Paper"], "arxiv123", "Title 123", title="Title 123")}
    mock_result = MagicMock()
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    paper = retriever.get_paper_by_arxiv_id("arxiv123")
    assert paper is not None
    assert paper["entity_id"] == "arxiv123"
    assert paper["title"] == "Title 123"

    mock_session.run.assert_called_once_with("MATCH (p:Paper {entity_id: $arxiv_id}) RETURN p", arxiv_id="arxiv123")

def test_get_paper_by_arxiv_id_not_found():
    """Verify get_paper_by_arxiv_id returns None when no matches exist."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_result = MagicMock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result

    paper = retriever.get_paper_by_arxiv_id("nonexistent")
    assert paper is None

def test_get_paper_by_title():
    """Verify get_paper_by_title constructs query with correct parameters."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_record = {"p": MockNode(["Paper"], "arxiv123", "Title 123", title="Title 123")}
    mock_result = MagicMock()
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    paper = retriever.get_paper_by_title("Title 123")
    assert paper is not None
    assert paper["entity_id"] == "arxiv123"
    assert "toLower(p.title) = toLower($title)" in mock_session.run.call_args[0][0]

def test_get_authors_of_paper():
    """Verify author names retrieval returns a clean list."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [{"name": "Author A"}, {"name": "Author B"}]
    mock_session.run.return_value = mock_records

    authors = retriever.get_authors_of_paper("arxiv123")
    assert authors == ["Author A", "Author B"]

def test_get_methods_for_paper():
    """Verify methods and relation types are correctly compiled."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [
        {"name": "Method A", "relation_type": "INTRODUCES"},
        {"name": "Method B", "relation_type": "MENTIONS"}
    ]
    mock_session.run.return_value = mock_records

    methods = retriever.get_methods_for_paper("arxiv123")
    assert len(methods) == 2
    assert methods[0]["name"] == "Method A"
    assert methods[0]["relation_type"] == "INTRODUCES"

def test_get_datasets_for_method():
    """Verify dataset name query executes successfully."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [{"name": "Dataset A"}, {"name": "Dataset B"}]
    mock_session.run.return_value = mock_records

    datasets = retriever.get_datasets_for_method("Method X")
    assert datasets == ["Dataset A", "Dataset B"]

def test_get_tasks_for_method():
    """Verify task query executes successfully."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [{"name": "Task A"}]
    mock_session.run.return_value = mock_records

    tasks = retriever.get_tasks_for_method("Method X")
    assert tasks == ["Task A"]

def test_get_papers_about_method():
    """Verify papers list query constructs properly."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [{"p": MockNode(["Paper"], "arxiv123", "Title X")}]
    mock_session.run.return_value = mock_records

    papers = retriever.get_papers_about_method("Method X")
    assert len(papers) == 1
    assert papers[0]["entity_id"] == "arxiv123"

def test_get_related_methods():
    """Verify related methods connection query executes successfully."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [{"name": "Method Y", "relation_type": "EXTENDS"}]
    mock_session.run.return_value = mock_records

    related = retriever.get_related_methods("Method X")
    assert len(related) == 1
    assert related[0]["name"] == "Method Y"
    assert related[0]["relation_type"] == "EXTENDS"

def test_search_entities_by_name():
    """Verify fuzzy query returns type-sorted entity results."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_records = [
        {"n": MockNode(["Method"], "m1", "LoRA-T")},
        {"n": MockNode(["Paper"], "p1", "Title LoRA")}
    ]
    mock_session.run.return_value = mock_records

    results = retriever.search_entities_by_name("lora")
    assert len(results) == 2
    # Sorted by entity type: Method, then Paper
    assert results[0]["entity_type"] == "Method"
    assert results[1]["entity_type"] == "Paper"

def test_get_top_entities():
    """Verify top degree centrality queries return list of dicts."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # 1. Top methods
    mock_session.run.return_value = [{"name": "Method A", "entity_id": "ma", "degree": 12}]
    top_m = retriever.get_top_methods(limit=1)
    assert len(top_m) == 1
    assert top_m[0]["name"] == "Method A"
    assert top_m[0]["degree"] == 12

    # 2. Top datasets
    mock_session.run.return_value = [{"name": "Dataset A", "entity_id": "da", "degree": 8}]
    top_d = retriever.get_top_datasets(limit=1)
    assert len(top_d) == 1
    assert top_d[0]["name"] == "Dataset A"
    assert top_d[0]["degree"] == 8

    # 3. Top papers
    mock_session.run.return_value = [{"title": "Paper A", "arxiv_id": "pa", "degree": 25}]
    top_p = retriever.get_top_papers(limit=1)
    assert len(top_p) == 1
    assert top_p[0]["title"] == "Paper A"
    assert top_p[0]["degree"] == 25

def test_get_method_context():
    """Verify 1-hop context query formats nodes and relationships properly."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    s_node = MockNode(["Method"], "method_lora", "LoRA")
    t_node = MockNode(["Paper"], "paper_lora", "LoRA paper")
    rel = MockRelationship("INTRODUCES", confidence=1.0)

    mock_records = [{"source_node": s_node, "relationship": rel, "target_node": t_node}]
    mock_session.run.return_value = mock_records

    subgraph = retriever.get_method_context("LoRA")
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["relationships"]) == 1
    assert subgraph["relationships"][0]["source"] == "method_lora"
    assert subgraph["relationships"][0]["target"] == "paper_lora"
    assert subgraph["relationships"][0]["relation"] == "INTRODUCES"
    assert subgraph["relationships"][0]["confidence"] == 1.0

def test_get_paper_subgraph():
    """Verify paper subgraph retrieves localization correctly."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    s_node = MockNode(["Paper"], "p1", "Paper 1")
    t_node = MockNode(["Method"], "m1", "Method 1")
    rel = MockRelationship("INTRODUCES")

    mock_records = [{"source_node": s_node, "relationship": rel, "target_node": t_node}]
    mock_session.run.return_value = mock_records

    subgraph = retriever.get_paper_subgraph("p1")
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["relationships"]) == 1

def test_get_entity_neighborhood():
    """Verify neighborhood query enforces hop count validation and appends start node."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Invalid hops count
    with pytest.raises(ValueError):
        retriever.get_entity_neighborhood("e1", hops=4)

    # Valid hops
    s_node = MockNode(["Method"], "e1", "Start Node")
    t_node = MockNode(["Dataset"], "d1", "Dataset 1")
    rel = MockRelationship("USES")

    mock_records = [{"source_node": s_node, "relationship": rel, "target_node": t_node}]
    mock_session.run.return_value = mock_records

    # Also mock get_paper_by_arxiv_id returning None and general search returning start node
    # to test fallback start node verification logic
    mock_result_gen = MagicMock()
    mock_result_gen.single.return_value = {"n": s_node}
    
    # Simple side_effect mapping for session run calls
    def side_effect_run(cypher_query, *args, **kwargs):
        if "MATCH path = " in cypher_query:
            mock_res = MagicMock()
            mock_res.__iter__.return_value = iter(mock_records)
            return mock_res
        elif "MATCH (n {entity_id: $entity_id}) RETURN n" in cypher_query:
            return mock_result_gen
        elif "MATCH (p:Paper {entity_id: $arxiv_id}) RETURN p" in cypher_query:
            mock_res = MagicMock()
            mock_res.single.return_value = None
            return mock_res
        return MagicMock()

    mock_session.run.side_effect = side_effect_run

    subgraph = retriever.get_entity_neighborhood("e1", hops=2)
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["relationships"]) == 1

def test_find_connection():
    """Verify connection shortest path query matches correct format."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    s_node = MockNode(["Paper"], "p1", "Paper 1")
    t_node = MockNode(["Method"], "m1", "Method 1")
    rel = MockRelationship("INTRODUCES")

    mock_records = [{"source_node": s_node, "relationship": rel, "target_node": t_node}]
    mock_session.run.return_value = mock_records

    subgraph = retriever.find_connection("p1", "m1")
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["relationships"]) == 1
    assert "shortestPath" in mock_session.run.call_args[0][0]

def test_retriever_query_error_handling():
    """Verify queries handle driver exception/errors safely returning defaults."""
    retriever = Neo4jKGRetriever()
    mock_driver = MagicMock()
    retriever.driver = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("Neo4j internal connection error")

    # Core retrieval returns None/empty list/subgraph dict on exception
    assert retriever.get_paper_by_arxiv_id("any") is None
    assert retriever.get_authors_of_paper("any") == []
    assert retriever.get_methods_for_paper("any") == []
    assert retriever.get_top_methods() == []
    
    subgraph = retriever.get_method_context("any")
    assert subgraph == {"nodes": [], "relationships": []}
