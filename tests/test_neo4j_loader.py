import json
import pytest
from unittest.mock import MagicMock, patch
from src.neo4j_loader import Neo4jLoader, ALLOWED_LABELS, ALLOWED_RELATIONSHIPS

def test_loader_connect():
    """Verify GraphDatabase.driver instantiation and close lifecycle calls."""
    with patch("src.neo4j_loader.GraphDatabase") as mock_graph_db:
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        loader = Neo4jLoader()
        loader.connect()
        
        mock_graph_db.driver.assert_called_once_with(loader.uri, auth=(loader.username, loader.password))
        mock_driver.verify_connectivity.assert_called_once()
        
        loader.close()
        mock_driver.close.assert_called_once()

def test_create_constraints():
    """Verify that a constraint is created for each label in ALLOWED_LABELS."""
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    loader.create_constraints()
    
    assert mock_session.run.call_count == len(ALLOWED_LABELS)
    # Check that CREATE CONSTRAINT statement is executed
    called_queries = [call[0][0] for call in mock_session.run.call_args_list]
    for label in ALLOWED_LABELS:
        assert any(f"CREATE CONSTRAINT {label.lower()}_entity_id" in q for q in called_queries)

def test_load_nodes_success(tmp_path):
    """Verify standard node loading behaves correctly with batch commits and SET properties."""
    entities_data = [
        {
            "entity_id": "paper1",
            "name": "Title One",
            "entity_type": "Paper",
            "confidence": 1.0,
            "title": "Title One"
        },
        {
            "entity_id": "author1",
            "name": "Author One",
            "entity_type": "Author",
            "confidence": 1.0
        },
        {
            "entity_id": "method1",
            "name": "Method One",
            "entity_type": "Method",
            "confidence": 0.95
        }
    ]
    entities_file = tmp_path / "entities.json"
    entities_file.write_text(json.dumps(entities_data))
    
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    # Load with batch size 2 (should result in 2 commit blocks)
    stats = loader.load_nodes(str(entities_file), batch_size=2)
    
    assert stats["nodes_processed"] == 3
    assert stats["node_batches"] == 2
    assert stats["label_counts"]["Paper"] == 1
    assert stats["label_counts"]["Author"] == 1
    assert stats["label_counts"]["Method"] == 1
    
    assert mock_tx.commit.call_count == 2
    assert mock_tx.run.call_count == 3
    
    # Check query structures
    run_args = [call[0] for call in mock_tx.run.call_args_list]
    assert "MERGE (n:Paper {entity_id: $entity_id})" in run_args[0][0]
    assert "MERGE (n:Author {entity_id: $entity_id})" in run_args[1][0]
    assert "MERGE (n:Method {entity_id: $entity_id})" in run_args[2][0]

def test_load_nodes_invalid_label(tmp_path):
    """Verify that loader aborts and raises ValueError when encountering unmapped labels."""
    entities_data = [
        {
            "entity_id": "paper1",
            "name": "Title One",
            "entity_type": "InvalidType"
        }
    ]
    entities_file = tmp_path / "entities.json"
    entities_file.write_text(json.dumps(entities_data))
    
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    with pytest.raises(ValueError) as excinfo:
        loader.load_nodes(str(entities_file))
    assert "InvalidType" in str(excinfo.value)
    # Transaction should not be committed on failure
    mock_tx.commit.assert_not_called()

def test_load_relationships_success(tmp_path):
    """Verify standard relationship loader uses MATCH and MERGE correctly with properties."""
    relationships_data = [
        {
            "source": "paper1",
            "target": "author1",
            "relation": "DEVELOPED_BY",
            "confidence": 0.98
        },
        {
            "source": "paper1",
            "target": "method1",
            "relation": "INTRODUCES",
            "confidence": 0.95
        }
    ]
    relationships_file = tmp_path / "relationships.json"
    relationships_file.write_text(json.dumps(relationships_data))
    
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    # Load with batch size 1 (should result in 2 commit blocks)
    stats = loader.load_relationships(str(relationships_file), batch_size=1)
    
    assert stats["relationships_processed"] == 2
    assert stats["relationship_batches"] == 2
    
    assert mock_tx.commit.call_count == 2
    assert mock_tx.run.call_count == 2
    
    # Check Cypher structure
    run_args = [call[0] for call in mock_tx.run.call_args_list]
    assert "MATCH (s {entity_id: $source})" in run_args[0][0]
    assert "MERGE (s)-[r:DEVELOPED_BY]->(t)" in run_args[0][0]
    assert "MERGE (s)-[r:INTRODUCES]->(t)" in run_args[1][0]

def test_load_relationships_invalid_type(tmp_path):
    """Verify loader aborts and raises ValueError when encountering unmapped relationship types."""
    relationships_data = [
        {
            "source": "paper1",
            "target": "author1",
            "relation": "INVALID_RELATION",
            "confidence": 0.98
        }
    ]
    relationships_file = tmp_path / "relationships.json"
    relationships_file.write_text(json.dumps(relationships_data))
    
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    with pytest.raises(ValueError) as excinfo:
        loader.load_relationships(str(relationships_file))
    assert "INVALID_RELATION" in str(excinfo.value)
    mock_tx.commit.assert_not_called()

def test_create_indexes():
    """Verify that name search indexes are created for Paper, Method, Concept, and Dataset."""
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    loader.create_indexes()
    
    assert mock_session.run.call_count == 4
    called_queries = [call[0][0] for call in mock_session.run.call_args_list]
    for label in ["Paper", "Method", "Concept", "Dataset"]:
        assert any(f"CREATE INDEX {label.lower()}_name_index" in q for q in called_queries)

def test_load_entities_list():
    """Verify that load_entities accepts direct lists and merges nodes correctly."""
    entities_data = [
        {
            "entity_id": "paper1",
            "name": "Title One",
            "entity_type": "Paper",
            "confidence": 1.0
        }
    ]
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    stats = loader.load_entities(entities_data)
    assert stats["nodes_processed"] == 1
    assert stats["node_batches"] == 1
    assert stats["label_counts"]["Paper"] == 1
    assert mock_tx.commit.call_count == 1
    assert mock_tx.run.call_count == 1

def test_load_relationships_list():
    """Verify that load_relationships accepts direct lists and merges relationships correctly."""
    relationships_data = [
        {
            "source": "paper1",
            "target": "author1",
            "relation": "DEVELOPED_BY",
            "confidence": 0.98
        }
    ]
    loader = Neo4jLoader()
    mock_driver = MagicMock()
    loader.driver = mock_driver
    
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
    
    stats = loader.load_relationships(relationships_data)
    assert stats["relationships_processed"] == 1
    assert stats["relationship_batches"] == 1
    assert mock_tx.commit.call_count == 1
    assert mock_tx.run.call_count == 1
