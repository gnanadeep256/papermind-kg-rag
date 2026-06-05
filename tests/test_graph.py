import pytest
from src.extract_knowledge import normalize_name, is_introduced_method, merge_entity, merge_relationship
from src.models.graph_models import Entity, Relationship, GroqEntity, GroqRelationship, GroqPayload

def test_normalize_name():
    """Test standard normalization of entity names (lower, strip, collapse spaces)."""
    assert normalize_name("  Graph Neural Network  ") == "graph neural network"
    assert normalize_name("graph   neural   network") == "graph neural network"
    assert normalize_name("BERT") == "bert"

def test_is_introduced_method():
    """Test that is_introduced_method correctly flags proposed methods based on titles."""
    # Substring match cases
    assert is_introduced_method("HANDOFF", "HANDOFF: Humanoid Agentic Task-Space Whole-Body Control")
    assert is_introduced_method("Code2LoRA", "Code2LoRA: Hypernetwork-Generated Adapters for Code LMs")
    assert is_introduced_method("TempoVLA", "TempoVLA: Learning Speed-Controllable Policies")
    
    # Acronym match cases
    assert is_introduced_method("YOIO", "You Only Index Once: Cross-Layer Sparse Attention")
    assert is_introduced_method("PWP", "Polynomial Weight Preconditioning: PC Layer for Pre-Training")
    
    # Non-matching cases (existing algorithms or unrelated names)
    assert not is_introduced_method("BERT", "You Only Index Once: Cross-Layer Sparse Attention")
    assert not is_introduced_method("LoRA", "HANDOFF: Humanoid Agentic Task-Space Whole-Body Control")

def test_merge_entity_new():
    """Test inserting a new entity into the cache."""
    global_entities = {}
    merge_entity(
        global_entities=global_entities,
        entity_id="bert",
        name="BERT",
        entity_type="Method",
        description="A language model.",
        confidence=0.9,
        paper_id="2103.00020v1"
    )
    
    key = ("bert", "Method")
    assert key in global_entities
    entity = global_entities[key]
    assert entity.entity_id == "bert"
    assert entity.name == "BERT"
    assert entity.entity_type == "Method"
    assert entity.description == "A language model."
    assert entity.confidence == 0.9
    assert entity.source_papers == ["2103.00020v1"]

def test_merge_entity_duplicate_longer_description():
    """Test duplicate insertion keeps the longest description, takes max confidence, and unions paper IDs."""
    global_entities = {}
    
    # First insert
    merge_entity(
        global_entities=global_entities,
        entity_id="bert",
        name="BERT",
        entity_type="Method",
        description="A language model.",
        confidence=0.8,
        paper_id="2103.00020v1"
    )
    
    # Duplicate insert with a longer description, higher confidence, and different paper ID
    merge_entity(
        global_entities=global_entities,
        entity_id="bert",
        name="bert",
        entity_type="Method",
        description="A pre-trained bidirectional transformer language representation model.",
        confidence=0.95,
        paper_id="2606.06467v1"
    )
    
    key = ("bert", "Method")
    entity = global_entities[key]
    # Original case is preserved from the first insertion
    assert entity.name == "BERT"
    # Longest description is retained
    assert entity.description == "A pre-trained bidirectional transformer language representation model."
    # Max confidence is taken
    assert entity.confidence == 0.95
    # Source papers are unioned
    assert set(entity.source_papers) == {"2103.00020v1", "2606.06467v1"}

def test_merge_relationship_success():
    """Test inserting and merging relationships correctly."""
    global_relationships = {}
    
    # First insert
    merge_relationship(
        global_relationships=global_relationships,
        source="bert",
        target="glue",
        relation="EVALUATED_ON",
        description="Evaluated on GLUE benchmark.",
        confidence=0.8,
        paper_id="2103.00020v1"
    )
    
    # Duplicate insert
    merge_relationship(
        global_relationships=global_relationships,
        source="bert",
        target="glue",
        relation="EVALUATED_ON",
        description="BERT was evaluated on GLUE task suite with good results.",
        confidence=0.9,
        paper_id="2606.06467v1"
    )
    
    key = ("bert", "glue", "EVALUATED_ON")
    assert key in global_relationships
    rel = global_relationships[key]
    assert rel.source == "bert"
    assert rel.target == "glue"
    assert rel.relation == "EVALUATED_ON"
    assert rel.description == "BERT was evaluated on GLUE task suite with good results."
    assert rel.confidence == 0.9
    assert set(rel.source_papers) == {"2103.00020v1", "2606.06467v1"}

def test_extraction_payload_validation():
    """Test that GroqEntity and GroqRelationship validate schemas correctly."""
    entity_data = {
        "name": "Attention",
        "entity_type": "Concept",
        "description": "Mechanism focusing on specific parts of input.",
        "confidence": 0.98
    }
    
    entity = GroqEntity(**entity_data)
    assert entity.name == "Attention"
    assert entity.confidence == 0.98
    
    rel_data = {
        "source": "Transformer",
        "target": "Attention",
        "relation": "USES",
        "description": "Transformer architecture uses self-attention.",
        "confidence": 0.95
    }
    
    rel = GroqRelationship(**rel_data)
    assert rel.relation == "USES"
    assert rel.confidence == 0.95
    
    payload = GroqPayload(entities=[entity], relationships=[rel])
    assert len(payload.entities) == 1
    assert len(payload.relationships) == 1
