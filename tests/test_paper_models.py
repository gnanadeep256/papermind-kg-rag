import pytest
from pydantic import ValidationError
from src.models.paper_models import Paper

def test_paper_model_success():
    """Test standard instantiation of Paper model with valid arguments."""
    data = {
        "paper_id": "2103.00020v1",
        "arxiv_id": "2103.00020",
        "version": 1,
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "authors": ["Alec Radford", "Jong Wook Kim", "Chris Hallacy"],
        "abstract": "CLIP (Contrastive Language-Image Pre-training) is a neural network trained...",
        "published": "2021-03-01T15:00:00Z",
        "updated": "2021-03-02T16:00:00Z",
        "primary_category": "cs.CV",
        "categories": ["cs.CV", "cs.LG"],
        "arxiv_url": "https://arxiv.org/abs/2103.00020v1",
        "pdf_url": "http://arxiv.org/pdf/2103.00020v1"
    }
    
    paper = Paper(**data)
    assert paper.paper_id == "2103.00020v1"
    assert paper.arxiv_id == "2103.00020"
    assert paper.version == 1
    assert paper.title == "Learning Transferable Visual Models From Natural Language Supervision"
    assert len(paper.authors) == 3
    assert paper.primary_category == "cs.CV"
    assert paper.arxiv_url == "https://arxiv.org/abs/2103.00020v1"
    assert paper.pdf_url == "http://arxiv.org/pdf/2103.00020v1"
    assert paper.updated == "2021-03-02T16:00:00Z"

def test_paper_model_missing_fields():
    """Test validation fails when required fields are missing."""
    # Title is missing
    data = {
        "paper_id": "2103.00020v1",
        "arxiv_id": "2103.00020",
        "version": 1,
        "authors": ["Alec Radford"],
        "abstract": "CLIP abstract description.",
        "published": "2021-03-01T15:00:00Z",
        "updated": "2021-03-01T15:00:00Z",
        "primary_category": "cs.CV",
        "categories": ["cs.CV"],
        "arxiv_url": "https://arxiv.org/abs/2103.00020v1"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Paper(**data)
    assert "Field required" in str(exc_info.value)
    assert "title" in str(exc_info.value)

def test_paper_model_invalid_types():
    """Test validation fails when field types are invalid."""
    # Authors list should be list of strings, not a single string
    data = {
        "paper_id": "2103.00020v1",
        "title": "Valid Title",
        "authors": "Alec Radford", # Invalid: should be a List[str]
        "abstract": "Valid abstract summary.",
        "published": "2021-03-01T15:00:00Z",
        "primary_category": "cs.CV",
        "categories": ["cs.CV"]
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Paper(**data)
    assert "Input should be a valid list" in str(exc_info.value)
