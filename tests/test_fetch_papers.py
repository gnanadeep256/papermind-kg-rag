import pytest
from src.fetch_papers import build_query, parse_feed, normalize_papers

def test_build_query():
    """Test building query URLs and request parameter generation."""
    categories = ["cs.AI", "cs.CL"]
    max_results = 15
    
    url, params = build_query(categories, max_results)
    assert url == "http://export.arxiv.org/api/query"
    assert params["search_query"] == "cat:cs.AI OR cat:cs.CL"
    assert params["max_results"] == 15
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"

def test_parse_feed_success():
    """Test XML parser correctly extracts fields from standard arXiv Atom format."""
    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2103.00020v1</id>
        <title>Learning Transferable Visual Models  \n From Natural Language Supervision</title>
        <summary>CLIP  abstract  description \n text.</summary>
        <published>2021-03-01T15:00:00Z</published>
        <updated>2021-03-02T16:00:00Z</updated>
        <author>
          <name>Alec Radford</name>
        </author>
        <author>
          <name>Jong Wook Kim</name>
        </author>
        <arxiv:primary_category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
        <link href="http://arxiv.org/pdf/2103.00020v1" rel="related" type="application/pdf" title="pdf"/>
      </entry>
    </feed>
    """
    
    parsed = parse_feed(mock_xml)
    assert len(parsed) == 1
    
    record = parsed[0]
    assert record["paper_id"] == "2103.00020v1"
    assert record["arxiv_id"] == "2103.00020"
    assert record["version"] == 1
    assert record["title"] == "Learning Transferable Visual Models From Natural Language Supervision"
    assert record["abstract"] == "CLIP abstract description text."
    assert record["published"] == "2021-03-01T15:00:00Z"
    assert record["updated"] == "2021-03-02T16:00:00Z"
    assert record["authors"] == ["Alec Radford", "Jong Wook Kim"]
    assert record["primary_category"] == "cs.CV"
    assert record["categories"] == ["cs.CV", "cs.LG"]
    assert record["arxiv_url"] == "https://arxiv.org/abs/2103.00020v1"
    assert record["pdf_url"] == "http://arxiv.org/pdf/2103.00020v1"

def test_parse_feed_empty():
    """Test XML parser behavior with empty feeds."""
    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    </feed>
    """
    parsed = parse_feed(mock_xml)
    assert len(parsed) == 0

def test_normalize_papers_success():
    """Test conversion of validated raw dicts into Paper instances."""
    parsed_records = [{
        "paper_id": "2103.00020v1",
        "arxiv_id": "2103.00020",
        "version": 1,
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "authors": ["Alec Radford", "Jong Wook Kim"],
        "abstract": "CLIP description.",
        "published": "2021-03-01T15:00:00Z",
        "updated": "2021-03-01T15:00:00Z",
        "primary_category": "cs.CV",
        "categories": ["cs.CV", "cs.LG"],
        "arxiv_url": "https://arxiv.org/abs/2103.00020v1",
        "pdf_url": "http://arxiv.org/pdf/2103.00020v1"
    }]
    
    normalized = normalize_papers(parsed_records)
    assert len(normalized) == 1
    assert normalized[0].paper_id == "2103.00020v1"
    assert normalized[0].arxiv_id == "2103.00020"
    assert normalized[0].version == 1

def test_normalize_papers_validation_skip():
    """Test invalid paper formats are skipped during normalization."""
    parsed_records = [
        {
            "paper_id": "2103.00020v1",
            "arxiv_id": "2103.00020",
            "version": 1,
            "title": "Valid Title",
            "authors": ["Alec Radford"],
            "abstract": "Valid abstract.",
            "published": "2021-03-01T15:00:00Z",
            "updated": "2021-03-01T15:00:00Z",
            "primary_category": "cs.CV",
            "categories": ["cs.CV"],
            "arxiv_url": "https://arxiv.org/abs/2103.00020v1",
            "pdf_url": "http://arxiv.org/pdf/2103.00020v1"
        },
        {
            "paper_id": "invalid_paper",
            "arxiv_id": "invalid_paper",
            "version": 1,
            "title": "",
            "authors": "invalid_string_instead_of_list", # Will fail validation
            "abstract": "Abstract",
            "published": "2021-03-01T15:00:00Z",
            "updated": "2021-03-01T15:00:00Z",
            "primary_category": "cs.CV",
            "categories": ["cs.CV"],
            "arxiv_url": "https://arxiv.org/abs/invalid_paper"
        }
    ]
    
    normalized = normalize_papers(parsed_records)
    # The second invalid record should be skipped silently
    assert len(normalized) == 1
    assert normalized[0].paper_id == "2103.00020v1"
