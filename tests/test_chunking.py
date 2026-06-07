import pytest
from src.chunk_documents import detect_section, chunk_document

def test_detect_section():
    """Verify section heading identification logic matches standard patterns."""
    assert detect_section("1 Introduction", "Abstract") == "Introduction"
    assert detect_section("3.2 Related Work", "Methodology") == "Related Work"
    assert detect_section("IV. Proposed Methodology", "Introduction") == "Proposed Methodology"
    assert detect_section("REFERENCES", "Evaluation") == "REFERENCES"
    assert detect_section("conclusion", "Abstract") == "conclusion"
    
    # Case of standard body paragraph (should return current section unchanged)
    body_text = "This is a body paragraph discussing some details of the algorithm discovery process."
    assert detect_section(body_text, "Introduction") == "Introduction"

def test_chunk_document_basic():
    """Verify that a document is chunked properly with section propagation."""
    doc = {
        "arxiv_id": "123.456",
        "title": "A Mock Study",
        "text": "1 Introduction\n\nThis is paragraph one of the introduction section. It is relatively short.\n\nThis is paragraph two of the introduction. It contains details about the framework.\n\n2 Methodology\n\nThis is a methodology paragraph."
    }
    
    chunks = chunk_document(doc, target_size=20, overlap=5)
    assert len(chunks) > 0
    assert chunks[0]["arxiv_id"] == "123.456"
    assert chunks[0]["title"] == "A Mock Study"
    assert chunks[0]["section"] == "Introduction"
    assert chunks[0]["chunk_word_count"] == len(chunks[0]["text"].split())
    assert "paragraph one" in chunks[0]["text"]

def test_chunk_document_large_paragraph_split():
    """Verify that long paragraphs exceeding target size are split into sentences."""
    # Combine many sentences into a single paragraph to exceed the word limit
    doc = {
        "arxiv_id": "123.456",
        "title": "Large Document",
        "text": "Abstract\n\nSentence one of abstract. Sentence two of abstract. Sentence three of abstract. Sentence four of abstract."
    }
    
    # Target size of 5 words (forces sentence splits)
    chunks = chunk_document(doc, target_size=5, overlap=2)
    assert len(chunks) > 1
    # Check that sentences are separated but complete
    assert "Sentence one of abstract." in chunks[0]["text"]
    assert chunks[0]["section"] == "Abstract"

def test_chunk_document_with_pages():
    """Verify chunking when document has pages information."""
    doc = {
        "arxiv_id": "789.012",
        "title": "A Page-indexed Study",
        "pages": [
            {
                "page_num": 1,
                "blocks": [
                    "1 Introduction",
                    "This is first paragraph on page one."
                ]
            },
            {
                "page_num": 2,
                "blocks": [
                    "This is second paragraph on page two.",
                    "2 Methodology",
                    "This is third paragraph on page two."
                ]
            }
        ]
    }
    
    chunks = chunk_document(doc, target_size=10, overlap=2)
    assert len(chunks) > 0
    # First chunk starts on page 1
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["section"] == "Introduction"
    
    # Check page_end propagation
    has_page_2 = False
    for c in chunks:
        assert "page_start" in c
        assert "page_end" in c
        if c["page_start"] == 2 or c["page_end"] == 2:
            has_page_2 = True
            
    assert has_page_2
