from pydantic import BaseModel, Field
from typing import List, Optional

class Paper(BaseModel):
    """
    Pydantic data model representing a parsed research paper.
    Contains field validations and type annotations.
    """
    paper_id: str = Field(..., description="Unique arXiv ID, including version")
    arxiv_id: str = Field(..., description="arXiv ID without version")
    version: int = Field(..., description="Version of the paper")
    title: str = Field(..., description="Cleaned title of the paper")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Cleaned abstract text")
    published: str = Field(..., description="Publication date string (ISO 8601 format)")
    updated: str = Field(..., description="Last updated date string (ISO 8601 format)")
    primary_category: str = Field(..., description="Primary arXiv subject category")
    categories: List[str] = Field(..., description="List of all subject categories")
    arxiv_url: str = Field(..., description="URL pointing to the paper landing page")
    pdf_url: Optional[str] = Field(None, description="URL pointing to the paper PDF")
