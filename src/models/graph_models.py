from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    """
    Represents a unified graph entity (node) in the final processed dataset.
    """
    entity_id: str = Field(..., description="Unique identifier for the entity (lowercase normalized name).")
    name: str = Field(..., description="Cleaned and standardized name of the entity.")
    entity_type: str = Field(..., description="Entity taxonomy label (Paper, Author, Category, Method, Concept, Dataset, Metric, Task, Organization).")
    description: Optional[str] = Field(None, description="Detailed explanation of the entity.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0 (1.0 for deterministic nodes).")
    in_degree: Optional[int] = Field(0, description="Number of incoming edges.")
    out_degree: Optional[int] = Field(0, description="Number of outgoing edges.")
    source_papers: List[str] = Field(..., description="List of paper IDs where this entity was found.")
    
    # Optional fields for Paper node metadata enrichment
    title: Optional[str] = Field(None, description="Title of the paper (only for Paper nodes).")
    published: Optional[str] = Field(None, description="ISO publication date of the paper (only for Paper nodes).")
    primary_category: Optional[str] = Field(None, description="Primary category of the paper (only for Paper nodes).")
    arxiv_url: Optional[str] = Field(None, description="ArXiv page URL of the paper (only for Paper nodes).")
    pdf_url: Optional[str] = Field(None, description="ArXiv PDF document URL (only for Paper nodes).")

class Relationship(BaseModel):
    """
    Represents a directed relationship (edge) between two entities in the final processed dataset.
    """
    source: str = Field(..., description="The entity_id of the source node.")
    target: str = Field(..., description="The entity_id of the target node.")
    relation: str = Field(..., description="The type of the relationship (e.g. WRITES, BELONGS_TO, MENTIONS, USES, BASED_ON, EXTENDS, INTRODUCES, EVALUATED_ON, DEVELOPED_BY, SOLVES, OUTPERFORMS, COMPARED_WITH).")
    description: Optional[str] = Field(None, description="Explanation of the relationship link.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0 (1.0 for deterministic edges).")
    source_papers: List[str] = Field(..., description="List of paper IDs that substantiate this relationship.")

# Schemas for Groq Structured Output
class GroqEntity(BaseModel):
    name: str = Field(..., description="Standardized name of the entity (e.g. Transformer, BERT, Attention).")
    entity_type: str = Field(..., description="Must be exactly one of: Method, Concept, Dataset, Metric, Task, Organization.")
    description: str = Field(..., description="Short explanation of its role in this abstract.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")

class GroqRelationship(BaseModel):
    source: str = Field(..., description="Standardized name of the source entity.")
    target: str = Field(..., description="Standardized name of the target entity.")
    relation: str = Field(..., description="Relationship verb (USES, BASED_ON, EXTENDS, INTRODUCES, EVALUATED_ON, DEVELOPED_BY, SOLVES, OUTPERFORMS, COMPARED_WITH).")
    description: str = Field(..., description="Explanation of how the source and target are linked in the paper.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")

class GroqPayload(BaseModel):
    entities: List[GroqEntity] = Field(..., description="List of extracted entities.")
    relationships: List[GroqRelationship] = Field(..., description="List of relationships connecting these entities.")
