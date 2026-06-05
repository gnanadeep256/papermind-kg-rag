import os
import re
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Optional

from dotenv import load_dotenv

from src.utils.config import load_config, ConfigError
from src.utils.logger import logger
from src.llm import query_groq_json
from src.models.graph_models import Entity, Relationship, GroqPayload
from pydantic import ValidationError

# Load environment variables
load_dotenv()

# Allowed Taxonomies
ALLOWED_ENTITY_TYPES = {"Method", "Concept", "Dataset", "Metric", "Task", "Organization"}
ALLOWED_RELATIONS = {
    "USES", "BASED_ON", "EXTENDS", "INTRODUCES", "EVALUATED_ON",
    "DEVELOPED_BY", "SOLVES", "OUTPERFORMS", "COMPARED_WITH"
}

# Quality thresholds
MIN_ENTITY_CONFIDENCE = 0.85
MIN_RELATIONSHIP_CONFIDENCE = 0.85
ALLOWED_METHOD_INTRODUCTIONS = 1

def normalize_name(name: str) -> str:
    """
    Standardizes names for strict deduplication.
    Trims whitespace, collapses multiple spaces, and lowercases.

    Args:
        name: Raw name string.

    Returns:
        Normalized lowercase name string.
    """
    cleaned = name.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()

def clean_display_name(name: str) -> str:
    """
    Cleans up duplicate spacing in names while preserving casing.

    Args:
        name: Raw name string.

    Returns:
        Cleaned name preserving casing.
    """
    return re.sub(r"\s+", " ", name.strip())

def is_introduced_method(method_name: str, title: str) -> bool:
    """
    Checks if a method is likely introduced by the paper based on its title.
    Returns True if the method name is a distinct token/word in the title (case-insensitive),
    or if the method name is an acronym of the title words.
    This prevents substring leakage (e.g. "lora" matching "code2lora").

    Args:
        method_name: Name of the method.
        title: Title of the paper.

    Returns:
        True if the method is likely proposed by the paper, False otherwise.
    """
    norm_method = normalize_name(method_name)
    norm_title = normalize_name(title)
    
    # 1. Stricter token match: check if method name exists as a complete word/token in the title
    tokens = re.findall(r"\b[a-zA-Z0-9-]+\b", norm_title)
    if norm_method in tokens:
        return True
        
    # 2. Acronym of all words in the title
    words = [w for w in re.split(r"\W+", norm_title) if w]
    acronym_all = "".join(w[0] for w in words if w)
    if norm_method == acronym_all:
        return True
        
    # 3. Acronym of title before the colon (if colon exists)
    if ":" in title:
        prefix = title.split(":")[0].strip().lower()
        prefix_words = [w for w in re.split(r"\W+", prefix) if w]
        acronym_prefix = "".join(w[0] for w in prefix_words if w)
        if norm_method == acronym_prefix:
            return True
            
        # Acronym of title after the colon
        suffix = title.split(":")[1].strip().lower()
        suffix_words = [w for w in re.split(r"\W+", suffix) if w]
        acronym_suffix = "".join(w[0] for w in suffix_words if w)
        if norm_method == acronym_suffix:
            return True
            
    return False

def entity_exists(global_entities: Dict[Tuple[str, str], Entity], entity_id: str) -> bool:
    """
    Checks if an entity with the given normalized entity_id exists in the cache.

    Args:
        global_entities: Cache dictionary.
        entity_id: Target entity ID.

    Returns:
        True if it exists, False otherwise.
    """
    for key in global_entities.keys():
        if key[0] == entity_id:
            return True
    return False

def merge_entity(
    global_entities: Dict[Tuple[str, str], Entity],
    entity_id: str,
    name: str,
    entity_type: str,
    description: Optional[str],
    confidence: Optional[float],
    paper_id: str,
    title: Optional[str] = None,
    published: Optional[str] = None,
    primary_category: Optional[str] = None,
    arxiv_url: Optional[str] = None,
    pdf_url: Optional[str] = None
) -> None:
    """
    Adds or merges an entity into the global entity cache.
    If the entity exists (matching normalized ID and type), it keeps the longest description.

    Args:
        global_entities: Global cache dictionary keyed by (entity_id, entity_type).
        entity_id: Primary key of the entity (normalized name).
        name: Clean display name of the entity.
        entity_type: Category label.
        description: Textual description.
        confidence: Extraction confidence score (0.0 to 1.0).
        paper_id: Versioned arXiv ID of the source paper.
        title: Optional title of the paper.
        published: Optional published date of the paper.
        primary_category: Optional primary category of the paper.
        arxiv_url: Optional arXiv web page URL.
        pdf_url: Optional arXiv PDF document URL.
    """
    key = (entity_id, entity_type)
    desc = description if description else ""
    conf = confidence if confidence is not None else 1.0
    
    if key in global_entities:
        existing = global_entities[key]
        existing_desc = existing.description if existing.description else ""
        if len(desc) > len(existing_desc):
            existing.description = desc
            
        if paper_id not in existing.source_papers:
            existing.source_papers.append(paper_id)
            
        if existing.confidence is not None:
            existing.confidence = max(existing.confidence, conf)
            
        # Update optional metadata fields if they are provided and not already set
        if title and not existing.title:
            existing.title = title
        if published and not existing.published:
            existing.published = published
        if primary_category and not existing.primary_category:
            existing.primary_category = primary_category
        if arxiv_url and not existing.arxiv_url:
            existing.arxiv_url = arxiv_url
        if pdf_url and not existing.pdf_url:
            existing.pdf_url = pdf_url
    else:
        global_entities[key] = Entity(
            entity_id=entity_id,
            name=clean_display_name(name),
            entity_type=entity_type,
            description=desc if desc else None,
            confidence=conf,
            source_papers=[paper_id],
            title=title,
            published=published,
            primary_category=primary_category,
            arxiv_url=arxiv_url,
            pdf_url=pdf_url
        )

def merge_relationship(
    global_relationships: Dict[Tuple[str, str, str], Relationship],
    source: str,
    target: str,
    relation: str,
    description: Optional[str],
    confidence: Optional[float],
    paper_id: str
) -> None:
    """
    Adds or merges a relationship into the global relationship cache.

    Args:
        global_relationships: Global cache dictionary keyed by (normalized_source, normalized_target, relation).
        source: Entity ID of the source.
        target: Entity ID of the target.
        relation: Relationship verb.
        description: Explanation string.
        confidence: Confidence score (0.0 to 1.0).
        paper_id: Versioned arXiv ID of the source paper.
    """
    norm_source = normalize_name(source)
    norm_target = normalize_name(target)
    key = (norm_source, norm_target, relation)
    desc = description if description else ""
    conf = confidence if confidence is not None else 1.0
    
    if key in global_relationships:
        existing = global_relationships[key]
        existing_desc = existing.description if existing.description else ""
        if len(desc) > len(existing_desc):
            existing.description = desc
            
        if paper_id not in existing.source_papers:
            existing.source_papers.append(paper_id)
            
        if existing.confidence is not None:
            existing.confidence = max(existing.confidence, conf)
    else:
        global_relationships[key] = Relationship(
            source=norm_source,
            target=norm_target,
            relation=relation,
            description=desc if desc else None,
            confidence=conf,
            source_papers=[paper_id]
        )

def extract_knowledge(abstract: str, prompt_template: str, model_name: str) -> GroqPayload:
    """
    Queries the Groq API to extract entities and relationships.
    Uses structured Pydantic schema validation output.

    Args:
        abstract: Text of the paper abstract.
        prompt_template: Guidelines and instructions loaded from extraction_prompt.txt.
        model_name: Name of the Groq model to use.

    Returns:
        Validated GroqPayload object.
    """
    prompt = f"{prompt_template}\n\nAbstract to analyze:\n{abstract}"
    raw_response = query_groq_json(prompt, model_name=model_name, temperature=0.0)
    payload = GroqPayload.model_validate_json(raw_response)
    return payload

def extract_knowledge_with_retry(
    abstract: str,
    prompt_template: str,
    model_name: str,
    retries: int = 3,
    initial_delay: float = 4.0
) -> GroqPayload:
    """
    Queries Groq API with retry logic and exponential backoff.

    Args:
        abstract: Text of the paper abstract.
        prompt_template: Extraction prompt guidelines.
        model_name: Name of the Groq model to use.
        retries: Number of retry attempts.
        initial_delay: Starting delay in seconds.

    Returns:
        GroqPayload containing entities and relationships.
    """
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            return extract_knowledge(abstract, prompt_template, model_name)
        except Exception as e:
            logger.warning(f"Groq API attempt {attempt} failed: {e}")
            if attempt == retries:
                logger.error("Max retries exceeded for Groq API call")
                raise
            time.sleep(delay)
            delay *= 2.0
            
    raise RuntimeError("Failed to extract knowledge after multiple retries")

def save_current_state(
    processed_dir: str,
    global_entities: Dict[Tuple[str, str], Entity],
    global_relationships: Dict[Tuple[str, str, str], Relationship],
    processed_paper_ids: Set[str],
    failed_count: int
) -> None:
    """
    Saves the cached entities and relationships to data/processed/.

    Args:
        processed_dir: Folder to write the processed output.
        global_entities: Local cache of entities.
        global_relationships: Local cache of relationships.
        processed_paper_ids: Set of processed paper IDs.
        failed_count: Total failures recorded.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Calculate degree statistics
    in_degrees = {}
    out_degrees = {}
    for rel in global_relationships.values():
        out_degrees[rel.source] = out_degrees.get(rel.source, 0) + 1
        in_degrees[rel.target] = in_degrees.get(rel.target, 0) + 1

    entities_list = []
    for ent in global_entities.values():
        norm_id = normalize_name(ent.entity_id)
        ent.in_degree = in_degrees.get(norm_id, 0)
        ent.out_degree = out_degrees.get(norm_id, 0)
        entities_list.append(ent.model_dump())
        
    relationships_list = [rel.model_dump() for rel in global_relationships.values()]
    
    # 1. Save entities.json
    entities_path = os.path.join(processed_dir, "entities.json")
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities_list, f, indent=2, ensure_ascii=False)
        
    # 2. Save relationships.json
    relationships_path = os.path.join(processed_dir, "relationships.json")
    with open(relationships_path, "w", encoding="utf-8") as f:
        json.dump(relationships_list, f, indent=2, ensure_ascii=False)
        
    # 3. Save graph_data.json
    graph_data_path = os.path.join(processed_dir, "graph_data.json")
    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "llm_provider": "groq",
            "papers_processed": len(processed_paper_ids),
            "papers_failed": failed_count,
            "total_entities": len(entities_list),
            "total_relationships": len(relationships_list),
            "processed_papers": list(processed_paper_ids)
        },
        "entities": entities_list,
        "relationships": relationships_list
    }
    with open(graph_data_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def main() -> None:
    """Main pipeline execution routine."""
    logger.info("Initializing Knowledge Extraction Pipeline (Groq-First Architecture)")
    
    # Configure Groq API key check
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not defined in environment variables. Pipeline aborted.")
        return
    
    try:
        config = load_config()
    except ConfigError as e:
        logger.error(f"Configuration load failed: {e}")
        return
        
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    
    papers_path = os.path.join(raw_dir, "papers.json")
    if not os.path.exists(papers_path):
        logger.error(f"papers.json not found at {papers_path}. Please run Phase 1 first.")
        return
        
    with open(papers_path, "r", encoding="utf-8") as f:
        papers_payload = json.load(f)
        
    all_papers = papers_payload.get("papers", [])
    
    # Load extraction prompt template
    prompt_path = "src/prompts/extraction_prompt.txt"
    if not os.path.exists(prompt_path):
        logger.error(f"Extraction prompt template not found at {prompt_path}")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    # Load model name from config
    model_name = config.get("llm", {}).get("extraction_model", "qwen/qwen3-32b")
    logger.info(f"Using Groq model for extraction: {model_name}")
        
    processed_paper_ids: Set[str] = set()
    global_entities: Dict[Tuple[str, str], Entity] = {}
    global_relationships: Dict[Tuple[str, str, str], Relationship] = {}
    failed_count = 0
    failed_papers: List[Dict[str, str]] = []
    
    # Checkpoint resumption
    graph_data_path = os.path.join(processed_dir, "graph_data.json")
    if os.path.exists(graph_data_path):
        try:
            with open(graph_data_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                
            processed_paper_ids = set(saved_data.get("metadata", {}).get("processed_papers", []))
            failed_count = saved_data.get("metadata", {}).get("papers_failed", 0)
            
            # Create lookup map for paper enrichment
            paper_meta_map = {p.get("arxiv_id"): p for p in all_papers if p.get("arxiv_id")}
            
            # Rehydrate entities
            for ent_dict in saved_data.get("entities", []):
                ent = Entity(**ent_dict)
                if ent.entity_type == "Paper":
                    paper_meta = paper_meta_map.get(ent.entity_id)
                    if paper_meta:
                        ent.name = paper_meta.get("title", ent.name)
                        ent.title = paper_meta.get("title", ent.title)
                        ent.published = paper_meta.get("published", ent.published)
                        ent.primary_category = paper_meta.get("primary_category", ent.primary_category)
                        ent.arxiv_url = paper_meta.get("arxiv_url", ent.arxiv_url)
                        ent.pdf_url = paper_meta.get("pdf_url", ent.pdf_url)
                key = (ent.entity_id, ent.entity_type)
                global_entities[key] = ent
                
            # Rehydrate relationships
            for rel_dict in saved_data.get("relationships", []):
                rel = Relationship(**rel_dict)
                key = (rel.source, rel.target, rel.relation)
                global_relationships[key] = rel
                
            logger.info(
                f"Resumed checkpoint. Processed: {len(processed_paper_ids)} | "
                f"Entities: {len(global_entities)} | Relationships: {len(global_relationships)}"
            )
            
            # Automatically save the updated state (enrichment/schema upgrade migration)
            save_current_state(
                processed_dir,
                global_entities,
                global_relationships,
                processed_paper_ids,
                failed_count
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}. Starting fresh.")
            
    # Filter out already processed papers
    papers_to_process = [p for p in all_papers if p["paper_id"] not in processed_paper_ids]
    
    # Limit to max_papers (configured scale limit)
    max_papers = config.get("extraction", {}).get("max_papers", 10)
    papers_to_process = papers_to_process[:max_papers]
    
    if not papers_to_process:
        logger.info("No new papers to process. Graph dataset is up to date.")
        return
        
    logger.info(f"Processing batch of {len(papers_to_process)} papers (Limit: {max_papers})")
    
    pause_seconds = config.get("extraction", {}).get("pause_seconds", 3.0)
    request_delay = config.get("extraction", {}).get("request_delay_seconds", 1.0)
    
    for i, paper in enumerate(papers_to_process):
        paper_id = paper["paper_id"]
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        abstract = paper["abstract"]
        authors = paper.get("authors", [])
        categories = paper.get("categories", [])
        version = paper.get("version", 1)
        published = paper.get("published", "")
        updated = paper.get("updated", "")
        primary_category = paper.get("primary_category", "")
        arxiv_url = paper.get("arxiv_url", "")
        pdf_url = paper.get("pdf_url", "")
        
        logger.info(f"Processing paper [{i+1}/{len(papers_to_process)}]: {paper_id} - {title[:60]}")
        
        # --- LAYER 1: DETERMINISTIC GRAPH CONSTRUCTION ---
        
        # 1. Deterministic Paper Entity
        paper_desc = f"Title: {title} | Published: {published} | Version: {version}"
        merge_entity(
            global_entities,
            entity_id=arxiv_id,
            name=title,
            entity_type="Paper",
            description=paper_desc,
            confidence=1.0,
            paper_id=paper_id,
            title=title,
            published=published,
            primary_category=primary_category,
            arxiv_url=arxiv_url,
            pdf_url=pdf_url
        )
        
        # 2. Deterministic Author Entities & WRITES Relationships
        for author in authors:
            author_id = normalize_name(author)
            merge_entity(
                global_entities,
                entity_id=author_id,
                name=author,
                entity_type="Author",
                description="Researcher co-authoring papers.",
                confidence=1.0,
                paper_id=paper_id
            )
            merge_relationship(
                global_relationships,
                source=author_id,
                target=arxiv_id,
                relation="WRITES",
                description=f"Author {author} co-authored paper {arxiv_id}.",
                confidence=1.0,
                paper_id=paper_id
            )
            
        # 3. Deterministic Category Entities & BELONGS_TO Relationships
        for category in categories:
            cat_id = normalize_name(category)
            merge_entity(
                global_entities,
                entity_id=cat_id,
                name=category,
                entity_type="Category",
                description="arXiv subject category classification.",
                confidence=1.0,
                paper_id=paper_id
            )
            merge_relationship(
                global_relationships,
                source=arxiv_id,
                target=cat_id,
                relation="BELONGS_TO",
                description=f"Paper {arxiv_id} belongs to subject category {category}.",
                confidence=1.0,
                paper_id=paper_id
            )
            
        # --- LAYER 2: GROQ STRUCTURED EXTRACTION ---
        try:
            payload = extract_knowledge_with_retry(abstract, prompt_template, model_name=model_name)
            
            # Keep track of valid extracted entities that meet the MIN_ENTITY_CONFIDENCE threshold
            valid_extracted_ids = set()
            
            # 1. Merge extracted entities
            for ent in payload.entities:
                if ent.entity_type in {"Paper", "Author", "Category"}:
                    continue
                if ent.entity_type not in ALLOWED_ENTITY_TYPES:
                    continue
                
                # Apply MIN_ENTITY_CONFIDENCE threshold
                if ent.confidence is not None and ent.confidence < MIN_ENTITY_CONFIDENCE:
                    logger.info(f"Skipping entity {ent.name} due to low confidence ({ent.confidence} < {MIN_ENTITY_CONFIDENCE})")
                    continue
                    
                ent_id = normalize_name(ent.name)
                merge_entity(
                    global_entities,
                    entity_id=ent_id,
                    name=ent.name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    confidence=ent.confidence,
                    paper_id=paper_id
                )
                valid_extracted_ids.add(ent_id)
                
                # Automatically create Paper -[:MENTIONS]-> Entity relationship
                merge_relationship(
                    global_relationships,
                    source=arxiv_id,
                    target=ent_id,
                    relation="MENTIONS",
                    description=f"Paper {arxiv_id} mentions {ent.name}.",
                    confidence=ent.confidence,
                    paper_id=paper_id
                )
                
            # 2. Gather candidates for Paper -[:INTRODUCES]-> Method
            introduced_candidates = []
            for ent in payload.entities:
                if ent.entity_type == "Method" and ent.confidence >= MIN_ENTITY_CONFIDENCE:
                    if is_introduced_method(ent.name, title):
                        introduced_candidates.append(ent)
                        
            # Enforce ALLOWED_METHOD_INTRODUCTIONS = 1 limit per paper
            if introduced_candidates:
                # Sort candidates by confidence descending, then by name length descending (preferring more specific names)
                introduced_candidates.sort(key=lambda x: (x.confidence if x.confidence is not None else 0.0, len(x.name)), reverse=True)
                best_candidate = introduced_candidates[0]
                ent_id = normalize_name(best_candidate.name)
                merge_relationship(
                    global_relationships,
                    source=arxiv_id,
                    target=ent_id,
                    relation="INTRODUCES",
                    description=f"The paper {arxiv_id} introduces the method {best_candidate.name}.",
                    confidence=best_candidate.confidence,
                    paper_id=paper_id
                )
                logger.info(f"Paper {arxiv_id} introduces method: {best_candidate.name}")
                    
            # 3. Merge extracted semantic relationships
            for rel in payload.relationships:
                if rel.relation not in ALLOWED_RELATIONS:
                    logger.warning(f"Skipping disallowed relationship type: {rel.relation}")
                    continue
                
                # Apply MIN_RELATIONSHIP_CONFIDENCE threshold
                if rel.confidence is not None and rel.confidence < MIN_RELATIONSHIP_CONFIDENCE:
                    logger.info(f"Skipping relationship {rel.source} -[{rel.relation}]-> {rel.target} due to low confidence ({rel.confidence} < {MIN_RELATIONSHIP_CONFIDENCE})")
                    continue
                    
                source_id = normalize_name(rel.source)
                target_id = normalize_name(rel.target)
                
                # Validate that both endpoints exist in our global_entities cache
                # This prevents dangling edges pointing to filtered/hallucinated entities
                if not entity_exists(global_entities, source_id) or not entity_exists(global_entities, target_id):
                    logger.info(f"Skipping relationship {rel.source} -[{rel.relation}]-> {rel.target} because one or both endpoint entities do not exist (were filtered or not found).")
                    continue
                
                merge_relationship(
                    global_relationships,
                    source=source_id,
                    target=target_id,
                    relation=rel.relation,
                    description=rel.description,
                    confidence=rel.confidence,
                    paper_id=paper_id
                )
                
            processed_paper_ids.add(paper_id)
            logger.info(f"Paper {paper_id} extraction completed successfully.")
            
        except Exception as e:
            failed_count += 1
            failed_papers.append({"paper_id": paper_id, "error": str(e)})
            logger.error(f"Failed to process paper {paper_id} in Groq Layer: {e}")
            
        # Save checkpoint after every paper
        save_current_state(
            processed_dir,
            global_entities,
            global_relationships,
            processed_paper_ids,
            failed_count
        )
        
        # Pacing pause
        if i < len(papers_to_process) - 1:
            delay_to_use = max(pause_seconds, request_delay)
            logger.info(f"Pacing delay of {delay_to_use}s...")
            time.sleep(delay_to_use)
            
    # Write failed log if any failed
    if failed_papers:
        failed_path = os.path.join(processed_dir, "failed_extractions.json")
        try:
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_papers, f, indent=2, ensure_ascii=False)
            logger.warning(f"Recorded {failed_count} failures to: {failed_path}")
        except Exception as e:
            logger.error(f"Failed to write failed extractions log: {e}")
            
    logger.info("Knowledge Extraction Pipeline completed successfully")

if __name__ == "__main__":
    main()
