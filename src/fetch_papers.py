import os
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from src.utils.config import load_config, ConfigError
from src.utils.logger import logger
from src.models.paper_models import Paper
from pydantic import ValidationError

def build_query(categories: List[str], max_results: int) -> Tuple[str, Dict[str, Any]]:
    """
    Constructs the base URL and query parameters for the arXiv query API.

    Args:
        categories: List of arXiv subject categories (e.g., ["cs.AI", "cs.CL"]).
        max_results: Maximum number of papers to fetch.

    Returns:
        A tuple of (base_url, query_parameters).
    """
    base_url = "http://export.arxiv.org/api/query"
    
    # Formulate category query with OR conditions
    query_parts = [f"cat:{cat}" for cat in categories]
    search_query = " OR ".join(query_parts)
    
    params = {
        "search_query": search_query,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    return base_url, params

def fetch_feed(url: str, params: Dict[str, Any]) -> str:
    """
    Fetches the Atom feed from the arXiv API with retry logic and timeout.

    Args:
        url: Base URL of the API.
        params: Query parameters.

    Returns:
        The response content as a string.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    retries = 3
    delay = 2.0
    
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Requesting arXiv API (attempt {attempt}/{retries})...")
            # Request timeout of 30 seconds as required
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            logger.info("Successfully fetched response from arXiv API")
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Request attempt {attempt} failed: {e}")
            if attempt == retries:
                logger.error("Max retries exceeded while fetching arXiv feed")
                raise
            time.sleep(delay)
            delay *= 2.0
            
    # Fallback to satisfy type checker (normally unreachable due to raise above)
    raise requests.RequestException("Failed to fetch feed after multiple retries")

def parse_feed(xml_data: str) -> List[Dict[str, Any]]:
    """
    Parses the arXiv Atom XML feed into a list of paper dictionaries.

    Args:
        xml_data: Raw XML string content.

    Returns:
        List of dictionaries containing extracted paper attributes.
    """
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error(f"Error parsing XML response: {e}")
        raise ValueError(f"Failed to parse XML response: {e}")
        
    parsed_records = []
    entries = root.findall("atom:entry", namespaces)
    logger.info(f"Extracting fields from {len(entries)} entry elements")
    
    for entry in entries:
        # Extract unique paper ID (e.g. http://arxiv.org/abs/2103.00020v1 -> 2103.00020v1)
        id_elem = entry.find("atom:id", namespaces)
        if id_elem is None or not id_elem.text:
            logger.warning("Skipping entry: Missing or empty 'id' element")
            continue
        id_url = id_elem.text.strip()
        paper_id = id_url.split("/abs/")[-1]
        
        # Parse version and base arxiv_id from paper_id
        arxiv_id = paper_id
        version = 1
        if "v" in paper_id:
            parts = paper_id.split("v")
            if len(parts) > 1 and parts[-1].isdigit():
                version = int(parts[-1])
                arxiv_id = "v".join(parts[:-1])
                
        # Standardize landing page url using https scheme
        arxiv_url = id_url.replace("http://arxiv.org", "https://arxiv.org")

        # Clean title spaces and newlines
        title_elem = entry.find("atom:title", namespaces)
        title = ""
        if title_elem is not None and title_elem.text:
            title = " ".join(title_elem.text.split())
            
        # Clean abstract summary spaces and newlines
        summary_elem = entry.find("atom:summary", namespaces)
        abstract = ""
        if summary_elem is not None and summary_elem.text:
            abstract = " ".join(summary_elem.text.split())
            
        # Extract publication and update timestamps
        published_elem = entry.find("atom:published", namespaces)
        published = ""
        if published_elem is not None and published_elem.text:
            published = published_elem.text.strip()
            
        updated_elem = entry.find("atom:updated", namespaces)
        updated = ""
        if updated_elem is not None and updated_elem.text:
            updated = updated_elem.text.strip()
        if not updated:
            updated = published
            
        # Extract authors list
        authors = []
        for author in entry.findall("atom:author", namespaces):
            name_elem = author.find("atom:name", namespaces)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
                
        # Extract categories & primary category
        primary_category = ""
        primary_elem = entry.find("arxiv:primary_category", namespaces)
        if primary_elem is not None:
            primary_category = primary_elem.attrib.get("term", "").strip()
            
        categories = []
        for category in entry.findall("atom:category", namespaces):
            term = category.attrib.get("term")
            if term:
                categories.append(term.strip())
                
        # Fallback if primary_category is missing
        if not primary_category and categories:
            primary_category = categories[0]
            
        # Extract PDF URL
        pdf_url = None
        for link in entry.findall("atom:link", namespaces):
            rel = link.attrib.get("rel")
            title_attr = link.attrib.get("title")
            href = link.attrib.get("href")
            type_attr = link.attrib.get("type")
            
            if type_attr == "application/pdf" or title_attr == "pdf" or rel == "related":
                if href and "pdf" in href:
                    pdf_url = href.strip()
                    break
        
        # Secondary fallback search for PDF url
        if not pdf_url:
            for link in entry.findall("atom:link", namespaces):
                href = link.attrib.get("href")
                if href and (href.endswith(".pdf") or "/pdf/" in href):
                    pdf_url = href.strip()
                    break
                    
        parsed_records.append({
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "version": version,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published,
            "updated": updated,
            "primary_category": primary_category,
            "categories": categories,
            "arxiv_url": arxiv_url,
            "pdf_url": pdf_url
        })
        
    return parsed_records

def normalize_papers(parsed_records: List[Dict[str, Any]]) -> List[Paper]:
    """
    Validates and converts parsed paper dictionaries into Pydantic models.

    Args:
        parsed_records: List of raw dictionaries.

    Returns:
        List of validated Paper objects.
    """
    valid_papers = []
    for record in parsed_records:
        try:
            paper = Paper(**record)
            valid_papers.append(paper)
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed for paper {record.get('paper_id')}: {e}")
            
    logger.info(f"Successfully validated and normalized {len(valid_papers)}/{len(parsed_records)} papers")
    return valid_papers

def save_papers(papers: List[Paper], raw_xml: str, config: Dict[str, Any]) -> None:
    """
    Saves the raw XML response and the metadata-wrapped normalized papers JSON.

    Args:
        papers: List of validated Paper models.
        raw_xml: Raw API feed XML response.
        config: Configuration dictionary.
    """
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Save raw XML feed
    xml_path = os.path.join(raw_dir, "arxiv_feed.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(raw_xml)
    logger.info(f"Saved raw feed XML to: {xml_path}")
    
    # 2. Format JSON wrapper output
    retrieved_at = datetime.utcnow().isoformat() + "Z"
    categories = config.get("arxiv", {}).get("categories", [])
    max_results = config.get("arxiv", {}).get("max_results", 100)
    
    output_data = {
        "metadata": {
            "retrieved_at": retrieved_at,
            "categories": categories,
            "max_results": max_results,
            "total_papers": len(papers)
        },
        "papers": [paper.model_dump() for paper in papers]
    }
    
    # 3. Save JSON structure
    json_path = os.path.join(raw_dir, "papers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON payload to: {json_path}")

def main() -> None:
    """Main pipeline execution routine."""
    logger.info("Initializing Paper Ingestion Pipeline")
    
    try:
        config = load_config()
    except ConfigError as e:
        logger.error(f"Configuration load failed: {e}")
        return
        
    categories = config.get("arxiv", {}).get("categories", ["cs.AI", "cs.CL"])
    max_results = config.get("arxiv", {}).get("max_results", 100)
    
    logger.info(f"Querying categories: {categories} | Max results: {max_results}")
    
    url, params = build_query(categories, max_results)
    
    try:
        raw_xml = fetch_feed(url, params)
    except Exception as e:
        logger.error(f"Failed to fetch papers feed from arXiv API: {e}")
        return
        
    try:
        parsed_records = parse_feed(raw_xml)
    except Exception as e:
        logger.error(f"Failed to parse XML response: {e}")
        return
        
    normalized = normalize_papers(parsed_records)
    
    save_papers(normalized, raw_xml, config)
    logger.info("Paper Ingestion Pipeline finished successfully")

if __name__ == "__main__":
    main()
