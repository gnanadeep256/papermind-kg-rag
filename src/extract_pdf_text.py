import os
import json
import re
from collections import Counter
from typing import Dict, Any, List
import fitz  # PyMuPDF
from loguru import logger

def extract_pdf_text() -> None:
    """
    Reads the downloaded PDFs, extracts layout blocks (paragraphs), cleans headers,
    footers, and page numbers, and stores them in data/processed/papers_text.json.
    Separates paragraphs with double newlines (\n\n) to preserve document structure.
    """
    pdf_dir = "data/raw/pdfs"
    manifest_path = os.path.join(pdf_dir, "download_manifest.json")
    output_path = "data/processed/papers_text.json"
    
    os.makedirs("data/processed", exist_ok=True)
    
    if not os.path.exists(manifest_path):
        logger.error(f"Download manifest not found at {manifest_path}. Please run download_pdfs first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    extracted_docs: List[Dict[str, Any]] = []
    
    # Load existing extracted text if it exists (for incremental extraction)
    existing_docs = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_docs = {doc["arxiv_id"]: doc for doc in data}
        except Exception as e:
            logger.warning(f"Could not load existing extracted text: {e}. Starting fresh.")
            
    success_count = 0
    fail_count = 0
    
    for arxiv_id, info in manifest.items():
        if info.get("status") != "success":
            continue
            
        local_path = info.get("local_path")
        if not local_path or not os.path.exists(local_path):
            logger.warning(f"PDF file for {arxiv_id} not found at {local_path}")
            continue
            
        # Skip extraction if it has already been done and contains page information
        if arxiv_id in existing_docs and "pages" in existing_docs[arxiv_id]:
            logger.debug(f"Skipping text extraction for {arxiv_id}, already processed with page information.")
            extracted_docs.append(existing_docs[arxiv_id])
            success_count += 1
            continue
            
        logger.info(f"Extracting text from {local_path}...")
        try:
            doc = fitz.open(local_path)
            full_text = []
            pages_data = []
            
            # Gather first and last lines on all pages to identify repeating headers/footers
            page_first_lines = []
            page_last_lines = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
                if text_lines:
                    page_first_lines.append(text_lines[0])
                    page_last_lines.append(text_lines[-1])
                else:
                    page_first_lines.append(None)
                    page_last_lines.append(None)
            
            # Flag header/footer lines repeating on > 20% of pages (min 2 pages)
            threshold = max(2, len(doc) // 5)
            first_line_counts = Counter([line for line in page_first_lines if line])
            last_line_counts = Counter([line for line in page_last_lines if line])
            
            headers_to_remove = {line for line, count in first_line_counts.items() if count > threshold}
            footers_to_remove = {line for line, count in last_line_counts.items() if count > threshold}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Extract layout blocks instead of raw text lines to preserve paragraphs
                blocks = page.get_text("blocks")
                page_blocks = []
                
                for b in blocks:
                    block_text = b[4].strip()
                    if not block_text:
                        continue
                    
                    # Split block text into lines and clean them
                    lines = [l.strip() for l in block_text.split("\n") if l.strip()]
                    cleaned_lines = []
                    
                    for line in lines:
                        if line in headers_to_remove or line in footers_to_remove:
                            continue
                        if line.isdigit() and len(line) <= 3:
                            continue
                        cleaned_lines.append(line)
                        
                    if cleaned_lines:
                        # Join block lines with spaces and normalize horizontal whitespace
                        cleaned_block = " ".join(cleaned_lines)
                        cleaned_block = re.sub(r"\s+", " ", cleaned_block).strip()
                        page_blocks.append(cleaned_block)
                
                # Join page blocks with double newlines
                full_text.append("\n\n".join(page_blocks))
                pages_data.append({
                    "page_num": page_num + 1,  # 1-indexed page number
                    "blocks": page_blocks
                })
                
            doc.close()
            
            # Join all pages with double newlines
            combined_text = "\n\n".join(full_text)
            
            extracted_docs.append({
                "arxiv_id": arxiv_id,
                "title": info.get("title", "Unknown"),
                "text": combined_text,
                "pages": pages_data
            })
            success_count += 1
            logger.info(f"Successfully extracted text from {arxiv_id} ({len(combined_text.split())} words).")
            
        except Exception as e:
            logger.error(f"Failed to extract text from {local_path}: {e}")
            fail_count += 1
            
    # Save the extracted docs list
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_docs, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("PDF TEXT EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Extracted: {success_count}")
    logger.info(f"Failed   : {fail_count}")
    logger.info(f"Total    : {success_count + fail_count}")
    logger.info("=" * 60)

if __name__ == "__main__":
    extract_pdf_text()
