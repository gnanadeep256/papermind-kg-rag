import os
import json
import re
from collections import Counter
from typing import Dict, Any, List, Tuple
from loguru import logger

def detect_section(text: str, current_section: str) -> str:
    """
    Identifies academic section headings using regex and standard section naming vocabulary.
    Returns the updated section name or the current one if no heading is matched.
    """
    cleaned = text.strip()
    if len(cleaned) > 80:  # Section headings are generally short
        return current_section
        
    # Match standard numbered section patterns, e.g. "1 Introduction", "3.2 Experiments", "II. Methodology"
    pattern = r'^(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+\.?)\s+([A-Z][A-Za-z0-9\s\-\,&]+)$'
    match = re.match(pattern, cleaned)
    if match:
        return match.group(1).strip()
        
    # Match standard unnumbered section titles
    common_headings = [
        "introduction", "background", "related work", 
        "methodology", "method", "proposed method", "approach", 
        "experiments", "evaluation", "results", "discussion", 
        "conclusion", "conclusions", "references", "appendix"
    ]
    cleaned_lower = cleaned.lower()
    for heading in common_headings:
        if cleaned_lower == heading or (cleaned_lower.startswith(heading) and len(cleaned_lower) < len(heading) + 5):
            return cleaned.strip()
            
    return current_section

def chunk_document(doc: Dict[str, Any], target_size: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Chunks a single document into paragraph/sentence-aware blocks of target_size words
    with a rolling overlap of overlap words. Track section names and page boundaries for each chunk.
    """
    text = doc.get("text", "")
    arxiv_id = doc["arxiv_id"]
    title = doc["title"]
    
    blocks: List[Tuple[str, int, str, int]] = []  # tuple of (text, word_count, section, page_num)
    current_section = "Abstract"
    
    if "pages" in doc:
        for page in doc["pages"]:
            page_num = page["page_num"]
            for para in page["blocks"]:
                para_clean = para.strip()
                if not para_clean:
                    continue
                
                # Section boundary check
                new_section = detect_section(para_clean, current_section)
                if new_section != current_section:
                    current_section = new_section
                    blocks.append((para_clean, len(para_clean.split()), current_section, page_num))
                    continue
                    
                word_count = len(para_clean.split())
                # If a single paragraph is too large, split it into sentences
                if word_count > target_size:
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', para_clean) if s.strip()]
                    for sent in sentences:
                        blocks.append((sent, len(sent.split()), current_section, page_num))
                else:
                    blocks.append((para_clean, word_count, current_section, page_num))
    else:
        # Fallback to single text block if pages info is missing (e.g. mock doc)
        raw_paragraphs = text.split("\n\n")
        for para in raw_paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue
                
            # Section boundary check
            new_section = detect_section(para_clean, current_section)
            if new_section != current_section:
                current_section = new_section
                blocks.append((para_clean, len(para_clean.split()), current_section, 1))
                continue
                
            word_count = len(para_clean.split())
            # If a single paragraph is too large, split it into sentences
            if word_count > target_size:
                sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', para_clean) if s.strip()]
                for sent in sentences:
                    blocks.append((sent, len(sent.split()), current_section, 1))
            else:
                blocks.append((para_clean, word_count, current_section, 1))
            
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    
    i = 0
    n = len(blocks)
    while i < n:
        current_chunk_blocks = []
        current_words = 0
        j = i
        
        # Accumulate blocks until target size is reached
        while j < n and current_words < target_size:
            current_chunk_blocks.append(blocks[j])
            current_words += blocks[j][1]
            j += 1
            
        chunk_text = "\n\n".join(b[0] for b in current_chunk_blocks)
        
        # Resolve the major section for this chunk by frequency count
        chunk_sections = [b[2] for b in current_chunk_blocks]
        major_section = Counter(chunk_sections).most_common(1)[0][0] if chunk_sections else current_section
        
        # Resolve page start and end
        page_start = current_chunk_blocks[0][3] if current_chunk_blocks else 1
        page_end = current_chunk_blocks[-1][3] if current_chunk_blocks else 1
        
        chunks.append({
            "chunk_id": f"{arxiv_id}_chunk_{chunk_index}",
            "arxiv_id": arxiv_id,
            "title": title,
            "section": major_section,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_word_count": len(chunk_text.split()),
            "chunk_index": chunk_index,
            "text": chunk_text
        })
        chunk_index += 1
        
        if j >= n:
            break
            
        # Implement semantic overlap backtrack: count words backwards from j-1
        overlap_words = 0
        k = j - 1
        while k >= i and overlap_words < overlap:
            overlap_words += blocks[k][1]
            k -= 1
            
        # Advance starting pointer. Ensure forward progress to prevent infinite loops.
        next_i = max(i + 1, k + 1)
        if next_i >= j:
            next_i = j
            
        i = next_i
        
    return chunks

def main() -> None:
    logger.info("Initializing document chunker...")
    input_path = "data/processed/papers_text.json"
    output_path = "data/processed/chunks.json"
    
    if not os.path.exists(input_path):
        logger.error(f"Extracted paper text file not found at {input_path}")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
        
    all_chunks = []
    for doc in docs:
        logger.info(f"Chunking paper: {doc.get('arxiv_id')} - {doc.get('title')[:40]}...")
        chunks = chunk_document(doc, target_size=800, overlap=120)
        all_chunks.extend(chunks)
        logger.info(f"Generated {len(chunks)} chunks for {doc.get('arxiv_id')}.")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
        
    logger.info(f"Completed chunking. Generated a total of {len(all_chunks)} chunks, persisted to {output_path}")

if __name__ == "__main__":
    main()
