from typing import Dict, Any, List
from difflib import SequenceMatcher

def get_overlap_similarity(a: str, b: str) -> float:
    """Calculates matching characters size divided by the minimum string length."""
    matcher = SequenceMatcher(None, a, b)
    matching_size = sum(block.size for block in matcher.get_matching_blocks())
    min_len = min(len(a), len(b))
    return matching_size / min_len if min_len > 0 else 0.0

class ContextCompressor:
    """
    Compresses retrieved context documents by removing redundant paragraphs
    while preserving page numbers, citations, and metadata anchors.
    """
    def __init__(self) -> None:
        pass

    def compress(self, formatted_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates paragraphs across passages using a global length-aware similarity algorithm.
        Returns compressed passages.
        """
        # Step 1: Parse all passages into list of paragraph dicts
        parsed_passages = []
        for p_idx, p in enumerate(formatted_passages):
            text = p.get("formatted_text", "")
            parts = text.split("Text:\n", 1)
            if len(parts) < 2:
                # No "Text:\n" found, just keep as is
                parsed_passages.append({
                    "headers": text,
                    "paragraphs": [],
                    "original_passage": p
                })
                continue
            
            headers, body = parts[0], parts[1]
            paragraphs = [para for para in body.split("\n\n") if para.strip()]
            parsed_passages.append({
                "headers": headers,
                "paragraphs": [{"text": para, "keep": True} for para in paragraphs],
                "original_passage": p
            })
            
        # Step 2: Compare all paragraphs globally to find duplicates
        kept_paras = [] # list of tuples: (passage_idx, para_idx, text_len, cleaned_text)
        
        for p_idx, parsed in enumerate(parsed_passages):
            for para_idx, para_dict in enumerate(parsed["paragraphs"]):
                text = para_dict["text"]
                cleaned = text.strip().lower()
                
                if len(cleaned) < 50:
                    # Short paragraphs are always kept
                    continue
                    
                # Find if there is a >95% duplicate among kept paragraphs
                duplicate_idx = -1
                for k_idx, (kp_p_idx, kp_para_idx, kp_len, kp_cleaned) in enumerate(kept_paras):
                    sim = get_overlap_similarity(cleaned, kp_cleaned)
                    if sim >= 0.95:
                        duplicate_idx = k_idx
                        break
                        
                if duplicate_idx != -1:
                    # Duplicate found! Compare lengths
                    kp_p_idx, kp_para_idx, kp_len, kp_cleaned = kept_paras[duplicate_idx]
                    
                    # Length difference ratio relative to the maximum length
                    max_len = max(len(cleaned), kp_len)
                    len_diff_ratio = abs(len(cleaned) - kp_len) / max_len if max_len > 0 else 0.0
                    
                    if len_diff_ratio < 0.20:
                        # Length difference < 20%: keep the shorter paragraph
                        if len(cleaned) < kp_len:
                            # New one is shorter: keep new one, mark the previous one as not kept
                            parsed_passages[kp_p_idx]["paragraphs"][kp_para_idx]["keep"] = False
                            kept_paras[duplicate_idx] = (p_idx, para_idx, len(cleaned), cleaned)
                        else:
                            # Old one is shorter/equal: discard new one
                            para_dict["keep"] = False
                    else:
                        # Length difference >= 20%: keep the longer paragraph
                        if len(cleaned) > kp_len:
                            # New one is longer: keep new one, mark the previous one as not kept
                            parsed_passages[kp_p_idx]["paragraphs"][kp_para_idx]["keep"] = False
                            kept_paras[duplicate_idx] = (p_idx, para_idx, len(cleaned), cleaned)
                        else:
                            # Old one is longer: discard new one
                            para_dict["keep"] = False
                else:
                    # No duplicate found: keep it
                    kept_paras.append((p_idx, para_idx, len(cleaned), cleaned))
                    
        # Step 3: Reconstruct the passages
        compressed_passages = []
        for parsed in parsed_passages:
            p = parsed["original_passage"]
            # If no paragraphs were parsed (e.g. no "Text:\n"), keep as is
            if len(p.get("formatted_text", "").split("Text:\n", 1)) < 2:
                compressed_passages.append(p)
                continue
                
            headers = parsed["headers"]
            unique_paragraphs = []
            for para_dict in parsed["paragraphs"]:
                if para_dict["keep"]:
                    unique_paragraphs.append(para_dict["text"])
                    
            if unique_paragraphs:
                compressed_body = "\n\n".join(unique_paragraphs)
                compressed_text = f"{headers}Text:\n{compressed_body}"
            else:
                # If all content was duplicate, keep citation but omit redundant text
                compressed_text = f"{headers}Text:\n[Duplicate content omitted - refer to previous citations]"
                
            p_copy = dict(p)
            p_copy["formatted_text"] = compressed_text
            compressed_passages.append(p_copy)
            
        return compressed_passages
