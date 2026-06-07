from typing import Dict, Any, List

class ContextBuilder:
    """
    Transforms raw hybrid retrieval results (graph structures and vector passages)
    into highly organized, natural language evidence context.
    """
    def __init__(self) -> None:
        pass

    def build_graph_facts(self, graph_context: Dict[str, Any]) -> List[str]:
        """Converts graph subgraphs into clean natural language bullet-point facts sorted by relationship importance."""
        facts = []
        nodes = graph_context.get("nodes", [])
        relationships = graph_context.get("relationships", [])
        
        # Create a mapping of entity_id -> node title/name and label
        node_map = {}
        for node in nodes:
            e_id = node.get("entity_id")
            name = node.get("name") or node.get("title") or e_id
            node_type = node.get("entity_type") or node.get("type") or "Entity"
            node_map[e_id] = (name, node_type)
            
        priority = {
            "INTRODUCES": 0,
            "SOLVES": 1,
            "EVALUATED_ON": 2,
            "USES": 3,
            "MENTIONS": 4,
            "BELONGS_TO": 5
        }
        
        def rel_sort_key(rel):
            r_type = rel.get("relation") or ""
            src = rel.get("source") or ""
            tgt = rel.get("target") or ""
            return (priority.get(r_type, 999), r_type, src, tgt)
            
        sorted_relationships = sorted(relationships, key=rel_sort_key)
        
        seen_facts = set()
        for rel in sorted_relationships:
            src = rel.get("source")
            tgt = rel.get("target")
            rel_type = rel.get("relation") or "connects with"
            
            src_name, src_type = node_map.get(src, (src, "Entity"))
            tgt_name, tgt_type = node_map.get(tgt, (tgt, "Entity"))
            
            # Format facts based on relation label
            if rel_type == "INTRODUCES":
                fact = f"Paper '{src_name}' introduces the method '{tgt_name}'."
            elif rel_type == "MENTIONS":
                fact = f"Paper '{src_name}' mentions the concept/entity '{tgt_name}'."
            elif rel_type == "EVALUATED_ON":
                fact = f"Method '{src_name}' is evaluated on dataset/benchmark '{tgt_name}'."
            elif rel_type == "USES":
                fact = f"Method '{src_name}' uses '{tgt_name}'."
            elif rel_type == "SOLVES":
                fact = f"Method '{src_name}' solves task '{tgt_name}'."
            elif rel_type == "COMPARED_WITH":
                fact = f"Method '{src_name}' is compared with method '{tgt_name}'."
            elif rel_type == "EXTENDS":
                fact = f"Method '{src_name}' extends method '{tgt_name}'."
            elif rel_type == "BELONGS_TO":
                fact = f"Paper '{src_name}' belongs to category '{tgt_name}'."
            elif rel_type == "AUTHORED_BY":
                fact = f"Paper '{src_name}' is authored by '{tgt_name}'."
            elif rel_type == "AFFILIATED_WITH":
                fact = f"Author/Entity '{src_name}' is affiliated with organization '{tgt_name}'."
            else:
                rel_clean = rel_type.lower().replace("_", " ")
                fact = f"{src_type} '{src_name}' {rel_clean} {tgt_type} '{tgt_name}'."
                
            if fact not in seen_facts:
                seen_facts.add(fact)
                facts.append(fact)
                
        return facts

    def build_supporting_passages(self, vector_context: List[Any]) -> List[Dict[str, Any]]:
        """
        Formats vector context passages including citations, page numbers, and explanations.
        Injects a [Citation N] label for each chunk.
        """
        passages = []
        for idx, chunk in enumerate(vector_context, 1):
            citation_label = f"[Citation {idx}]"
            
            title = chunk.title if hasattr(chunk, "title") else chunk.get("title", "Unknown Paper")
            section = chunk.section if hasattr(chunk, "section") else chunk.get("section", "Unknown Section")
            page_start = chunk.page_start if hasattr(chunk, "page_start") else chunk.get("page_start", 1)
            page_end = chunk.page_end if hasattr(chunk, "page_end") else chunk.get("page_end", 1)
            page_str = f"{page_start}-{page_end}" if page_start != page_end else f"{page_start}"
            
            # Format structured explanations
            exps = chunk.explanations if hasattr(chunk, "explanations") else chunk.get("explanations", [])
            exp_strs = []
            for exp in exps:
                exp_type = exp.get("type") if isinstance(exp, dict) else exp.type
                if exp_type == "semantic_match":
                    score = exp.get("score") if isinstance(exp, dict) else exp.score
                    score_val = f" (score: {score:.4f})" if score is not None else ""
                    exp_strs.append(f"semantic similarity match{score_val}")
                elif exp_type == "graph_neighbor":
                    path_len = exp.get("path_length") if isinstance(exp, dict) else exp.path_length
                    exp_strs.append(f"graph neighbor connection ({path_len} hops)")
                elif exp_type == "introduced_method":
                    exp_strs.append("paper introduces the target method in the knowledge graph")
                elif exp_type == "entity_link":
                    exp_strs.append("matched via semantic entity linking to query")
                elif exp_type == "cross_encoder_boost":
                    score = exp.get("score") if isinstance(exp, dict) else exp.score
                    score_val = f" (score: {score:.4f})" if score is not None else ""
                    exp_strs.append(f"cross-encoder reranking boost{score_val}")
                elif exp_type == "packed_context":
                    exp_strs.append("packed contiguous paper context")
                elif exp_type == "neighbor_expansion":
                    exp_strs.append("expanded neighborhood entity context")
                elif exp_type == "paper_match":
                    exp_strs.append("explicit paper match found in query")
            
            reason_str = ", ".join(exp_strs) if exp_strs else "semantic vector match"
            chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            
            formatted_block = (
                f"Source Citation: {citation_label}\n"
                f"Paper: {title}\n"
                f"Section: {section}\n"
                f"Pages: {page_str}\n"
                f"Selection Reasons: {reason_str}\n"
                f"Text:\n{chunk_text}"
            )
            
            passages.append({
                "citation_label": citation_label,
                "formatted_text": formatted_block,
                "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id"),
                "arxiv_id": chunk.arxiv_id if hasattr(chunk, "arxiv_id") else chunk.get("arxiv_id"),
                "original_chunk": chunk
            })
            
        return passages
