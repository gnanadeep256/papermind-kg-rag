import time
import re
import numpy as np
from typing import Dict, Any, List, Set, Optional, Tuple, Sequence
from pydantic import BaseModel
from loguru import logger

from src.retriever import BaseRetriever, RetrievalExplanation, Citation, RetrievalResult
from src.vector_retriever import VectorRetriever
from src.kg_retriever import Neo4jKGRetriever
from src.utils.config import load_config
from src.evidence.base_policy import RetrievalReason, SelectedEvidenceChunk

class HybridRetriever(BaseRetriever):
    """
    Orchestrates hybrid retrieval combining Neo4j Knowledge Graph queries
    and FAISS semantic vector searches. Merges, deduplicates, and ranks context.
    """
    def __init__(self, vectorstore_dir: str = "data/vectorstore") -> None:
        self.vector_retriever = VectorRetriever(vectorstore_dir)
        self.kg_retriever = Neo4jKGRetriever()
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self._retrieval_cache = {}
        
        # Load configuration parameters
        self.config = load_config()
        self.retrieval_config = self.config.get("retrieval", {})
        
        # Load configurable options
        self.enable_cross_encoder = self.retrieval_config.get("enable_cross_encoder", False)
        self.cross_encoder_model_name = self.retrieval_config.get("cross_encoder_model", "BAAI/bge-reranker-base")
        self.semantic_weight = self.retrieval_config.get("semantic_weight", 0.80)
        self.graph_weight = self.retrieval_config.get("graph_weight", 0.20)
        self.top_k_entity_matches = self.retrieval_config.get("top_k_entity_matches", 5)
        self.relative_threshold = self.retrieval_config.get("relative_threshold", 0.90)
        self.token_budgets = self.retrieval_config.get("token_budget", {
            "paper": 6000,
            "method": 5000,
            "dataset": 4000,
            "research": 6000
        })
        
        # Cross encoder instance
        self.cross_encoder = None
        
    def load(self) -> None:
        """Loads underlying retrievers and builds local entity cache from Neo4j."""
        logger.info("Initializing Hybrid Retriever components...")
        self.vector_retriever.load()
        self.kg_retriever.connect()
        self.load_entity_cache()
        
        # Load optional Cross-Encoder if enabled
        if self.enable_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder: {self.cross_encoder_model_name}...")
                self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                logger.info("CrossEncoder loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}. Falling back to standard reranking.")
                self.enable_cross_encoder = False
                
        logger.info("Hybrid Retriever components loaded successfully.")

    def load_entity_cache(self) -> None:
        """Fetches all entity names, labels, and IDs from Neo4j and caches them case-insensitively, along with their embeddings."""
        self.entity_cache = {}
        query = """
        MATCH (n)
        RETURN coalesce(n.name, n.title) AS name, labels(n)[0] AS type, n.entity_id AS entity_id
        """
        try:
            with self.kg_retriever.driver.session() as session:
                res = session.run(query)
                for record in res:
                    name = record["name"]
                    if not name:
                        continue
                    entity_type = record["type"] or "Unknown"
                    entity_id = record["entity_id"]
                    
                    self.entity_cache[name.lower()] = {
                        "name": name,
                        "type": entity_type,
                        "entity_id": entity_id
                    }
            logger.info(f"Loaded {len(self.entity_cache)} entities into memory cache.")
            
            # Compute embeddings for all entity names using vector retriever's model
            if self.entity_cache and self.vector_retriever.model is not None:
                names = [ent["name"] for ent in self.entity_cache.values()]
                logger.info(f"Encoding {len(names)} entity names for semantic linking...")
                embeddings = self.vector_retriever.model.encode(names, normalize_embeddings=True)
                for (name_lower, ent), emb in zip(self.entity_cache.items(), embeddings):
                    ent["embedding"] = emb
                logger.info("Entity embeddings cached successfully.")
        except Exception as e:
            logger.error(f"Failed to load entity cache from Neo4j: {e}")

    def get_connected_entities_for_papers(self, arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        """Queries the graph for methods and datasets directly associated with given papers."""
        if not arxiv_ids:
            return []
        query = """
        MATCH (p:Paper)-[:MENTIONS|INTRODUCES]-(e)
        WHERE p.entity_id IN $arxiv_ids
          AND (e:Method OR e:Dataset OR e:Concept OR e:Task)
        RETURN coalesce(e.name, e.title) AS name, labels(e)[0] AS type, e.entity_id AS entity_id
        """
        entities = []
        try:
            with self.kg_retriever.driver.session() as session:
                res = session.run(query, arxiv_ids=arxiv_ids)
                for record in res:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "entity_id": record["entity_id"]
                    })
        except Exception as e:
            logger.error(f"Failed to fetch connected entities for papers: {e}")
        return entities

    def detect_intent(self, query: str, query_entities: List[Dict[str, Any]]) -> str:
        """Heuristically determines query intent: paper, method, dataset, or research."""
        query_lower = query.lower()
        
        # Check entity types matched in query text
        entity_types = {ent["type"] for ent in query_entities}
        
        # Prioritize dataset intent if dataset/benchmark is explicitly asked about and not a method explanation/how-it-works query
        if ("dataset" in query_lower or "benchmark" in query_lower) and not re.search(r"\b(how does|work|methods?|algorithms?)\b", query_lower):
            return "dataset"
            
        if "Paper" in entity_types:
            return "paper"
        if "Method" in entity_types:
            return "method"
        if "Dataset" in entity_types:
            return "dataset"
            
        # Fallback keyword checking
        if re.search(r"\b(summarize|explain|papers?|arxiv|about)\b", query_lower):
            return "paper"
        if re.search(r"\b(how does|work|methods?|algorithms?|approaches?|frameworks?)\b", query_lower):
            return "method"
        if re.search(r"\b(datasets?|benchmarks?|corpora|corpus|evaluations?)\b", query_lower):
            return "dataset"
            
        return "research"

    def mmr_diversify(self, query_emb: np.ndarray, candidates: List[Dict[str, Any]], top_k: int, lambda_param: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filters raw candidates using Max Marginal Relevance to avoid redundant/overlapping passages.
        """
        if not candidates or top_k <= 0:
            return []
        if len(candidates) <= top_k:
            return candidates
            
        texts = [c["text"] for c in candidates]
        try:
            embs = self.vector_retriever.model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error encoding candidates in MMR: {e}")
            return candidates[:top_k]
            
        selected_indices = [0]
        
        while len(selected_indices) < top_k and len(selected_indices) < len(candidates):
            best_mmr = -1e9
            best_idx = -1
            
            for i, cand in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                sim_query = float(np.dot(query_emb, embs[i]))
                max_sim_selected = max(float(np.dot(embs[i], embs[sel])) for sel in selected_indices)
                
                mmr_score = lambda_param * sim_query - (1 - lambda_param) * max_sim_selected
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
                    
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break
                
        return [candidates[idx] for idx in selected_indices]

    def retrieve(self, query: str, top_k_vector: Optional[int] = None, top_k_graph: Optional[int] = None, max_chunks_per_paper: int = 2, category: Optional[str] = None) -> RetrievalResult:
        """
        Gathers context from both vector store and Neo4j graph, ranks by similarity
        and graph connectivity, deduplicates files, and constructs structured Pydantic outputs.
        """
        # Retrieval Cache Check
        normalized_query = re.sub(r"\s+", " ", query.strip().lower())
        now = time.time()
        if normalized_query in self._retrieval_cache:
            ts, cached_res = self._retrieval_cache[normalized_query]
            if now - ts < 180:  # 3 minutes TTL
                logger.info(f"Retrieval cache hit for query: '{query}'")
                return cached_res

        start_time = time.time()
        
        # --- Step 1: Candidate Entity Discovery (Semantic + Regex Exact) ---
        query_entities: List[Dict[str, Any]] = []
        query_entity_ids: Set[str] = set()
        semantic_linked_entity_ids: Set[str] = set()
        
        # A. Semantic Entity Linking
        if self.entity_cache and self.vector_retriever.model is not None:
            # Encode query using the same model
            query_emb = self.vector_retriever.model.encode([query], normalize_embeddings=True)[0]
            
            candidates_list = []
            for name_lower, ent in self.entity_cache.items():
                if "embedding" in ent:
                    sim = float(np.dot(query_emb, ent["embedding"]))
                    candidates_list.append((sim, ent))
            
            candidates_list.sort(key=lambda x: x[0], reverse=True)
            top_n = candidates_list[:self.top_k_entity_matches]
            if top_n:
                best_score = top_n[0][0]
                threshold = self.relative_threshold * best_score
                for sim, ent in top_n:
                    if sim >= threshold:
                        if ent["entity_id"] not in query_entity_ids:
                            ent_copy = dict(ent)
                            ent_copy["similarity"] = sim
                            query_entities.append(ent_copy)
                            query_entity_ids.add(ent["entity_id"])
                            semantic_linked_entity_ids.add(ent["entity_id"])
                            
        # B. Regex Exact Fallback
        query_lower = query.lower()
        sorted_names = sorted(self.entity_cache.keys(), key=len, reverse=True)
        for name_lower in sorted_names:
            if name_lower in query_lower:
                pattern = rf"\b{re.escape(name_lower)}\b"
                if re.search(pattern, query_lower):
                    entity = self.entity_cache[name_lower]
                    if entity["entity_id"] not in query_entity_ids:
                        query_entities.append(dict(entity))
                        query_entity_ids.add(entity["entity_id"])
                        
        # Adaptive Retrieval Policy based on classified category & Token Budget
        prioritized_sections = []
        if category:
            if category == "summary":
                token_budget = 5000
            elif category in ["method", "workflow", "implementation", "how"]:
                token_budget = 5000
            elif category in ["dataset", "evaluation"]:
                token_budget = 4000
            elif category in ["definition", "why"]:
                token_budget = 4000
            elif category == "future_work":
                token_budget = 5000
                prioritized_sections = ["conclusion", "future work", "future", "discussion", "outlook"]
            elif category == "limitations":
                token_budget = 5000
                prioritized_sections = ["conclusion", "limitations", "limitation", "discussion"]
            elif category == "comparison":
                token_budget = 5000
            elif "architecture" in query.lower() or "design" in query.lower():
                token_budget = 5000
            else:
                token_budget = 5000
        else:
            intent_heur = self.detect_intent(query, query_entities)
            token_budget = self.token_budgets.get(intent_heur, 5000)

        # Set top_k vector targets dynamically based on token budget (approx 800 tokens per chunk)
        chosen_top_k_vector = max(4, int(token_budget / 800))
        self._current_prioritized_sections = prioritized_sections

        # --- Step 2: Intent Classification & Strategy Routing ---
        intent = self.detect_intent(query, query_entities)
        
        # Set default weights and token budgets per strategy routing
        if intent == "paper":
            strategy_name = "vector-heavy"
            chosen_top_k_graph = top_k_graph if top_k_graph is not None else 2
            hops = 1
        elif intent == "method":
            strategy_name = "hybrid"
            chosen_top_k_graph = top_k_graph if top_k_graph is not None else 8
            hops = 1
        elif intent == "dataset":
            strategy_name = "graph-heavy"
            chosen_top_k_graph = top_k_graph if top_k_graph is not None else 10
            hops = 2
        else: # research
            strategy_name = "deep-hybrid"
            chosen_top_k_graph = top_k_graph if top_k_graph is not None else 12
            hops = 2

        # --- Step 3: Semantic Vector Search & MMR Diversification ---
        vector_start = time.time()
        # Fetch extra raw candidates to account for MMR and Cross-Encoder filtering
        fetch_k = chosen_top_k_vector * 3
        raw_candidates_k = fetch_k * max_chunks_per_paper
        raw_chunks = self.vector_retriever.search(query, k=raw_candidates_k)
        
        # Stage 1: Evidence Pre-Verification (Check valid format and not empty)
        verified_raw_chunks = []
        for chunk in raw_chunks:
            if chunk.get("text") and chunk.get("chunk_id") and chunk.get("arxiv_id"):
                verified_raw_chunks.append(chunk)
        raw_chunks = verified_raw_chunks
        
        vector_time_ms = (time.time() - vector_start) * 1000
        
        # Filter raw chunks to enforce max chunks per paper limit
        paper_counts: Dict[str, int] = {}
        filtered_chunks: List[Dict[str, Any]] = []
        
        for chunk in raw_chunks:
            arxiv_id = chunk["arxiv_id"]
            count = paper_counts.get(arxiv_id, 0)
            if count < max_chunks_per_paper:
                filtered_chunks.append(chunk)
                paper_counts[arxiv_id] = count + 1
                
        # Apply MMR Diversification
        if filtered_chunks and self.vector_retriever.model is not None:
            prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
            query_emb = self.vector_retriever.model.encode([prefixed_query], normalize_embeddings=True)[0]
            filtered_chunks = self.mmr_diversify(query_emb, filtered_chunks, chosen_top_k_vector)

        # Optional Cross-Encoder Reranking
        rerank_start = time.time()
        if self.enable_cross_encoder and self.cross_encoder is not None and filtered_chunks:
            pairs = [[query, c["text"]] for c in filtered_chunks]
            ce_scores = self.cross_encoder.predict(pairs)
            for chunk, score in zip(filtered_chunks, ce_scores):
                sig_score = 1.0 / (1.0 + np.exp(-float(score)))
                chunk["rerank_score"] = sig_score
            filtered_chunks.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
            
        rerank_time_ms = (time.time() - rerank_start) * 1000
        
        # --- Step 4: Expand Entities from Retrieved Vector Chunks ---
        candidate_entities: List[Dict[str, Any]] = list(query_entities)
        candidate_entity_ids: Set[str] = set(query_entity_ids)
        
        chunk_arxiv_ids = [c["arxiv_id"] for c in filtered_chunks]
        for chunk in filtered_chunks:
            arxiv_id = chunk["arxiv_id"]
            title = chunk["title"]
            if arxiv_id not in candidate_entity_ids:
                ent = {"name": title, "type": "Paper", "entity_id": arxiv_id}
                candidate_entities.append(ent)
                candidate_entity_ids.add(arxiv_id)
                
        # Find methods/datasets directly linked to retrieved paper nodes
        connected_nodes = self.get_connected_entities_for_papers(chunk_arxiv_ids)
        for node in connected_nodes:
            if node["entity_id"] not in candidate_entity_ids:
                candidate_entities.append(node)
                candidate_entity_ids.add(node["entity_id"])
                
        # Limit candidate entities to prevent retrieval explosion
        candidate_entities = candidate_entities[:chosen_top_k_graph]
        
        # --- Step 5: Graph Context Extraction ---
        graph_start = time.time()
        subgraphs: List[Dict[str, List[Dict[str, Any]]]] = []
        
        for entity in candidate_entities:
            entity_id = entity["entity_id"]
            entity_type = entity["type"]
            entity_name = entity["name"]
            
            if entity_type == "Paper":
                subgraphs.append(self.kg_retriever.get_paper_subgraph(entity_id))
            elif entity_type == "Method":
                subgraphs.append(self.kg_retriever.get_method_context(entity_name))
                papers_subgraph = {"nodes": self.kg_retriever.get_papers_about_method(entity_name), "relationships": []}
                subgraphs.append(papers_subgraph)
            else:
                subgraphs.append(self.kg_retriever.get_entity_neighborhood(entity_id, hops=hops))
                
        # Merge subgraphs
        merged_nodes: Dict[str, Dict[str, Any]] = {}
        merged_relationships: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        
        for sg in subgraphs:
            for node in sg.get("nodes", []):
                e_id = node.get("entity_id")
                if e_id:
                    if e_id in merged_nodes:
                        merged_nodes[e_id].update(node)
                    else:
                        merged_nodes[e_id] = dict(node)
                        
            for rel in sg.get("relationships", []):
                source = rel.get("source")
                target = rel.get("target")
                relation = rel.get("relation")
                if source and target and relation:
                    key = (source, target, relation)
                    if key not in merged_relationships:
                        merged_relationships[key] = dict(rel)
                        
        graph_context = {
            "nodes": list(merged_nodes.values()),
            "relationships": list(merged_relationships.values())
        }
        graph_time_ms = (time.time() - graph_start) * 1000
        
        # --- Step 6: Configurable Graph Bonus Scoring & Explanations ---
        # Adjacency list representation of merged graph context for hop calculation
        adj: Dict[str, Set[str]] = {}
        degree_map: Dict[str, int] = {}
        for rel in graph_context["relationships"]:
            src = rel["source"]
            tgt = rel["target"]
            if src not in adj: adj[src] = set()
            if tgt not in adj: adj[tgt] = set()
            adj[src].add(tgt)
            adj[tgt].add(src)
            degree_map[src] = degree_map.get(src, 0) + 1
            degree_map[tgt] = degree_map.get(tgt, 0) + 1
            
        max_degree = max(degree_map.values()) if degree_map else 1
        if max_degree == 0:
            max_degree = 1
            
        def get_shortest_path_score(paper_id: str, q_entity_ids: Set[str]) -> Tuple[float, int]:
            if not q_entity_ids:
                return 0.0, 0
            if paper_id in q_entity_ids:
                return 1.0, 0
            if paper_id not in adj:
                return 0.0, 0
            # 1 Hop
            for n in adj[paper_id]:
                if n in q_entity_ids:
                    return 1.0, 1
            # 2 Hops
            for n in adj[paper_id]:
                if n in adj:
                    for n2 in adj[n]:
                        if n2 in q_entity_ids:
                            return 0.5, 2
            return 0.0, 0
            
        unpacked_chunks: List[Dict[str, Any]] = []
        source_papers_map: Dict[str, str] = {}
        matched_paper_ids = {ent["entity_id"] for ent in query_entities if ent["type"] == "Paper"}
        
        for chunk in filtered_chunks:
            arxiv_id = chunk["arxiv_id"]
            title = chunk["title"]
            source_papers_map[arxiv_id] = title
            
            similarity_score = chunk.get("score", chunk.get("similarity_score", 0.0))
            reranker_score = chunk.get("rerank_score", similarity_score)
            
            # Hop and path score calculation
            path_score, path_len = get_shortest_path_score(arxiv_id, query_entity_ids)
            
            # Node degree calculation of target matched entity
            normalized_degree = 0.0
            connected_entity_id = None
            if path_len == 1:
                for n in adj.get(arxiv_id, []):
                    if n in query_entity_ids:
                        deg = degree_map.get(n, 0)
                        normalized_degree = deg / max_degree
                        connected_entity_id = n
                        break
            elif path_len == 2:
                found = False
                for n in adj.get(arxiv_id, []):
                    if n in adj:
                        for n2 in adj[n]:
                            if n2 in query_entity_ids:
                                deg = degree_map.get(n2, 0)
                                normalized_degree = deg / max_degree
                                connected_entity_id = n2
                                found = True
                                break
                        if found:
                            break
                            
            graph_bonus = 0.04 * path_score + 0.01 * normalized_degree
            combined_score = self.semantic_weight * reranker_score + self.graph_weight * graph_bonus
            
            page_start = chunk.get("page_start", 1)
            page_end = chunk.get("page_end", 1)
            page_str = f"{page_start}-{page_end}" if page_start != page_end else f"{page_start}"
            
            # Preformatted context text
            context_text = (
                f"Paper:\n{title}\n\n"
                f"Section:\n{chunk.get('section', 'Unknown')}\n\n"
                f"Pages:\n{page_str}\n\n"
                f"Text:\n{chunk['text']}"
            )
            
            # Retrieval Explanations (Structured)
            explanations: List[RetrievalExplanation] = []
            explanations.append(RetrievalExplanation(
                type="semantic_match",
                score=float(reranker_score)
            ))
            
            if self.enable_cross_encoder:
                explanations.append(RetrievalExplanation(
                    type="cross_encoder_boost",
                    score=float(reranker_score)
                ))
                
            if path_len > 0:
                explanations.append(RetrievalExplanation(
                    type="graph_neighbor",
                    path_length=path_len
                ))
                if path_len >= 2:
                    explanations.append(RetrievalExplanation(
                        type="neighbor_expansion"
                    ))
                    
            if connected_entity_id in semantic_linked_entity_ids:
                explanations.append(RetrievalExplanation(
                    type="entity_link"
                ))
                
            if arxiv_id in matched_paper_ids:
                explanations.append(RetrievalExplanation(
                    type="paper_match"
                ))
                
            # Check if chunk paper introduced any matched methods
            is_intro = False
            for rel in graph_context["relationships"]:
                if rel["source"] == arxiv_id and rel["relation"] == "INTRODUCES" and rel["target"] in query_entity_ids:
                    is_intro = True
                    break
            if is_intro:
                explanations.append(RetrievalExplanation(
                    type="introduced_method"
                ))
                
            chunk_info = {
                "chunk_id": chunk["chunk_id"],
                "arxiv_id": arxiv_id,
                "title": title,
                "section": chunk.get("section", "Unknown"),
                "page_start": page_start,
                "page_end": page_end,
                "chunk_word_count": chunk.get("chunk_word_count", len(chunk["text"].split())),
                "text": chunk["text"],
                "similarity_score": float(similarity_score),
                "reranker_score": float(reranker_score),
                "graph_bonus": float(graph_bonus),
                "combined_score": float(combined_score),
                "context_text": context_text,
                "explanations": [exp.model_dump() for exp in explanations]
            }
            unpacked_chunks.append(chunk_info)
            
        # --- Step 6.5: Reusable Evidence Policy & Hierarchy Strategy ---
        from src.evidence.policy_factory import PolicyFactory
        policy = PolicyFactory.create(intent, self.retrieval_config, self.entity_cache)
        policy_result = policy.execute(unpacked_chunks, graph_context, query_entities)
        unpacked_chunks = list(policy_result.selected_chunks)
            
        # --- Step 7: Contiguous Context Chunk Packing ---
        # Group by arxiv_id
        paper_chunks: Dict[str, List[Any]] = {}
        for chunk in unpacked_chunks:
            a_id = chunk.arxiv_id if hasattr(chunk, "arxiv_id") else chunk["arxiv_id"]
            if a_id not in paper_chunks:
                paper_chunks[a_id] = []
            paper_chunks[a_id].append(chunk)
            
        packed_chunks: List[Any] = []
        
        for arxiv_id, chunks in paper_chunks.items():
            def get_chunk_idx(c):
                c_id = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
                match = re.search(r"chunk_(\d+)", c_id)
                if match:
                    return int(match.group(1))
                return 0
                
            chunks.sort(key=get_chunk_idx)
            
            current_group: List[Any] = []
            for c in chunks:
                if not current_group:
                    current_group.append(c)
                else:
                    prev_c = current_group[-1]
                    prev_idx = get_chunk_idx(prev_c)
                    curr_idx = get_chunk_idx(c)
                    if curr_idx == prev_idx + 1:
                        current_group.append(c)
                    else:
                        packed_chunks.append(self._merge_chunk_group(current_group))
                        current_group = [c]
            if current_group:
                packed_chunks.append(self._merge_chunk_group(current_group))
                
        # Re-sort packed chunks by combined score descending
        packed_chunks.sort(key=lambda x: x.combined_score if hasattr(x, "combined_score") else x["combined_score"], reverse=True)
        
        # --- Step 8: Adaptive Token Budgeting ---
        final_context: List[Any] = []
        total_tokens_used = 0
        for chunk in packed_chunks:
            chunk_word_count = chunk.chunk_word_count if hasattr(chunk, "chunk_word_count") else chunk["chunk_word_count"]
            est_tokens = int(chunk_word_count * 1.3)
            # Ensure we retrieve at least one chunk even if it exceeds the budget
            if not final_context or (total_tokens_used + est_tokens <= token_budget):
                final_context.append(chunk)
                total_tokens_used += est_tokens
            else:
                break
                
        # Deduplicate source papers based on selected final context
        source_papers_final = []
        selected_arxiv_ids = {chunk.arxiv_id if hasattr(chunk, "arxiv_id") else chunk["arxiv_id"] for chunk in final_context}
        for a_id in selected_arxiv_ids:
            if a_id in source_papers_map:
                source_papers_final.append({"arxiv_id": a_id, "title": source_papers_map[a_id]})
                
        # Build Citations aligning with final context chunks
        citations: List[Citation] = []
        for chunk in final_context:
            explanations_list = chunk.explanations if hasattr(chunk, "explanations") else chunk.get("explanations", [])
            selected_by_types = []
            for exp in explanations_list:
                exp_type = exp.get("type") if isinstance(exp, dict) else exp.type
                # Map cross_encoder_boost to cross_encoder for clean naming
                if exp_type == "cross_encoder_boost":
                    exp_type = "cross_encoder"
                selected_by_types.append(exp_type)
                
            citations.append(Citation(
                paper_title=chunk.title if hasattr(chunk, "title") else chunk["title"],
                arxiv_id=chunk.arxiv_id if hasattr(chunk, "arxiv_id") else chunk["arxiv_id"],
                section=chunk.section if hasattr(chunk, "section") else chunk["section"],
                page_start=chunk.page_start if hasattr(chunk, "page_start") else chunk["page_start"],
                page_end=chunk.page_end if hasattr(chunk, "page_end") else chunk["page_end"],
                chunk_id=chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk["chunk_id"],
                similarity_score=chunk.similarity_score if hasattr(chunk, "similarity_score") else chunk["similarity_score"],
                graph_bonus=chunk.graph_bonus if hasattr(chunk, "graph_bonus") else chunk["graph_bonus"],
                combined_score=chunk.combined_score if hasattr(chunk, "combined_score") else chunk["combined_score"],
                selected_by=selected_by_types,
                retrieval_reason=chunk.retrieval_reason if hasattr(chunk, "retrieval_reason") else chunk.get("retrieval_reason")
            ))
            
        fusion_time_ms = (time.time() - start_time) * 1000
        
        # --- Step 9: Compile Metadata & Return ---
        count_methods = sum(1 for n in graph_context["nodes"] if (n.get("entity_type") == "Method" or n.get("type") == "Method"))
        count_datasets = sum(1 for n in graph_context["nodes"] if (n.get("entity_type") == "Dataset" or n.get("type") == "Dataset"))
        count_concepts = sum(1 for n in graph_context["nodes"] if (n.get("entity_type") == "Concept" or n.get("type") == "Concept"))
        
        coverage = {
            "papers": len(source_papers_final),
            "methods": count_methods,
            "datasets": count_datasets,
            "concepts": count_concepts
        }
        
        # Compile final metadata and return
        policy_stats = {
            "name": intent,
            "fallback_used": policy_result.telemetry.fallback_used,
            "tier_distribution": {
                "tier1": policy_result.telemetry.tier_distribution.tier1,
                "tier2": policy_result.telemetry.tier_distribution.tier2,
                "tier3": policy_result.telemetry.tier_distribution.tier3
            },
            "evidence": {
                "words": policy_result.telemetry.evidence_words,
                "budget": policy_result.telemetry.evidence_budget,
                "utilization": policy_result.telemetry.utilization
            }
        }
        

        retrieval_metadata = {
            "intent": intent,
            "routing_strategy": strategy_name,
            "hops": hops,
            "token_budget": token_budget,
            "token_used": total_tokens_used,
            "entity_matches": len(query_entities),
            "query_entities": [ent["entity_id"] for ent in query_entities],
            "graph_nodes": len(graph_context["nodes"]),
            "graph_relationships": len(graph_context["relationships"]),
            "vector_candidates": len(raw_chunks),
            "packed_chunks": len(final_context),
            "coverage": coverage,
            "policy": policy_stats,
            "fusion_time_ms": float(fusion_time_ms),
            "vector_time_ms": float(vector_time_ms),
            "graph_time_ms": float(graph_time_ms),
            "rerank_time_ms": float(rerank_time_ms)
        }
        
        result = RetrievalResult(
            query=query,
            graph_context=graph_context,
            vector_context=final_context,
            source_papers=source_papers_final,
            citations=citations,
            retrieval_metadata=retrieval_metadata
        )
        self._retrieval_cache[normalized_query] = (time.time(), result)
        return result

    def _merge_chunk_group(self, group: List[Any]) -> SelectedEvidenceChunk:
        """Merges a list of contiguous chunks into a single SelectedEvidenceChunk."""
        if len(group) == 1:
            if isinstance(group[0], SelectedEvidenceChunk):
                return group[0]
            # If dict fallback is needed
            c = group[0]
            return SelectedEvidenceChunk(
                chunk_id=c.get("chunk_id"),
                arxiv_id=c.get("arxiv_id"),
                title=c.get("title"),
                section=c.get("section", "Unknown"),
                page_start=c.get("page_start", 1),
                page_end=c.get("page_end", 1),
                chunk_word_count=c.get("chunk_word_count", len(c["text"].split())),
                text=c.get("text"),
                similarity_score=c.get("similarity_score"),
                reranker_score=c.get("reranker_score"),
                graph_bonus=c.get("graph_bonus", 0.0),
                combined_score=c.get("combined_score"),
                context_text=c.get("context_text", ""),
                explanations=c.get("explanations", []),
                retrieval_reason=c.get("retrieval_reason")
            )
            
        first = group[0]
        
        def _get_val(c, attr, default=None):
            return getattr(c, attr) if hasattr(c, attr) else c.get(attr, default)

        merged_text = "\n\n".join([_get_val(c, "text") for c in group])
        
        page_start = min([_get_val(c, "page_start") for c in group])
        page_end = max([_get_val(c, "page_end") for c in group])
        total_word_count = sum([_get_val(c, "chunk_word_count") for c in group])
        
        max_similarity = max([_get_val(c, "similarity_score") for c in group])
        max_reranker = max([_get_val(c, "reranker_score") or _get_val(c, "similarity_score") for c in group])
        max_graph_bonus = max([_get_val(c, "graph_bonus") for c in group])
        max_combined_score = max([_get_val(c, "combined_score") for c in group])
        
        # Merge explanations (uniquely by type)
        seen_types = set()
        merged_explanations = []
        for c in group:
            exps = _get_val(c, "explanations", [])
            for exp in exps:
                exp_type = exp.get("type") if isinstance(exp, dict) else exp.type
                if exp_type not in seen_types:
                    seen_types.add(exp_type)
                    merged_explanations.append(exp)
                    
        if len(group) > 1:
            packed_exp = RetrievalExplanation(type="packed_context")
            merged_explanations.append(packed_exp.model_dump())
            
        title = _get_val(first, "title")
        sections = []
        for c in group:
            sec = _get_val(c, "section", "Unknown")
            if sec not in sections:
                sections.append(sec)
        section_str = ", ".join(sections)
        
        page_str = f"{page_start}-{page_end}" if page_start != page_end else f"{page_start}"
        context_text = (
            f"Paper:\n{title}\n\n"
            f"Section:\n{section_str}\n\n"
            f"Pages:\n{page_str}\n\n"
            f"Text:\n{merged_text}"
        )

        # Consolidate retrieval reasons
        from src.evidence.base_policy import RetrievalReason, RetrievalRanking, RetrievalMerge, RetrievalSource, WeightBreakdown
        merged_reason = None
        reasons_raw = [getattr(c, "retrieval_reason", None) if hasattr(c, "retrieval_reason") else c.get("retrieval_reason") for c in group]
        reasons = [r for r in reasons_raw if r is not None]
        if reasons:
            parsed_reasons = []
            for r in reasons:
                if isinstance(r, dict):
                    parsed_reasons.append(RetrievalReason.model_validate(r))
                else:
                    parsed_reasons.append(r)
            best_reason = min(parsed_reasons, key=lambda r: r.tier)
            best_ranking = best_reason.ranking
            best_source = best_reason.source
            
            merged_reason = RetrievalReason(
                policy=best_reason.policy,
                tier=best_reason.tier,
                strategy=best_reason.strategy,
                ranking=RetrievalRanking(
                    semantic_score=max(r.ranking.semantic_score for r in parsed_reasons),
                    graph_overlap=max(r.ranking.graph_overlap for r in parsed_reasons),
                    reranker_score=max((r.ranking.reranker_score or 0.0) for r in parsed_reasons) if any(r.ranking.reranker_score is not None for r in parsed_reasons) else None,
                    graph_bonus=max(r.ranking.graph_bonus for r in parsed_reasons),
                    final_score=max(r.ranking.final_score for r in parsed_reasons),
                    weight_breakdown=best_ranking.weight_breakdown
                ),
                source=best_source,
                merge=RetrievalMerge(
                    merged_chunks=len(group),
                    merged_word_count=sum(_get_val(c, "chunk_word_count") for c in group),
                    merged_chunk_ids=[_get_val(c, "chunk_id") for c in group],
                    provenance_sources=sorted(list({r.strategy for r in parsed_reasons}))
                )
            )

        return SelectedEvidenceChunk(
            chunk_id=_get_val(first, "chunk_id"),
            arxiv_id=_get_val(first, "arxiv_id"),
            title=title,
            section=section_str,
            page_start=page_start,
            page_end=page_end,
            chunk_word_count=total_word_count,
            text=merged_text,
            similarity_score=max_similarity,
            reranker_score=max_reranker,
            graph_bonus=max_graph_bonus,
            combined_score=max_combined_score,
            context_text=context_text,
            explanations=merged_explanations,
            retrieval_reason=merged_reason
        )

    def close(self) -> None:
        """Closes graph retriever connection."""
        self.kg_retriever.close()
