import os
import time
import requests
import re
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger
from pydantic import BaseModel

from src.hybrid_retriever import HybridRetriever, Citation, RetrievalResult
from src.observability.trace_models import TraceContext
from src.question_classifier import QuestionClassifier
from src.context_builder import ContextBuilder
from src.context_compressor import ContextCompressor
from src.citation_validator import CitationValidator
from src.confidence_estimator import ConfidenceEstimator

class EvaluationContext(BaseModel):
    vector_context: List[Any]
    graph_context: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]

class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    provenance: Dict[str, Any]
    metadata: Dict[str, Any]
    evaluation_context: Optional[EvaluationContext] = None


class GroundedAnswerGenerator:
    """
    Orchestrates the modular answer generation pipeline:
    Query -> Classifier -> Retriever -> ContextBuilder -> Compressor -> LLM -> Validator -> Confidence -> Result.
    """
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self.classifier = QuestionClassifier()
        self.context_builder = ContextBuilder()
        self.context_compressor = ContextCompressor()
        
        # Load config parameters
        self.config = self.retriever.config
        self.gen_config = self.config.get("generation", {})
        self.retrieval_config = self.config.get("retrieval", {})
        
        self.provider = self.gen_config.get("provider", "gemini")
        self.primary_model = self.gen_config.get("primary_model", "gemini-2.5-flash")
        self.deep_model = self.gen_config.get("deep_model", "gemini-2.5-pro")
        self.fallback_provider = self.gen_config.get("fallback_provider", "groq")
        self.fallback_model = self.gen_config.get("fallback_model", "llama-3")
        self.min_confidence_threshold = self.gen_config.get("min_confidence_threshold", 0.45)
        
        # Citation validator thresholds and options
        warning_th = self.retrieval_config.get("citation_alignment", {}).get("warning_threshold", 0.65)
        reject_th = self.retrieval_config.get("citation_alignment", {}).get("reject_threshold", 0.50)
        replace_inv = self.retrieval_config.get("replace_invalid_citations", True)
        
        self.citation_validator = CitationValidator(
            retriever=self.retriever,
            warning_threshold=warning_th,
            reject_threshold=reject_th,
            replace_invalid=replace_inv
        )
        
        # Dynamic weights
        self.weights = self.gen_config.get("weights", {
            "semantic": 0.45,
            "graph": 0.20,
            "citation": 0.20,
            "rerank": 0.15
        })
        self.confidence_estimator = ConfidenceEstimator(self.min_confidence_threshold, weights=self.weights)
        
        self.debug_mode = self.gen_config.get("debug_mode", True)
        self.semantic_drift_threshold = self.gen_config.get("semantic_drift_threshold", 0.55)

    def generate_answer(self, query: str, trace_context: Optional[TraceContext] = None) -> GenerationResult:
        """
        Executes the full grounded generation pipeline for a user question.
        """
        start_time = time.time()
        obs_cfg = self.config.get("observability", {})
        obs_enabled = obs_cfg.get("enabled", False)
        
        # Initialize standalone trace context if enabled but not passed
        trace_ctx = None
        if trace_context is not None:
            trace_ctx = trace_context
        elif obs_enabled:
            import uuid
            trace_ctx = TraceContext(
                experiment_id="standalone",
                run_id=f"run_{int(start_time)}",
                query_id=f"q_{int(start_time)}",
                trace_id=f"trace_{uuid.uuid4().hex[:8]}"
            )
        
        # 1. Question Classification
        intent_start = time.time()
        category = self.classifier.classify(query)
        logger.info(f"Query classified as: {category}")
        intent_time = (time.time() - intent_start) * 1000
        
        # 2. Hybrid Retrieval
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(query)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        # 3. Context Building
        # Convert graph nodes and relationships to natural facts
        context_start = time.time()
        graph_facts = self.context_builder.build_graph_facts(retrieval_result.graph_context)
        
        # Convert vector chunks to Supporting Passage blocks
        supporting_passages = self.context_builder.build_supporting_passages(retrieval_result.vector_context)
        
        # 4. Context Compression
        compressed_passages = self.context_compressor.compress(supporting_passages)
        context_time = (time.time() - context_start) * 1000
        
        # Measure context token usage
        total_tokens = sum(int((c.chunk_word_count if hasattr(c, "chunk_word_count") else c.get("chunk_word_count", 0)) * 1.3) for c in retrieval_result.vector_context)
        
        # 5. Build XML structured prompt for LLM
        facts_str = "\n".join(f"- {fact}" for fact in graph_facts) if graph_facts else "No graph relationship facts available."
        passages_str = "\n\n".join(p["formatted_text"] for p in compressed_passages) if compressed_passages else "No supporting passages available."
        
        citations_list = []
        for p in compressed_passages:
            orig = p["original_chunk"]
            orig_title = orig.title if hasattr(orig, "title") else orig.get("title", "Unknown")
            orig_arxiv_id = orig.arxiv_id if hasattr(orig, "arxiv_id") else orig.get("arxiv_id", "Unknown")
            orig_section = orig.section if hasattr(orig, "section") else orig.get("section", "Unknown")
            orig_page_start = orig.page_start if hasattr(orig, "page_start") else orig.get("page_start", 1)
            orig_page_end = orig.page_end if hasattr(orig, "page_end") else orig.get("page_end", 1)
            citations_list.append(f"{p['citation_label']}: {orig_title} (arXiv: {orig_arxiv_id}), Section: {orig_section}, Pages: {orig_page_start}-{orig_page_end}")
        citations_str = "\n".join(citations_list) if citations_list else "No citations available."
        
        prompt = (
            f"<intent_routing>\n"
            f"Query Category: {category.upper()}\n"
            f"Intent: {retrieval_result.retrieval_metadata.get('intent', 'unknown').upper()}\n"
            f"</intent_routing>\n\n"
            f"<graph_facts>\n"
            f"{facts_str}\n"
            f"</graph_facts>\n\n"
            f"<vector_context>\n"
            f"{passages_str}\n"
            f"</vector_context>\n\n"
            f"<citations>\n"
            f"{citations_str}\n"
            f"</citations>\n\n"
            f"<question>\n"
            f"{query}\n"
            f"</question>\n"
        )
        
        # Append XML style template
        style_template = self._load_prompt_template(category)
        prompt = f"{prompt}\n\n{style_template}"
        
        system_instruction = self._load_system_prompt()
        
        # 6. LLM Generation with Fallback handler
        generation_start = time.time()
        answer = ""
        provider_attempted = self.provider
        provider_used = self.provider
        fallback_used = False
        fallback_reason = None
        retry_count = 0
        generation_model = self.primary_model if self.provider == "gemini" else self.fallback_model
        
        prompt_tokens_act = None
        completion_tokens_act = None
        
        try:
            if self.provider == "gemini":
                logger.info(f"Querying primary generator Gemini ({self.primary_model})...")
                answer, prompt_tokens_act, completion_tokens_act = self._query_gemini_api(prompt, system_instruction, self.primary_model)
                generation_model = self.primary_model
            elif self.provider == "groq":
                logger.info(f"Querying primary generator Groq ({self.fallback_model})...")
                answer, prompt_tokens_act, completion_tokens_act = self._query_groq_api(prompt, system_instruction, self.fallback_model)
                generation_model = self.fallback_model
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.warning(f"Primary generator call failed: {e}. Attempting failover fallback...")
            fallback_used = True
            fallback_reason = str(e)
            retry_count = 1
            if self.provider == "gemini" and self.fallback_provider == "groq":
                provider_used = "groq"
                generation_model = self.fallback_model
                answer, prompt_tokens_act, completion_tokens_act = self._query_groq_api(prompt, system_instruction, self.fallback_model)
            elif self.provider == "groq" and self.fallback_provider == "gemini":
                provider_used = "gemini"
                generation_model = self.primary_model
                answer, prompt_tokens_act, completion_tokens_act = self._query_gemini_api(prompt, system_instruction, self.primary_model)
            else:
                logger.error("No compatible fallback provider configured or fallback failed.")
                raise e
                
        generation_time_ms = (time.time() - generation_start) * 1000
        
        # 7. Citation Validation (Semantic and paragraph-level)
        citation_start = time.time()
        validated_answer, used_citations, invalid_citations = self.citation_validator.validate(
            answer,
            retrieval_result.citations,
            retrieval_result.vector_context
        )
        citation_time_ms = (time.time() - citation_start) * 1000
        
        # 8. Confidence & Coverage Estimations
        confidence_start = time.time()
        confidence, coverage, details = self.confidence_estimator.estimate(
            retrieval_result,
            used_citations,
            citation_precision=self.citation_validator.citation_precision
        )
        
        # 9. Semantic Drift Detection
        # Strip all citation tags for a clean text drift check
        clean_answer = re.sub(r"\[Citation\s+\d+\]", "", validated_answer)
        clean_answer = re.sub(r"\[Invalid\s+Citation\s+Removed\]", "", clean_answer)
        clean_answer = re.sub(r"\s+", " ", clean_answer).strip()
        
        merged_context = "\n\n".join([(c.text if hasattr(c, "text") else c.get("text", "")) for c in retrieval_result.vector_context])
        # Include graph facts to form a single evidence document
        graph_facts_text = "\n".join(graph_facts) if graph_facts else ""
        evidence_document = merged_context
        if graph_facts_text:
            evidence_document = evidence_document + "\n\n" + graph_facts_text
            
        # Resolve embedding model and capture SQLite cache metrics
        cache_hits = 0
        cache_misses = 0
        cache_lookup_latency = 0.0
        cache_insert_latency = 0.0
        invalidation_strategy = "unknown"
        
        emb_model = None
        if self.citation_validator and self.citation_validator.model:
            emb_model = self.citation_validator.model
        elif self.retriever and hasattr(self.retriever, "vector_retriever") and self.retriever.vector_retriever.model:
            emb_model = self.retriever.vector_retriever.model
            
        if emb_model and hasattr(emb_model, "hits"):
            cache_hits = emb_model.hits
            cache_misses = emb_model.misses
            cache_lookup_latency = sum(emb_model.lookup_latencies_ms) if emb_model.lookup_latencies_ms else 0.0
            cache_insert_latency = sum(emb_model.insert_latencies_ms) if emb_model.insert_latencies_ms else 0.0
            if hasattr(emb_model, "reset_stats"):
                emb_model.reset_stats()
            if hasattr(emb_model, "cache"):
                invalidation_strategy = getattr(emb_model.cache, "invalidation_strategy", "unknown")
            
        drift_score = 1.0
        if emb_model is not None and clean_answer and evidence_document:
            try:
                answer_emb = emb_model.encode([clean_answer], normalize_embeddings=True)[0]
                context_emb = emb_model.encode([evidence_document], normalize_embeddings=True)[0]
                drift_score = float(np.dot(answer_emb, context_emb))
            except Exception as e:
                logger.error(f"Error during semantic drift check: {e}")
                
        # 10. Abstention Guard Evaluation
        should_abstain = self.confidence_estimator.should_abstain(
            retrieval_result,
            confidence,
            details
        )
        
        # Force abstention if answer has drifted beyond threshold
        if drift_score < self.semantic_drift_threshold:
            logger.warning(f"Semantic drift detected ({drift_score:.2f} < {self.semantic_drift_threshold}). Forcing abstention.")
            should_abstain = True
            
        # Enforce strict insufficiency response if confidence or matching is too low
        abstain_message = "I do not have sufficient evidence in the retrieved context to answer this query."
        is_abstained = False
        if should_abstain or abstain_message.lower() in validated_answer.lower():
            validated_answer = abstain_message
            used_citations = []
            confidence = 0.0
            coverage = 0.0
            is_abstained = True
            
        confidence_time_ms = (time.time() - confidence_start) * 1000
            
        # Compile provenance features across the final retrieved context chunks
        provenance_flags = {
            "semantic_match": False,
            "graph_neighbor": False,
            "entity_link": False,
            "cross_encoder": False,
            "packed_context": False,
            "neighbor_expansion": False,
            "paper_match": False,
            "introduced_method": False
        }
        for chunk in retrieval_result.vector_context:
            chunk_exps = chunk.explanations if hasattr(chunk, "explanations") else chunk.get("explanations", [])
            for exp in chunk_exps:
                exp_type = exp.get("type") if isinstance(exp, dict) else exp.type
                if exp_type == "semantic_match":
                    provenance_flags["semantic_match"] = True
                elif exp_type == "graph_neighbor":
                    provenance_flags["graph_neighbor"] = True
                elif exp_type == "entity_link":
                    provenance_flags["entity_link"] = True
                elif exp_type == "cross_encoder_boost":
                    provenance_flags["cross_encoder"] = True
                elif exp_type == "packed_context":
                    provenance_flags["packed_context"] = True
                elif exp_type == "neighbor_expansion":
                    provenance_flags["neighbor_expansion"] = True
                elif exp_type == "paper_match":
                    provenance_flags["paper_match"] = True
                elif exp_type == "introduced_method":
                    provenance_flags["introduced_method"] = True

        provenance_features = {}
        
        # 1. semantic_match
        sem_active = provenance_flags["semantic_match"]
        provenance_features["semantic_match"] = {
            "active": sem_active,
            "reason": None if sem_active else "no_vector_chunks_selected"
        }
        
        # 2. graph_neighbor
        graph_active = provenance_flags["graph_neighbor"]
        provenance_features["graph_neighbor"] = {
            "active": graph_active,
            "reason": None if graph_active else "no_graph_connections_to_query_entities"
        }
        
        # 3. entity_link
        el_active = provenance_flags["entity_link"]
        el_reason = None
        if not el_active:
            if not retrieval_result.retrieval_metadata.get("entity_matches", 0):
                el_reason = "no_entities_matched_in_query"
            else:
                el_reason = "no_selected_chunks_linked_to_semantic_entities"
        provenance_features["entity_link"] = {
            "active": el_active,
            "reason": el_reason
        }
        
        # 4. cross_encoder
        ce_active = provenance_flags["cross_encoder"]
        ce_reason = None
        if not ce_active:
            if not getattr(self.retriever, "enable_cross_encoder", False):
                ce_reason = "disabled_in_config"
            else:
                ce_reason = "no_vector_candidates"
        provenance_features["cross_encoder"] = {
            "active": ce_active,
            "reason": ce_reason
        }
        
        # 5. packed_context
        packed_active = provenance_flags["packed_context"]
        provenance_features["packed_context"] = {
            "active": packed_active,
            "reason": None if packed_active else "no_contiguous_chunks_packed"
        }
        
        # 6. neighbor_expansion
        ne_active = provenance_flags["neighbor_expansion"]
        ne_reason = None
        if not ne_active:
            hops_used = retrieval_result.retrieval_metadata.get("hops", 1)
            if hops_used < 2:
                ne_reason = f"hop_limit_{hops_used}"
            else:
                ne_reason = "no_2_hop_neighbors_found"
        provenance_features["neighbor_expansion"] = {
            "active": ne_active,
            "reason": ne_reason
        }
        
        # 7. paper_match
        pm_active = provenance_flags["paper_match"]
        provenance_features["paper_match"] = {
            "active": pm_active,
            "reason": None if pm_active else "no_explicit_paper_in_query"
        }
        
        # 8. introduced_method
        intro_active = provenance_flags["introduced_method"]
        provenance_features["introduced_method"] = {
            "active": intro_active,
            "reason": None if intro_active else "no_introduced_methods_matched"
        }

        # 11. Compile telemetry & metadata
        provenance = {
            "papers_used": len(retrieval_result.source_papers),
            "chunks_used": len(retrieval_result.vector_context),
            "graph_nodes_used": len(retrieval_result.graph_context["nodes"]),
            "graph_edges_used": len(retrieval_result.graph_context["relationships"]),
            "token_context": total_tokens,
            "retrieval_time_ms": float(retrieval_time_ms),
            "generation_time_ms": float(generation_time_ms),
            "features": provenance_features
        }
        
        citation_precision_val = None
        if not is_abstained and self.citation_validator.generated_count > 0:
            citation_precision_val = float(self.citation_validator.citation_precision)

        metadata = {
            "category": category,
            "provider_attempted": provider_attempted,
            "provider_used": provider_used,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "retry_count": retry_count,
            "generation_model": generation_model,
            "invalid_citations": invalid_citations,
            "citation_warnings": list(self.citation_validator.warnings),
            "citation_precision": citation_precision_val,
            "drift_score": float(drift_score),
            "answer_coverage": float(coverage),
            "confidence_breakdown": details.get("confidence_breakdown", {}),
            "retrieval_metrics": details,
            "total_execution_time_ms": float((time.time() - start_time) * 1000)
        }
        
        evaluation_context = EvaluationContext(
            vector_context=retrieval_result.vector_context,
            graph_context=retrieval_result.graph_context,
            retrieval_metadata=retrieval_result.retrieval_metadata
        )
        
        # Telemetry parsing wrapped in a safe try-except block
        if trace_ctx is not None:
            try:
                from datetime import datetime
                from src.observability.trace_models import (
                    QueryTrace, RetrievalTrace, ChunkSummary, PolicyTrace,
                    CacheTrace, GenerationTrace, CitationTrace, ConfidenceTrace
                )
                
                # 1. QueryTrace
                query_trace = QueryTrace(
                    timestamp=datetime.now().isoformat(),
                    user_query=query,
                    detected_intent=category,
                    selected_policy=retrieval_result.retrieval_metadata.get("intent", "unknown"),
                    run_mode="LIVE",
                    evaluation_version=str(self.config.get("evaluation", {}).get("version", "1.0.0"))
                )
                trace_ctx.query = query_trace

                # 2. RetrievalTrace
                retrieved_chunks_trace = []
                for chunk in retrieval_result.vector_context:
                    c_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "unknown")
                    c_arxiv = chunk.arxiv_id if hasattr(chunk, "arxiv_id") else chunk.get("arxiv_id", "unknown")
                    c_sec = chunk.section if hasattr(chunk, "section") else chunk.get("section", "unknown")
                    c_page_start = chunk.page_start if hasattr(chunk, "page_start") else chunk.get("page_start", 1)
                    c_page_end = chunk.page_end if hasattr(chunk, "page_end") else chunk.get("page_end", 1)
                    c_words = chunk.chunk_word_count if hasattr(chunk, "chunk_word_count") else chunk.get("chunk_word_count", 0)
                    
                    c_sem = chunk.similarity_score if hasattr(chunk, "similarity_score") else chunk.get("similarity_score", 0.0)
                    c_rerank = chunk.reranker_score if hasattr(chunk, "reranker_score") else chunk.get("reranker_score", 0.0)
                    c_graph_bonus = chunk.graph_bonus if hasattr(chunk, "graph_bonus") else chunk.get("graph_bonus", 0.0)
                    c_combined = chunk.combined_score if hasattr(chunk, "combined_score") else chunk.get("combined_score", 0.0)
                    
                    retrieved_chunks_trace.append(ChunkSummary(
                        chunk_id=c_id,
                        arxiv_id=c_arxiv,
                        section=c_sec,
                        page_start=c_page_start,
                        page_end=c_page_end,
                        word_count=c_words,
                        semantic_score=float(c_sem),
                        reranker_score=float(c_rerank),
                        graph_bonus=float(c_graph_bonus),
                        combined_score=float(c_combined)
                    ))
                
                context_hash_trace = None
                if obs_cfg.get("save_context_hash", True):
                    concatenated_text = "\n\n".join([(c.text if hasattr(c, "text") else c.get("text", "")) for c in retrieval_result.vector_context])
                    import hashlib
                    context_hash_trace = hashlib.sha256(concatenated_text.encode("utf-8")).hexdigest()

                retrieval_trace = RetrievalTrace(
                    retrieved_papers=[p.arxiv_id if hasattr(p, "arxiv_id") else p.get("arxiv_id", "") for p in retrieval_result.source_papers],
                    retrieved_chunks=retrieved_chunks_trace,
                    graph_nodes_count=len(retrieval_result.graph_context.get("nodes", [])),
                    graph_edges_count=len(retrieval_result.graph_context.get("relationships", [])),
                    context_hash=context_hash_trace,
                    context_tokens_estimated=total_tokens
                )
                trace_ctx.retrieval = retrieval_trace

                # 3. PolicyTrace
                policy_stats = retrieval_result.retrieval_metadata.get("policy", {})
                
                overlaps = []
                for chunk in retrieval_result.vector_context:
                    reason = getattr(chunk, "retrieval_reason", None) or chunk.get("retrieval_reason")
                    if reason:
                        ranking = getattr(reason, "ranking", None) or reason.get("ranking")
                        if ranking:
                            overlap = getattr(ranking, "graph_overlap", None) or ranking.get("graph_overlap", 0.0)
                            overlaps.append(overlap)
                graph_overlap_ratio = sum(overlaps) / len(overlaps) if overlaps else 0.0

                policy_name = policy_stats.get("name", "unknown")
                policies = self.config.get("retrieval", {}).get("policies", {})
                policy_cfg = policies.get(policy_name, {})
                ranking_cfg = policy_cfg.get("ranking", {})
                semantic_weight = ranking_cfg.get("semantic", self.config.get("retrieval", {}).get("semantic_weight", 0.80))
                graph_weight = ranking_cfg.get("graph_overlap", self.config.get("retrieval", {}).get("graph_weight", 0.20))

                policy_trace = PolicyTrace(
                    policy_selected=policy_name,
                    fallback_used=policy_stats.get("fallback_used", False),
                    tier_distribution=policy_stats.get("tier_distribution", {}),
                    budget_limit_words=policy_stats.get("evidence", {}).get("budget", 0),
                    budget_used_words=policy_stats.get("evidence", {}).get("words", 0),
                    budget_utilization=policy_stats.get("evidence", {}).get("utilization", 0.0),
                    graph_overlap_ratio=graph_overlap_ratio,
                    semantic_weight=semantic_weight,
                    graph_weight=graph_weight,
                    execution_time_ms=float(retrieval_result.retrieval_metadata.get("fusion_time_ms", retrieval_time_ms))
                )
                trace_ctx.policy = policy_trace

                # 4. CacheTrace
                cache_trace = CacheTrace(
                    hits=cache_hits,
                    misses=cache_misses,
                    hit_rate=(cache_hits / (cache_hits + cache_misses)) if (cache_hits + cache_misses) > 0 else 0.0,
                    lookup_latency_ms=cache_lookup_latency,
                    insert_latency_ms=cache_insert_latency,
                    invalidation_strategy=invalidation_strategy,
                    models_pruned=[]
                )
                trace_ctx.cache = cache_trace

                # 5. GenerationTrace
                import hashlib
                prompt_hash_val = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                sys_hash_val = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
                
                from src.observability.cost_estimator import CostEstimatorFactory
                estimator = CostEstimatorFactory.get_estimator(provider_used)
                estimated_prompt_tokens = int(len(prompt.split()) * 1.3)
                estimated_completion_tokens = int(len(answer.split()) * 1.3)
                
                p_tok = prompt_tokens_act if prompt_tokens_act is not None else estimated_prompt_tokens
                c_tok = completion_tokens_act if completion_tokens_act is not None else estimated_completion_tokens
                estimated_cost = estimator.estimate_cost(generation_model, p_tok, c_tok)

                generation_trace = GenerationTrace(
                    provider_attempted=provider_attempted,
                    provider_used=provider_used,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    model_id=generation_model,
                    prompt_hash=prompt_hash_val,
                    prompt_text=prompt if obs_cfg.get("save_prompt_text", False) else None,
                    system_template_hash=sys_hash_val,
                    temperature=float(self.gen_config.get("temperature", 0.0)),
                    max_tokens=int(self.gen_config.get("max_tokens", 1024) if self.gen_config.get("max_tokens") is not None else 1024),
                    prompt_tokens_actual=prompt_tokens_act,
                    completion_tokens_actual=completion_tokens_act,
                    total_tokens_actual=(prompt_tokens_act + completion_tokens_act) if (prompt_tokens_act is not None and completion_tokens_act is not None) else None,
                    prompt_tokens_estimated=estimated_prompt_tokens,
                    completion_tokens_estimated=estimated_completion_tokens,
                    total_tokens_estimated=estimated_prompt_tokens + estimated_completion_tokens,
                    estimated_cost=estimated_cost,
                    latency_ms=generation_time_ms
                )
                trace_ctx.generation = generation_trace

                # 6. CitationTrace
                def c_to_dict(c):
                    if hasattr(c, "model_dump"):
                        return c.model_dump()
                    return {
                        "paper_title": getattr(c, "paper_title", ""),
                        "arxiv_id": getattr(c, "arxiv_id", ""),
                        "section": getattr(c, "section", ""),
                        "page_start": getattr(c, "page_start", 1),
                        "page_end": getattr(c, "page_end", 1),
                        "chunk_id": getattr(c, "chunk_id", ""),
                        "similarity_score": getattr(c, "similarity_score", 0.0),
                        "graph_bonus": getattr(c, "graph_bonus", 0.0),
                        "combined_score": getattr(c, "combined_score", 0.0)
                    }
                citation_trace = CitationTrace(
                    generated_citations_count=self.citation_validator.generated_count,
                    validated_citations=[c_to_dict(c) for c in used_citations],
                    rejected_citations=[{"raw_tag": tag} for tag in invalid_citations],
                    semantic_alignment_scores=self.citation_validator.semantic_alignment_scores,
                    citation_precision=citation_precision_val,
                    citation_coverage=float(coverage)
                )
                trace_ctx.citation = citation_trace

                # 7. ConfidenceTrace
                confidence_trace = ConfidenceTrace(
                    semantic_score=float(details.get("confidence_breakdown", {}).get("semantic", 0.0)),
                    graph_score=float(details.get("confidence_breakdown", {}).get("graph", 0.0)),
                    citation_score=float(details.get("confidence_breakdown", {}).get("citation", 0.0)),
                    reranker_score=float(details.get("confidence_breakdown", {}).get("reranker", 0.0)),
                    final_confidence=float(confidence),
                    confidence_breakdown=details.get("confidence_breakdown", {}),
                    abstention_trigger=is_abstained,
                    drift_score=float(drift_score)
                )
                trace_ctx.confidence = confidence_trace

                # 8. Latency waterfall
                trace_ctx.latency_waterfall_ms = {
                    "intent_classification": float(intent_time),
                    "retrieval": float(retrieval_time_ms),
                    "context_assembly": float(context_time),
                    "generation": float(generation_time_ms),
                    "citation_validation": float(citation_time_ms),
                    "confidence_estimation": float(confidence_time_ms),
                    "total_query": float((time.time() - start_time) * 1000)
                }

                # Flush trace if standalone
                if trace_ctx.experiment_id == "standalone":
                    from src.observability.trace_manager import LocalTracer
                    standalone_tracer = LocalTracer("reports/experiments/standalone", compress=obs_cfg.get("compress", True))
                    standalone_tracer.log_trace(trace_ctx)

            except Exception as te:
                logger.error(f"Error compiling trace telemetry: {te}")

        return GenerationResult(
            query=query,
            answer=validated_answer,
            citations=used_citations,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata,
            evaluation_context=evaluation_context
        )


    def _load_prompt_template(self, category: str) -> str:
        """Loads prompt template from the XML files, falling back to default."""
        path = f"src/prompts/{category}.xml"
        if not os.path.exists(path):
            path = "src/prompts/default.xml"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt template from {path}: {e}")
            return "<style_instruction>Answer the query based on context.</style_instruction>"

    def _load_system_prompt(self) -> str:
        path = "src/prompts/system.xml"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            return "Answer the user question using only the provided context."

    def _query_gemini_api(self, prompt: str, system_instruction: str, model: str) -> tuple[str, Optional[int], Optional[int]]:
        """Makes direct HTTP request to the Google Gemini API using standard key variable."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not defined.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }
        
        if system_instruction:
            data["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction}
                ]
            }
            
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        
        try:
            text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            usage = res_json.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount")
            completion_tokens = usage.get("candidatesTokenCount")
            return text, prompt_tokens, completion_tokens
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Gemini API response: {res_json}. Error: {e}")
            raise ValueError("Invalid response format from Gemini API.")

    def _query_groq_api(self, prompt: str, system_instruction: str, model: str) -> tuple[str, Optional[int], Optional[int]]:
        """Makes client request to the Groq SDK."""
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not defined.")
            
        client = Groq(api_key=api_key)
        
        # Map generic model name to Groq model ID
        if model in ["llama-3", "llama3-70b-8192"]:
            model = "llama-3.3-70b-versatile"
        elif model in ["llama-3.1-8b", "llama3-8b-8192"]:
            model = "llama-3.1-8b-instant"
            
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        text = response.choices[0].message.content
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
        return text, prompt_tokens, completion_tokens
