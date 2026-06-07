import os
import time
import requests
import re
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger
from pydantic import BaseModel

from src.hybrid_retriever import HybridRetriever, Citation, RetrievalResult
from src.question_classifier import QuestionClassifier
from src.context_builder import ContextBuilder
from src.context_compressor import ContextCompressor
from src.citation_validator import CitationValidator
from src.confidence_estimator import ConfidenceEstimator

class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    provenance: Dict[str, Any]
    metadata: Dict[str, Any]

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

    def generate_answer(self, query: str) -> GenerationResult:
        """
        Executes the full grounded generation pipeline for a user question.
        """
        start_time = time.time()
        
        # 1. Question Classification
        category = self.classifier.classify(query)
        logger.info(f"Query classified as: {category}")
        
        # 2. Hybrid Retrieval
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(query)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        # 3. Context Building
        # Convert graph nodes and relationships to natural facts
        graph_facts = self.context_builder.build_graph_facts(retrieval_result.graph_context)
        
        # Convert vector chunks to Supporting Passage blocks
        supporting_passages = self.context_builder.build_supporting_passages(retrieval_result.vector_context)
        
        # 4. Context Compression
        compressed_passages = self.context_compressor.compress(supporting_passages)
        
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
        
        try:
            if self.provider == "gemini":
                logger.info(f"Querying primary generator Gemini ({self.primary_model})...")
                answer = self._query_gemini_api(prompt, system_instruction, self.primary_model)
                generation_model = self.primary_model
            elif self.provider == "groq":
                logger.info(f"Querying primary generator Groq ({self.fallback_model})...")
                answer = self._query_groq_api(prompt, system_instruction, self.fallback_model)
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
                answer = self._query_groq_api(prompt, system_instruction, self.fallback_model)
            elif self.provider == "groq" and self.fallback_provider == "gemini":
                provider_used = "gemini"
                generation_model = self.primary_model
                answer = self._query_gemini_api(prompt, system_instruction, self.primary_model)
            else:
                logger.error("No compatible fallback provider configured or fallback failed.")
                raise e
                
        generation_time_ms = (time.time() - generation_start) * 1000
        
        # 7. Citation Validation (Semantic and paragraph-level)
        validated_answer, used_citations, invalid_citations = self.citation_validator.validate(
            answer,
            retrieval_result.citations,
            retrieval_result.vector_context
        )
        
        # 8. Confidence & Coverage Estimations
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
        
        # Resolve embedding model
        emb_model = None
        if self.citation_validator and self.citation_validator.model:
            emb_model = self.citation_validator.model
        elif self.retriever and hasattr(self.retriever, "vector_retriever") and self.retriever.vector_retriever.model:
            emb_model = self.retriever.vector_retriever.model
            
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
        
        return GenerationResult(
            query=query,
            answer=validated_answer,
            citations=used_citations,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata
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

    def _query_gemini_api(self, prompt: str, system_instruction: str, model: str) -> str:
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
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Gemini API response: {res_json}. Error: {e}")
            raise ValueError("Invalid response format from Gemini API.")

    def _query_groq_api(self, prompt: str, system_instruction: str, model: str) -> str:
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
        return response.choices[0].message.content
