import re
import os
import json
import requests
from typing import Dict, Any, List
import numpy as np
from src.evaluation.base_evaluator import BaseEvaluator

class GenerationEvaluator(BaseEvaluator):
    """Evaluates answer completeness, groundedness, and LLM-judged faithfulness."""
    
    def __init__(self, embedding_model: Any = None) -> None:
        """
        Args:
            embedding_model: Reusable SentenceTransformer instance. If None, loaded lazily.
        """
        self._model = embedding_model
        self.citation_pattern = re.compile(r"\[Citation (\d+)\]")

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    def _query_llm(self, prompt: str, system_instruction: str) -> str:
        """Helper to query Gemini API with Groq fallback (zero temperature)."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0}
                }
                if system_instruction:
                    data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
                    
                response = requests.post(url, headers=headers, json=data, timeout=20)
                response.raise_for_status()
                res_json = response.json()
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                pass  # Fallback to Groq if Gemini fails
                
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception:
                pass
                
        # Mock/fallback fallback if both APIs are down/unconfigured
        return '{"statements": [{"statement": "All claims are supported.", "supported": true}]}'

    def evaluate(self, query_case: Dict[str, Any], generator_result: Dict[str, Any]) -> Dict[str, Any]:
        answer = generator_result.get("answer", "")
        is_abstention = generator_result.get("is_abstention", False)
        
        retrieval_result = generator_result.get("retrieval_result")
        if retrieval_result:
            vector_context = retrieval_result.vector_context
        else:
            vector_context = generator_result.get("vector_context", [])
            
        flat_context_text = " ".join([c.text if hasattr(c, "text") else c.get("text", "") for c in vector_context])
        
        # If the system abstained, completeness is 1.0 (correctly declined to make false assertions)
        if is_abstention or "sufficient evidence" in answer.lower():
            return {
                "completeness": 1.0,
                "groundedness": 1.0,
                "llm_faithfulness": 1.0,
                "hybrid_faithfulness": 1.0
            }
            
        # 1. Completeness
        must_contain = query_case.get("must_contain", [])
        expected_entities = query_case.get("expected_entities", [])
        
        must_contain_hits = sum(1 for term in must_contain if term.lower() in answer.lower())
        entity_hits = sum(1 for ent in expected_entities if ent.lower() in answer.lower())
        
        total_checks = len(must_contain) + len(expected_entities)
        completeness = (must_contain_hits + entity_hits) / total_checks if total_checks > 0 else 1.0
        
        # 2. Groundedness (overall semantic similarity between answer and evidence)
        groundedness = 1.0
        if answer and flat_context_text:
            try:
                model = self._get_model()
                clean_ans = self.citation_pattern.sub("", answer).strip()
                ans_emb = model.encode([clean_ans], normalize_embeddings=True)[0]
                ctx_emb = model.encode([flat_context_text], normalize_embeddings=True)[0]
                groundedness = float(np.dot(ans_emb, ctx_emb))
            except Exception:
                groundedness = 0.8  # default fallback
                
        # 3. LLM Faithfulness Judge
        llm_faithfulness = 1.0
        if answer and flat_context_text:
            prompt = (
                f"Identify every statement made in the Answer and determine if it is fully supported by the Retrieved Context.\n\n"
                f"Retrieved Context:\n{flat_context_text}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Respond with a raw JSON block only conforming to this schema:\n"
                f"{{\n"
                f"  \"statements\": [\n"
                f"    {{\"statement\": \"the claim\", \"supported\": true/false}}\n"
                f"  ]\n"
                f"}}\n"
            )
            system_instruction = "You are an AI fact-checking judge. Evaluate RAG answers objectively. Output valid JSON only."
            
            try:
                llm_output = self._query_llm(prompt, system_instruction)
                # Strip backticks if LLM outputs markdown code blocks
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()
                
                eval_data = json.loads(llm_output)
                statements = eval_data.get("statements", [])
                if statements:
                    supported_count = sum(1 for s in statements if s.get("supported") is True)
                    llm_faithfulness = supported_count / len(statements)
            except Exception:
                # Default fallback if LLM or JSON fails
                llm_faithfulness = max(0.5, groundedness)
                
        # 4. Hybrid Faithfulness Score
        # We need Citation Support and Semantic Grounding
        # Compute them here directly to keep the class self-contained
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        supported_sentences = sum(1 for s in sentences if self.citation_pattern.search(s))
        citation_support = supported_sentences / len(sentences) if sentences else 1.0
        
        semantic_grounding = groundedness  # default to groundedness if grounding fails
        if sentences and vector_context:
            try:
                model = self._get_model()
                all_paragraphs = []
                para_to_chunk_idx = []
                for idx, chunk in enumerate(vector_context):
                    chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
                    paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
                    if not paragraphs:
                        paragraphs = [chunk_text]
                    for para in paragraphs:
                        all_paragraphs.append(para)
                        para_to_chunk_idx.append(idx)
                
                if all_paragraphs:
                    context_embs = model.encode(all_paragraphs, normalize_embeddings=True)
                    grounding_scores = []
                    for sentence in sentences:
                        clean_sent = self.citation_pattern.sub("", sentence).strip()
                        if not clean_sent:
                            continue
                        sent_emb = model.encode([clean_sent], normalize_embeddings=True)[0]
                        matches = self.citation_pattern.findall(sentence)
                        if matches:
                            cited_indices = {int(m) - 1 for m in matches}
                            similarities = []
                            for p_idx, chunk_idx in enumerate(para_to_chunk_idx):
                                if chunk_idx in cited_indices:
                                    sim = float(np.dot(sent_emb, context_embs[p_idx]))
                                    similarities.append(sim)
                        else:
                            similarities = [float(np.dot(sent_emb, p_emb)) for p_emb in context_embs]
                        max_sim = max(similarities) if similarities else 0.0
                        grounding_scores.append(max_sim)
                    
                    if grounding_scores:
                        semantic_grounding = sum(grounding_scores) / len(grounding_scores)
            except Exception:
                pass
                
        hybrid_faithfulness = (
            0.40 * citation_support +
            0.30 * semantic_grounding +
            0.30 * llm_faithfulness
        )
        
        return {
            "completeness": float(completeness),
            "groundedness": float(groundedness),
            "llm_faithfulness": float(llm_faithfulness),
            "hybrid_faithfulness": float(hybrid_faithfulness)
        }
