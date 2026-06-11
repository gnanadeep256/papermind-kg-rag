import os
import time
import fitz
import json
import re
import numpy as np
import faiss
import streamlit as st
import requests
from typing import Dict, Any, List
from loguru import logger

from src.chunk_documents import chunk_document
from chat_modules.chat_retriever import TemporaryRetriever
from src.answer_generator import GroundedAnswerGenerator
from utils import renumber_citations

def get_embedding_model():
    """Retrieves embedding model."""
    from src.llm import get_embedding_model as src_get_embedding_model
    from src.utils.config import load_config
    config = load_config()
    model_name = config.get("retrieval", {}).get("embedding_model", "BAAI/bge-small-en-v1.5")
    return src_get_embedding_model(model_name)

def get_groq_client(api_key: str):
    """Retrieves cached Groq client."""
    from groq import Groq
    return Groq(api_key=api_key)

def call_llm(prompt: str, system_instruction: str = "") -> str:
    """Lightweight LLM call for extraction utilities, with fallback."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            pass
            
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = get_groq_client(groq_key)
            if client:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile",
                    temperature=0.0
                )
                return chat_completion.choices[0].message.content
        except Exception as e:
            pass
            
    return "Error: No API keys configured or failed to query LLMs."

def extract_metadata_from_text(first_page_text: str) -> dict:
    """Extracts key paper metadata via LLM."""
    prompt = f"""
    Given the following beginning text of a research paper, extract its key metadata.
    Output your result in JSON format ONLY, without markdown code blocks, with these exact keys:
    "title", "arxiv_id", "authors", "abstract", "categories" (list of subject categories like cs.AI, cs.CL).
    
    Beginning text:
    {first_page_text[:4000]}
    """
    system_inst = "You are a helpful scientific metadata extraction assistant. Return raw JSON only."
    res = call_llm(prompt, system_inst)
    try:
        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        elif "```" in res:
            res = res.split("```")[1].split("```")[0]
        return json.loads(res.strip())
    except Exception:
        return {
            "title": "",
            "arxiv_id": "",
            "authors": [],
            "abstract": "",
            "categories": ["cs.AI"]
        }

def extract_entities_and_relations(title: str, abstract: str) -> dict:
    """Extracts KG nodes and relations for corpus integration.
    
    Defensively validates the LLM response to ensure it is a dict with
    'entities' and 'relationships' keys. If the LLM returns a list, an
    array-of-arrays, or any unexpected structure, falls back to empty.
    """
    prompt = f"""
    Given the title and abstract of a research paper, extract key scientific entities and relationships.
    Output your response in JSON format ONLY, without markdown code blocks, with two keys: "entities" and "relationships".
    
    Allowed entity types: Method, Concept, Dataset, Metric, Task, Organization.
    Allowed relationship types: USES, BASED_ON, EXTENDS, EVALUATED_ON, DEVELOPED_BY, SOLVES, OUTPERFORMS, COMPARED_WITH.
    
    Each entity in the list must be a JSON object with:
      "name": Name of entity (acronyms or short nouns, lowercase)
      "entity_type": One of the allowed entity types above
      "description": Brief description of this entity (max 2 sentences)
      
    Each relationship in the list must be a JSON object with:
      "source": Name of source entity
      "target": Name of target entity
      "relation": One of the allowed relationship types above
      "description": Brief description of how they connect
      
    Paper Details:
    Title: {title}
    Abstract: {abstract}
    """
    system_inst = "You are a scientific entity extraction parser. Return raw JSON matching the requested schema."
    res = call_llm(prompt, system_inst)
    try:
        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        elif "```" in res:
            res = res.split("```")[1].split("```")[0]
        parsed = json.loads(res.strip())
        # Validate that the parsed result is a dict with the expected keys.
        # The LLM can occasionally return a top-level list instead of an object.
        if not isinstance(parsed, dict):
            return {"entities": [], "relationships": []}
        # Ensure entities and relationships are lists of dicts, not lists of lists.
        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [e for e in entities if isinstance(e, dict)]
        relationships = parsed.get("relationships", [])
        if not isinstance(relationships, list):
            relationships = []
        relationships = [r for r in relationships if isinstance(r, dict)]
        return {"entities": entities, "relationships": relationships}
    except Exception:
        return {"entities": [], "relationships":[]}

def process_uploaded_pdf(uploaded_file) -> bool:
    """Ingests the uploaded PDF, extracts text, chunks, embeds, builds FAISS, and caches in st.session_state."""
    try:
        bytes_data = uploaded_file.read()
        
        # Step 1: Create temp directory and save file
        temp_dir = "data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        saved_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(saved_pdf_path, "wb") as f:
            f.write(bytes_data)
            
        # Step 2: Open and extract page text
        doc = fitz.open(saved_pdf_path)
        text_by_page = {}
        for page_num in range(len(doc)):
            text_by_page[page_num + 1] = doc[page_num].get_text()
            
        # Step 3: Save extracted text to JSON on disk
        extracted_text_path = os.path.join(temp_dir, f"{uploaded_file.name}_extracted.json")
        with open(extracted_text_path, "w", encoding="utf-8") as f:
            json.dump(text_by_page, f, indent=2)
            
        total_pages = len(doc)
        first_page_text = text_by_page.get(1, "")
        
        if not text_by_page:
            st.error("No text extracted from PDF.")
            return False
            
        p_bar = st.progress(0.0)
        status_text = st.empty()
        
        status_text.write("Ingesting document: Extracting metadata...")
        time.sleep(0.05)
        p_bar.progress(0.2)
        
        metadata = extract_metadata_from_text(first_page_text)
        p_bar.progress(0.4)
        
        status_text.write("Ingesting document: Chunking document with section tracking...")
        # Build structured page blocks, stripping PDF page header noise (page numbers, running titles)
        pages_info = []
        header_pattern = re.compile(
            r'^(\d{1,3}\n[A-Z][^\n]{0,80}\n|[A-Z][^\n]{0,80}\n\d{1,3}\n)',
            re.MULTILINE
        )
        for p_num, p_text in sorted(text_by_page.items(), key=lambda x: int(x[0])):
            # Clean page header noise: lines that are just a page number or short running title
            clean_text = header_pattern.sub('', p_text)
            # Split into blocks on double newlines
            raw_blocks = [b.strip() for b in clean_text.split("\n\n") if b.strip()]
            # Further split blocks on embedded section headings (e.g. "2.2\nSoft Cascade Decoding")
            final_blocks = []
            section_heading_re = re.compile(
                r'(?:^|\n)(\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 \-]+|[A-Z][A-Z\s]+)(?:\n)',
            )
            for block in raw_blocks:
                # Check if block contains an embedded section heading
                m = section_heading_re.search(block)
                if m and m.start() > 0:
                    # Split: text before heading, heading itself, text after
                    before = block[:m.start()].strip()
                    heading = m.group(1).strip()
                    after = block[m.end():].strip()
                    if before:
                        final_blocks.append(before)
                    if heading:
                        final_blocks.append(heading)
                    if after:
                        final_blocks.append(after)
                else:
                    final_blocks.append(block)
            pages_info.append({
                "page_num": p_num,
                "blocks": final_blocks
            })

        doc_dict = {
            "arxiv_id": metadata.get("arxiv_id") or "temp_paper",
            "title": metadata.get("title") or "Uploaded Paper",
            "pages": pages_info
        }
        # Use smaller chunk size for uploaded papers to get more granular chunks
        chunks = chunk_document(doc_dict, target_size=400, overlap=80)
        
        chunks_path = os.path.join(temp_dir, f"{uploaded_file.name}_chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        p_bar.progress(0.6)
        
        status_text.write("Ingesting document: Embedding chunks...")
        emb_model = get_embedding_model()
        texts = [c["text"] for c in chunks]
        embeddings = emb_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        p_bar.progress(0.8)
        
        status_text.write("Ingesting document: Creating temporary in-memory vector index...")
        dim = embeddings.shape[1] if len(embeddings) > 0 else 384
        index = faiss.IndexFlatIP(dim)
        if len(embeddings) > 0:
            index.add(embeddings)
        p_bar.progress(1.0)
        
        st.session_state["active_pdf_name"] = uploaded_file.name
        st.session_state["active_pdf_pages"] = total_pages
        st.session_state["active_chunks"] = chunks
        st.session_state["active_index"] = index
        st.session_state["active_metadata"] = metadata
        st.session_state["active_chat_history"] = []
        
        status_text.success("Document ingestion pipeline completed. Ready to chat.")
        time.sleep(0.3)
        status_text.empty()
        p_bar.empty()
        return True
        
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return False

def _retrieve_top_chunks(query: str, chunks: list, index, top_k: int = 4) -> list:
    """Retrieves top-k most relevant chunks from the in-memory FAISS index."""
    from src.llm import get_embedding_model
    from src.utils.config import load_config
    import numpy as np

    config = load_config()
    model_name = config.get("retrieval", {}).get("embedding_model", "BAAI/bge-small-en-v1.5")
    model = get_embedding_model(model_name)

    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    k_val = min(top_k, len(chunks))
    if k_val == 0:
        return []

    scores, indices = index.search(q_emb, k=k_val)
    results = []
    for rank_idx, score in zip(indices[0], scores[0]):
        if rank_idx < 0 or rank_idx >= len(chunks):
            continue
        c = dict(chunks[rank_idx])
        c["_score"] = float(score)
        results.append(c)
    return results


def _direct_chat_answer(query: str, context_chunks: list, history: list, metadata: dict) -> dict:
    """
    Lightweight LLM call for Chat With Paper mode.

    Uses Groq llama-3.1-8b-instant as primary (highest free-tier rate limits)
    with llama-3.3-70b and Gemini as fallbacks. Sends a compact, focused prompt
    instead of the full 5-stage GroundedAnswerGenerator pipeline, avoiding
    the rate-limit cascade that causes 'Local Extractive Fallback'.

    Returns a dict with: answer, confidence, citations, supporting_chunks.
    """
    # Build compact context from top chunks (keep total < 2500 words)
    context_parts = []
    total_words = 0
    used_chunks = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "")
        wc = len(text.split())
        if total_words + wc > 2500:
            break
        context_parts.append(f"[Citation {i}] (Section: {chunk.get('section','?')}, pp.{chunk.get('page_start','?')}-{chunk.get('page_end','?')})\n{text}")
        total_words += wc
        used_chunks.append((i, chunk))

    context_str = "\n\n---\n\n".join(context_parts)

    # Build brief conversation history (last 2 turns only)
    history_str = ""
    recent = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
    if recent:
        history_str = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {str(m.get('content',''))[:200]}"
            for m in recent
        )
        history_str = f"\nPrevious conversation (summary):\n{history_str}\n"

    paper_title = metadata.get("title", "the uploaded paper")

    system_prompt = (
        "You are a precise scientific paper assistant. Answer questions about the provided paper "
        "using ONLY the supplied context excerpts. Cite sources inline as [Citation 1], [Citation 2] etc. "
        "matching the chunk numbers given. If the context does not contain enough information "
        "to answer, say so clearly. Be concise and accurate."
    )

    user_prompt = (
        f"Paper: {paper_title}\n"
        f"{history_str}\n"
        f"Context from paper:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer using the context above. Cite chunk numbers inline as [Citation 1], [Citation 2] etc."
    )

    # Fallback chain: cheapest/fastest Groq model first, then heavier models
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    answer = None
    model_used = "unknown"

    # Try Groq llama-3.1-8b-instant first (30K TPM free tier — highest limits)
    if groq_key:
        for model_id in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model_id,
                    temperature=0.1,
                    max_tokens=800,
                )
                answer = resp.choices[0].message.content.strip()
                model_used = f"groq/{model_id}"
                break
            except Exception as e:
                logger.warning(f"Groq {model_id} failed: {e}")

    # Gemini fallback
    if not answer and gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            import requests
            data = {
                "contents": [{"parts": [{"text": user_prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800}
            }
            resp = requests.post(url, json=data, timeout=25)
            resp.raise_for_status()
            answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            model_used = "gemini/gemini-2.0-flash"
        except Exception as e:
            logger.warning(f"Gemini fallback failed: {e}")

    # Last resort: extractive summary from top chunk
    if not answer:
        if used_chunks:
            top_text = used_chunks[0][1].get("text", "")[:400]
            answer = (
                f"Based on the paper context:\n\n{top_text}...\n\n"
                f"*(Note: Could not reach LLM API — showing top retrieved passage instead.)* [Citation 1]"
            )
        else:
            answer = "No relevant context found in the uploaded paper for this question."
        model_used = "local_extractive"

    # Map inline citations back to chunk metadata
    citations_out = []
    for num, chunk in used_chunks:
        citations_out.append({
            "idx": num,
            "section": chunk.get("section", "Body"),
            "page_start": chunk.get("page_start", 1),
            "page_end": chunk.get("page_end", 1),
            "arxiv_id": chunk.get("arxiv_id", metadata.get("arxiv_id", "")),
            "paper_title": chunk.get("title", metadata.get("title", "Uploaded Paper")),
            "similarity_score": chunk.get("_score", 0.0),
        })

    # Renumber sequentially
    renumbered_answer, renumbered_citations = renumber_citations(answer, citations_out)

    # Overwrite 'idx' to be sequential 1, 2, ... and construct supporting_chunks_out
    final_citations = []
    supporting_chunks_out = []
    for i, cit in enumerate(renumbered_citations):
        orig_num = cit["idx"]
        # Find chunk by original idx
        chunk_match = next((c for num, c in used_chunks if num == orig_num), None)
        chunk_text = chunk_match.get("text", "") if chunk_match else ""

        cit_copy = dict(cit)
        cit_copy["idx"] = i + 1
        final_citations.append(cit_copy)

        supporting_chunks_out.append({
            "pages": f"{cit['page_start']}-{cit['page_end']}",
            "section": cit["section"],
            "text": chunk_text,
            "similarity": cit["similarity_score"]
        })

    # Simple confidence: average similarity of cited chunks
    sims = [c["similarity_score"] for c in final_citations if c["similarity_score"] > 0]
    confidence = float(sum(sims) / len(sims)) if sims else 0.55

    return {
        "answer": renumbered_answer,
        "confidence": min(confidence, 0.95),
        "citations": final_citations,
        "supporting_chunks": supporting_chunks_out,
        "model_used": model_used,
    }


def handle_chat_query(user_msg: str):
    """Orchestrates RAG answering over the uploaded PDF using a lightweight direct LLM call.

    Bypasses the full GroundedAnswerGenerator pipeline to avoid rate-limit cascades
    on free-tier APIs. Uses Groq llama-3.1-8b-instant as primary model.
    """
    if not st.session_state.get("active_index"):
        st.error("No active document loaded.")
        return

    st.session_state["active_chat_history"].append({"role": "user", "content": user_msg})

    with st.spinner("Analyzing paper context..."):
        try:
            retriever = TemporaryRetriever(
                index=st.session_state["active_index"],
                chunks=st.session_state["active_chunks"],
                metadata=st.session_state["active_metadata"]
            )
            retriever.load()

            # Retrieve relevant chunks using TemporaryRetriever
            retrieval_res = retriever.retrieve(user_msg)

            # Convert retrieved SelectedEvidenceChunk objects to dicts
            context_chunks = []
            for chunk in retrieval_res.vector_context:
                chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk.dict()
                chunk_dict["_score"] = chunk.similarity_score
                context_chunks.append(chunk_dict)

            res = _direct_chat_answer(
                query=user_msg,
                context_chunks=context_chunks,
                history=st.session_state["active_chat_history"][:-1],
                metadata=st.session_state["active_metadata"]
            )

            st.session_state["active_chat_history"].append({
                "role": "assistant",
                "content": res["answer"],
                "confidence": res["confidence"],
                "supporting_chunks": res["supporting_chunks"],
                "citations": res["citations"]
            })

        except Exception as e:
            st.error(f"Error analyzing paper context: {e}")
            logger.exception("Chat query handling encountered an error")
            st.session_state["active_chat_history"].append({
                "role": "assistant",
                "content": "An error occurred while analyzing the document context. Please verify the API key configuration."
            })

def integrate_into_corpus(m_title: str, m_arxiv: str, m_authors: str, m_abstract: str, m_category: str, extracted_text_path: str, chunks: list) -> bool:
    """Permanently integrates a research paper into the main GraphRAG Neo4j and FAISS databases."""
    try:
        # Step 1: Saving PDF mapping
        paper_id = m_arxiv if m_arxiv.strip() else f"temp_{int(time.time())}"
        proc_dir = "data/processed"
        papers_text_path = os.path.join(proc_dir, "papers_text.json")

        # Copy original PDF file to permanent directory (data/raw/pdfs/<paper_id>.pdf) if exists
        pdf_src = extracted_text_path.replace("_extracted.json", "")
        if os.path.exists(pdf_src):
            pdf_dest_dir = "data/raw/pdfs"
            os.makedirs(pdf_dest_dir, exist_ok=True)
            try:
                import shutil
                shutil.copy2(pdf_src, os.path.join(pdf_dest_dir, f"{paper_id}.pdf"))
                logger.info(f"Copied PDF source {pdf_src} to {pdf_dest_dir}/{paper_id}.pdf")
            except Exception as e:
                logger.warning(f"Failed to copy PDF file to raw pdfs: {e}")

        # Step 1: Saving PDF text mapping
        # papers_text.json is stored as a list of dicts: [{arxiv_id, title, text, pages}, ...]
        papers_text_list = []
        if os.path.exists(papers_text_path):
            with open(papers_text_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Handle both list format (current) and legacy dict format
            if isinstance(raw, list):
                papers_text_list = raw
            elif isinstance(raw, dict):
                # Legacy flat dict {paper_id: text} → convert to list format
                papers_text_list = [{"arxiv_id": k, "title": k, "text": v, "pages": {}} for k, v in raw.items()]
                
        with open(extracted_text_path, "r", encoding="utf-8") as f:
            text_by_page_loaded = json.load(f)
        # text_by_page_loaded may be a dict {"1": "text", ...} or a list
        if isinstance(text_by_page_loaded, dict):
            full_text = "\n\n".join(str(v) for v in text_by_page_loaded.values())
        else:
            full_text = "\n\n".join(str(v) for v in text_by_page_loaded)
            
        # Remove existing entry for this paper_id if re-ingesting
        papers_text_list = [p for p in papers_text_list if isinstance(p, dict) and p.get("arxiv_id") != paper_id]
        papers_text_list.append({
            "arxiv_id": paper_id,
            "title": m_title,
            "text": full_text,
            "pages": text_by_page_loaded if isinstance(text_by_page_loaded, dict) else {}
        })
        
        with open(papers_text_path, "w", encoding="utf-8") as f:
            json.dump(papers_text_list, f, indent=2)
            
        # Step 2: Saving sentence chunks
        main_chunks_path = os.path.join(proc_dir, "chunks.json")
        main_chunks = []
        if os.path.exists(main_chunks_path):
            with open(main_chunks_path, "r", encoding="utf-8") as f:
                main_chunks = json.load(f)
                
        new_main_chunks = []
        for idx, c in enumerate(chunks):
            new_main_chunks.append({
                "chunk_id": f"{paper_id}_chunk_{idx}",
                "arxiv_id": paper_id,
                "section": c.get("section", "Body"),
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "chunk_word_count": c["chunk_word_count"],
                "text": c["text"]
            })
            
        main_chunks.extend(new_main_chunks)
        with open(main_chunks_path, "w", encoding="utf-8") as f:
            json.dump(main_chunks, f, indent=2)
            
        # Step 3: Computing vector embeddings
        vector_dir = "data/vectorstore"
        faiss_index_path = os.path.join(vector_dir, "faiss.index")
        chunk_meta_path = os.path.join(vector_dir, "chunk_metadata.json")
        index_meta_path = os.path.join(vector_dir, "index_metadata.json")
        
        if os.path.exists(faiss_index_path) and os.path.exists(chunk_meta_path):
            index_main = faiss.read_index(faiss_index_path)
            with open(chunk_meta_path, "r", encoding="utf-8") as f:
                chunk_meta = json.load(f)
                
            model = get_embedding_model()
            new_texts = [c["text"] for c in new_main_chunks]
            new_embs = model.encode(new_texts, show_progress_bar=False, normalize_embeddings=True)
            new_embs = np.array(new_embs).astype('float32')
            
            index_main.add(new_embs)
            faiss.write_index(index_main, faiss_index_path)
            
            chunk_meta.extend(new_main_chunks)
            with open(chunk_meta_path, "w", encoding="utf-8") as f:
                json.dump(chunk_meta, f, indent=2)
                
            with open(index_meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "embedding_model": "BAAI/bge-small-en-v1.5",
                    "dimension": 384,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "total_chunks": len(chunk_meta),
                    "incremental_update": True
                }, f, indent=2)
                
        # Step 4: Knowledge Extraction
        extracted_graph = extract_entities_and_relations(m_title, m_abstract)
        
        # Step 5: Neo4j Merge
        from src.neo4j_loader import Neo4jLoader
        author_names = [a.strip() for a in m_authors.split(",") if a.strip()]
        paper_node = {
            "entity_id": paper_id,
            "entity_type": "Paper",
            "name": m_title,
            "description": m_abstract,
            "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "primary_category": m_category,
            "pdf_url": f"https://arxiv.org/abs/{paper_id}"
        }
        
        loader = Neo4jLoader()
        loader.connect()
        loader.load_entities([paper_node])
        
        extracted_entities = extracted_graph.get("entities", [])
        for e in extracted_entities:
            e["entity_id"] = e["name"].lower()
            extracted_graph["relationships"].append({
                "source": paper_id,
                "target": e["entity_id"],
                "relation": "MENTIONS",
                "description": f"Paper mentions method {e['name']}"
            })
            
        loader.load_entities(extracted_entities)
        
        author_nodes = []
        author_rels = []
        for auth in author_names:
            auth_id = auth.lower().replace(" ", "_")
            author_nodes.append({
                "entity_id": auth_id,
                "entity_type": "Author",
                "name": auth,
                "description": f"Author {auth}"
            })
            author_rels.append({
                "source": auth_id,
                "target": paper_id,
                "relation": "WRITES",
                "description": f"Author writes paper {m_title}"
            })
            
        loader.load_entities(author_nodes)
        loader.load_relationships(author_rels)
        
        cat_node = {
            "entity_id": m_category.lower(),
            "entity_type": "Category",
            "name": m_category,
            "description": f"Category {m_category}"
        }
        cat_rel = {
            "source": paper_id,
            "target": m_category.lower(),
            "relation": "BELONGS_TO",
            "description": f"Paper belongs to category {m_category}"
        }
        loader.load_entities([cat_node])
        loader.load_relationships([cat_rel])
        
        for r in extracted_graph.get("relationships", []):
            r["source"] = r["source"].lower()
            r["target"] = r["target"].lower()
            
        loader.load_relationships(extracted_graph.get("relationships", []))
        loader.close()
        
        # Step 6: FAISS Update (appending graph metadata)
        main_graph_path = os.path.join(proc_dir, "graph_data.json")
        if os.path.exists(main_graph_path):
            with open(main_graph_path, "r", encoding="utf-8") as f:
                main_graph = json.load(f)
                
            main_graph["entities"].append(paper_node)
            main_graph["entities"].extend(extracted_entities)
            main_graph["entities"].extend(author_nodes)
            main_graph["entities"].append(cat_node)
            
            ent_ids = set()
            dedup_ents = []
            for e in main_graph["entities"]:
                # Guard: skip any malformed non-dict entries (e.g. from partial saves)
                if not isinstance(e, dict):
                    continue
                eid = e.get("entity_id")
                if eid and eid not in ent_ids:
                    ent_ids.add(eid)
                    dedup_ents.append(e)
            main_graph["entities"] = dedup_ents
            
            main_graph["relationships"].extend(author_rels)
            main_graph["relationships"].append(cat_rel)
            main_graph["relationships"].extend(extracted_graph.get("relationships", []))
            
            rel_ids = set()
            dedup_rels = []
            for r in main_graph["relationships"]:
                # Guard: skip any malformed non-dict entries
                if not isinstance(r, dict):
                    continue
                src = r.get("source", "")
                rel = r.get("relation", "")
                tgt = r.get("target", "")
                r_key = f"{src}_{rel}_{tgt}"
                if r_key not in rel_ids:
                    rel_ids.add(r_key)
                    dedup_rels.append(r)
            main_graph["relationships"] = dedup_rels
            
            with open(main_graph_path, "w", encoding="utf-8") as f:
                json.dump(main_graph, f, indent=2)
                
        # Step 7: Cache Update
        from scripts import generate_graph_stats
        from scripts import prepare_ui_cache
        generate_graph_stats.main()
        prepare_ui_cache.main()
        return True
        
    except Exception as ex:
        import traceback
        full_tb = traceback.format_exc()
        st.error(f"Failed to integrate into corpus: {ex}")
        st.code(full_tb, language="python")
        return False
