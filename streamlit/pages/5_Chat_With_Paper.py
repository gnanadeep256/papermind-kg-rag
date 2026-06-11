import streamlit as st
import os
import sys

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_css, render_footer, render_llm_answer
from chat_modules.chat_service import (
    process_uploaded_pdf, handle_chat_query, integrate_into_corpus
)

# Set page config
st.set_page_config(
    page_title="Chat With Paper",
    layout="wide"
)

# Load layout styles
load_css()

st.markdown("<h1>Chat With Paper</h1>", unsafe_allow_html=True)
st.write("Upload a research PDF to construct a session-only index for temporary chat, or permanently integrate it into the GraphRAG corpus.")

# Initialize persistent session state keys
if "active_pdf_name" not in st.session_state:
    st.session_state["active_pdf_name"] = None
if "active_pdf_pages" not in st.session_state:
    st.session_state["active_pdf_pages"] = 0
if "active_chunks" not in st.session_state:
    st.session_state["active_chunks"] = []
if "active_index" not in st.session_state:
    st.session_state["active_index"] = None
if "active_metadata" not in st.session_state:
    st.session_state["active_metadata"] = {}
if "active_chat_history" not in st.session_state:
    st.session_state["active_chat_history"] = []
if "clicked_suggested" not in st.session_state:
    st.session_state["clicked_suggested"] = None

# Upload Section
uploaded_file = st.file_uploader("Upload Paper PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state["active_pdf_name"] != uploaded_file.name:
        if process_uploaded_pdf(uploaded_file):
            st.rerun()

if st.session_state["active_pdf_name"] is not None:
    pdf_name = st.session_state["active_pdf_name"]
    total_pages = st.session_state["active_pdf_pages"]
    chunks = st.session_state["active_chunks"]
    metadata = st.session_state["active_metadata"]
    
    st.sidebar.markdown("### Document Audit & Verification")
    temp_dir = "data/temp_uploads"
    saved_pdf_path = os.path.join(temp_dir, pdf_name)
    extracted_text_path = os.path.join(temp_dir, f"{pdf_name}_extracted.json")
    
    dir_ok = os.path.exists(temp_dir)
    pdf_ok = os.path.exists(saved_pdf_path) and os.path.getsize(saved_pdf_path) > 0
    text_ok = os.path.exists(extracted_text_path)
    page_count_ok = total_pages > 0
    chunk_count_ok = len(chunks) > 0
    
    def get_status_str(ok_val):
        return "✓ PASS" if ok_val else "FAIL"
        
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.85rem; line-height: 1.6; color:#0f172a; border:1px solid #e2e8f0; border-radius:6px; padding:10px; background-color:#ffffff;">
            <strong>Verification Checklist</strong><br/>
            &bull; Temp directory: <code>{get_status_str(dir_ok)}</code><br/>
            &bull; PDF file saved: <code>{get_status_str(pdf_ok)}</code><br/>
            &bull; Text extracted: <code>{get_status_str(text_ok)}</code><br/>
            &bull; Pages verified: <code>{total_pages} ({get_status_str(page_count_ok)})</code><br/>
            &bull; Chunks verified: <code>{len(chunks)} ({get_status_str(chunk_count_ok)})</code>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    if os.path.exists(saved_pdf_path):
        try:
            with open(saved_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.sidebar.download_button(
                label="📥 Download Active PDF",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            pass

    if st.sidebar.button("Remove Active Paper", use_container_width=True, type="secondary"):
        st.session_state["active_pdf_name"] = None
        st.session_state["active_pdf_pages"] = 0
        st.session_state["active_chunks"] = []
        st.session_state["active_index"] = None
        st.session_state["active_metadata"] = {}
        st.session_state["active_chat_history"] = []
        st.session_state["clicked_suggested"] = None
        st.rerun()
        
    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 15px; margin-bottom: 20px;">
            <h4 style="margin-top:0; color:#4f46e5; font-family:'Outfit',sans-serif;">Temporary Paper Context Active</h4>
            <p style="margin: 0; font-size: 0.9rem; color:#64748b;">
                <strong>Document</strong>: {pdf_name} &bull; 
                <strong>Pages</strong>: {total_pages} &bull; 
                <strong>Chunks</strong>: {len(chunks)}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_qa, col_ingest = st.columns([2, 1])
    
    with col_qa:
        st.markdown("### Conversation Pane")
        
        st.markdown("<p style='font-size: 0.85rem; color:#64748b; margin-bottom: 5px;'>Suggested Questions:</p>", unsafe_allow_html=True)
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            if st.button("Summarize this paper", use_container_width=True):
                st.session_state.clicked_suggested = "Summarize this paper"
            if st.button("What is the methodology?", use_container_width=True):
                st.session_state.clicked_suggested = "What methodology is used in this paper?"
        with c_s2:
            if st.button("What datasets are used?", use_container_width=True):
                st.session_state.clicked_suggested = "What datasets are used for evaluation?"
            if st.button("What are the limitations?", use_container_width=True):
                st.session_state.clicked_suggested = "What are the limitations of this work?"
        with c_s3:
            if st.button("What future work is proposed?", use_container_width=True):
                st.session_state.clicked_suggested = "What future work is proposed?"
                
        if st.button("Clear Chat History", type="secondary"):
            st.session_state["active_chat_history"] = []
            st.rerun()
            
        for msg in st.session_state["active_chat_history"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_llm_answer(msg["content"])
                else:
                    st.markdown(msg["content"])
                    
                if msg.get("role") == "assistant" and "confidence" in msg:
                    conf_pct = int(msg["confidence"] * 100)
                    st.markdown(
                        f"""
                        <div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; font-size: 0.8rem; color:#64748b; margin-top:8px; display:inline-block;">
                            <strong>Confidence Score</strong>: {conf_pct}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # --- References Panel ---
                    citations = msg.get("citations", [])
                    if citations:
                        ref_html = []
                        ref_html.append(
                            '<div style="margin-top:10px; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; background:#fafafa;">'
                            '<div style="font-weight:700; color:#4f46e5; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">References</div>'
                        )
                        for cit in citations:
                            idx = cit["idx"]
                            section = cit.get("section", "Body") or "Body"
                            p_start = cit.get("page_start", "—")
                            p_end = cit.get("page_end", "—")
                            arxiv_id = cit.get("arxiv_id", "")
                            paper_title = cit.get("paper_title", "") or ""
                            sim = cit.get("similarity_score", 0.0) or 0.0
                            title_short = paper_title[:55] + "…" if len(paper_title) > 55 else paper_title

                            if arxiv_id and not arxiv_id.startswith("temp_"):
                                link_html = f'<a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" style="color: #4f46e5; text-decoration: none; font-weight: 500;">{title_short or arxiv_id}</a>'
                            else:
                                link_html = f'<em>{title_short or "Uploaded Paper"}</em>'

                            ref_line = f'<div style="font-size:0.82rem; color:#0f172a; margin-top:5px; padding-left:4px;"><strong>[{idx}]</strong> {link_html} &bull; &sect;{section} &bull; pp.{p_start}&ndash;{p_end} &bull; sim:{sim:.2f}</div>'
                            ref_html.append(ref_line)
                        ref_html.append('</div>')
                        st.markdown("\n".join(ref_html), unsafe_allow_html=True)
                    
                    # --- Supporting Chunks Expander ---
                    if msg.get("supporting_chunks"):
                        with st.expander("Show supporting chunks"):
                            for c_idx, ch in enumerate(msg["supporting_chunks"], 1):
                                st.markdown(f"**Chunk {c_idx}**")
                                if "similarity" in ch:
                                    st.markdown(f"*Similarity: `{ch['similarity']:.4f}` | Pages: `{ch['pages']}` | Section: `{ch.get('section', 'Body')}`*")
                                else:
                                    st.markdown(f"*Pages: `{ch['pages']}` | Section: `{ch.get('section', 'Body')}`*")
                                preview_text = ch['text'][:150] + "..." if len(ch['text']) > 150 else ch['text']
                                st.markdown(f"**Preview**: *{preview_text}*")
                                with st.expander("Expand full text"):
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; font-size: 0.85rem; color:#0f172a; text-align: justify;">
                                            {ch['text']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                st.markdown("---")

                                
        user_msg = st.chat_input("Ask a question about this paper...")
        if st.session_state.clicked_suggested:
            user_msg = st.session_state.clicked_suggested
            st.session_state.clicked_suggested = None
            
        if user_msg:
            handle_chat_query(user_msg)
            st.rerun()
            
    with col_ingest:
        st.markdown("### Permanent Ingestion")
        with st.expander("Configure Metadata details", expanded=True):
            m_title = st.text_input("Title", value=metadata.get("title", ""))
            m_arxiv = st.text_input("ArXiv ID (if any)", value=metadata.get("arxiv_id", ""))
            
            authors_list = metadata.get("authors", [])
            if isinstance(authors_list, str):
                authors_list = [a.strip() for a in authors_list.split(",") if a.strip()]
            authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
            m_authors = st.text_input("Authors (comma-separated)", value=authors_str)
            m_abstract = st.text_area("Abstract", value=metadata.get("abstract", ""))
            
            cats_list = metadata.get("categories", ["cs.AI"])
            if isinstance(cats_list, str):
                cats_list = [cats_list]
            m_category = st.text_input("Primary Category", value=cats_list[0] if cats_list else "cs.AI")
            
        if "last_ingested_paper_id" not in st.session_state:
            st.session_state["last_ingested_paper_id"] = None

        if st.button("Add to Corpus", use_container_width=True):
            with st.spinner("Integrating paper into permanent corpus..."):
                paper_id_used = m_arxiv.strip() if m_arxiv.strip() else None
                if integrate_into_corpus(m_title, m_arxiv, m_authors, m_abstract, m_category, extracted_text_path, chunks):
                    st.success("Permanent ingestion pipeline completed. Document integrated.")
                    st.session_state["last_ingested_paper_id"] = paper_id_used or m_title[:30]
                else:
                    st.error("Failed to integrate document.")

        # --- Corpus Verification Panel ---
        st.markdown("---")
        verify_id = st.session_state.get("last_ingested_paper_id") or ((m_arxiv or "").strip() or (m_title or "")[:30] or None)
        if verify_id and st.button("Check Corpus Status", use_container_width=True):
            import json as _json, os as _os, faiss as _faiss
            checks = []

            # 1. papers_text.json
            try:
                with open("data/processed/papers_text.json", "r", encoding="utf-8") as _f:
                    _pt = _json.load(_f)
                found_text = any(
                    (isinstance(p, dict) and p.get("arxiv_id") == verify_id) or
                    (isinstance(p, str) and verify_id in p)
                    for p in _pt
                )
                checks.append(("papers_text.json", found_text, f"{len(_pt)} papers stored"))
            except Exception as _e:
                checks.append(("papers_text.json", False, str(_e)))

            # 2. chunks.json
            try:
                with open("data/processed/chunks.json", "r", encoding="utf-8") as _f:
                    _ch = _json.load(_f)
                paper_chunks = [c for c in _ch if isinstance(c, dict) and c.get("arxiv_id") == verify_id]
                checks.append(("chunks.json", len(paper_chunks) > 0, f"{len(paper_chunks)} chunks for this paper ({len(_ch)} total)"))
            except Exception as _e:
                checks.append(("chunks.json", False, str(_e)))

            # 3. graph_data.json
            try:
                with open("data/processed/graph_data.json", "r", encoding="utf-8") as _f:
                    _gd = _json.load(_f)
                paper_node = next((e for e in _gd.get("entities", []) if isinstance(e, dict) and e.get("entity_id") == verify_id), None)
                checks.append(("graph_data.json (Paper node)", paper_node is not None, f"Node: {paper_node.get('name', '?')[:50] if paper_node else 'NOT FOUND'}"))
            except Exception as _e:
                checks.append(("graph_data.json", False, str(_e)))

            # 4. FAISS index chunk count
            try:
                with open("data/vectorstore/chunk_metadata.json", "r", encoding="utf-8") as _f:
                    _cm = _json.load(_f)
                faiss_chunks = [c for c in _cm if isinstance(c, dict) and c.get("arxiv_id") == verify_id]
                _idx = _faiss.read_index("data/vectorstore/faiss.index")
                checks.append(("FAISS index", len(faiss_chunks) > 0, f"{len(faiss_chunks)} vectors for this paper ({_idx.ntotal} total in index)"))
            except Exception as _e:
                checks.append(("FAISS index", False, str(_e)))

            # Render results
            all_ok = all(ok for _, ok, _ in checks)
            st.markdown(
                f"""<div style="border:1px solid {'#bbf7d0' if all_ok else '#fecaca'}; border-radius:10px; padding:14px 16px; background:{'#f0fdf4' if all_ok else '#fff1f2'}; margin-top:8px;">
                <div style="font-weight:700; color:{'#16a34a' if all_ok else '#dc2626'}; font-family:'Outfit',sans-serif; margin-bottom:10px; font-size:0.95rem;">
                    {'✅ Paper is fully integrated into corpus' if all_ok else '⚠️ Some integration checks failed'}
                </div>""",
                unsafe_allow_html=True
            )
            for name, ok, detail in checks:
                icon = "✅" if ok else "❌"
                color = "#16a34a" if ok else "#dc2626"
                st.markdown(
                    f"""<div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:6px; font-size:0.82rem;">
                        <span style="color:{color}; font-size:1rem; flex-shrink:0;">{icon}</span>
                        <span><strong style="color:#0f172a;">{name}</strong><br/>
                        <span style="color:#64748b;">{detail}</span></span>
                    </div>""",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            if not all_ok:
                st.caption(f"Paper ID checked: `{verify_id}`")

else:
    st.info("No active paper session. Please upload a scientific research PDF above to start.")

# Footer
render_footer()
