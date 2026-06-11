import streamlit as st
import numpy as np
import os
import sys

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_css, get_papers_cache, load_json_cache, render_footer
from sentence_transformers import SentenceTransformer

# Set Streamlit Page Config
st.set_page_config(
    page_title="Paper Explorer",
    page_icon=None,
    layout="wide"
)

# Load layout styles
load_css()

st.markdown("<h1>Paper Explorer</h1>", unsafe_allow_html=True)
st.write("Browse papers, inspect page-level text chunks, view local graph neighbors, and get related paper recommendations with text-based reasoning.")

# Load papers cache
papers = get_papers_cache()

# Cache embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

# Precompute abstract embeddings for all papers
@st.cache_data
def precompute_embeddings(_model, papers_cache):
    embeddings_map = {}
    ids = list(papers_cache.keys())
    if not ids:
        return embeddings_map
        
    abstracts = [papers_cache[aid]["abstract"] for aid in ids]
    encoded = _model.encode(abstracts, show_progress_bar=False, normalize_embeddings=True)
    
    for idx, aid in enumerate(ids):
        embeddings_map[aid] = encoded[idx]
    return embeddings_map

if not papers:
    st.markdown(
        """
        <div class="glass-card" style="border-left: 5px solid #dc2626; padding: 20px;">
            <h5 style="color: #dc2626; margin-top: 0;">UI cache missing</h5>
            <p>Please run the initialization script to precompute application caches:</p>
            <code>python scripts/prepare_ui_cache.py</code>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    model = get_embedding_model()
    embeddings_map = precompute_embeddings(model, papers)
    
    paper_ids = sorted(list(papers.keys()))
    
    # Filter search block
    st.markdown("### Search Filters")
    col_s1, col_s2 = st.columns([2, 1])
    
    with col_s1:
        search_query = st.text_input("Search papers...", placeholder="Search by title, author name, keyword, arXiv ID...")
    with col_s2:
        year_filter = st.selectbox("Filter by publication year", ["All"] + sorted(list({str(p.get("year", "")) for p in papers.values() if p.get("year")}), reverse=True))
    
    # Run filters
    filtered_ids = []
    q = search_query.lower().strip()
    
    for aid in paper_ids:
        p = papers[aid]
        title = p.get("title", "").lower()
        abstract = p.get("abstract", "").lower()
        published = p.get("published", "")
        year = str(p.get("year", "")) if "year" in p else (published[:4] if published else "")
        arxiv_id = p.get("arxiv_id", "").lower()
        
        # Extract authors
        authors = [rel["source_name"].lower() for rel in p["relationships"] if rel["relation"] == "WRITES"]
        authors_str = " ".join(authors)
        
        # Check text search matches
        text_match = (not q) or (q in title or q in abstract or q in arxiv_id or q in authors_str)
        # Check year match
        year_match = (year_filter == "All") or (year == year_filter)
        
        if text_match and year_match:
            filtered_ids.append(aid)
            
    if not filtered_ids:
        st.warning("No papers match your search filters.")
    else:
        # Paper selection dropdown
        paper_options = {aid: f"{papers[aid]['title']} ({aid})" for aid in filtered_ids}
        selected_id = st.selectbox(
            "Select a paper to explore:",
            filtered_ids,
            format_func=lambda x: paper_options[x]
        )
        
        if selected_id:
            p_data = papers[selected_id]
            published = p_data.get("published", "")
            year = str(p_data.get("year", "")) if "year" in p_data else (published[:4] if published else "N/A")
            
            # Authors list
            authors = [rel["source_name"] for rel in p_data["relationships"] if rel["relation"] == "WRITES"]
            
            # Header Card
            st.markdown(
                f"""
                <div class="glass-card" style="margin-top: 15px; margin-bottom: 20px;">
                    <span class="badge badge-paper">{p_data['primary_category']}</span>
                    <h2 style="margin: 8px 0 10px 0; color: #4f46e5; font-family: 'Outfit', sans-serif;">{p_data['title']}</h2>
                    <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 12px; line-height: 1.5;">
                        <strong>Authors</strong>: {", ".join(authors) if authors else "Unknown"}<br/>
                        <strong>Publication Year</strong>: {year} &bull; 
                        <strong>ArXiv ID</strong>: {p_data['arxiv_id']} &bull; 
                        <a href="{p_data['pdf_url']}" target="_blank" style="color:#4f46e5; text-decoration:none; font-weight:600;">Open on arXiv</a>
                    </p>
                    <h5 style="margin-bottom: 6px; color: #0f172a; font-family: 'Outfit', sans-serif;">Abstract</h5>
                    <p style="font-size: 0.95rem; line-height: 1.6; color: #334155; text-align: justify; margin: 0;">{p_data['abstract']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Subgraph/Neighbors list
            connected_methods = set()
            connected_datasets = set()
            connected_concepts = set()
            
            for rel in p_data["relationships"]:
                t_type = rel.get("target_type") or rel.get("source_type")
                t_name = rel.get("target_name") or rel.get("source_name")
                if t_type == "Method":
                    connected_methods.add(t_name)
                elif t_type == "Dataset":
                    connected_datasets.add(t_name)
                elif t_type == "Concept":
                    connected_concepts.add(t_name)
            
            # Display Neighbors
            st.markdown("### Graph Neighbors")
            cn1, cn2, cn3 = st.columns(3)
            with cn1:
                st.markdown("##### Connected Methods")
                if connected_methods:
                    badges_html = "".join([f'<span class="badge badge-method" style="margin-right:6px; margin-bottom:6px;">{m}</span>' for m in sorted(connected_methods)])
                    st.markdown(f'<div style="display:flex; flex-wrap:wrap; margin-top:5px;">{badges_html}</div>', unsafe_allow_html=True)
                else:
                    st.write("None registered in database")
            with cn2:
                st.markdown("##### Connected Datasets")
                if connected_datasets:
                    badges_html = "".join([f'<span class="badge badge-dataset" style="margin-right:6px; margin-bottom:6px;">{d}</span>' for d in sorted(connected_datasets)])
                    st.markdown(f'<div style="display:flex; flex-wrap:wrap; margin-top:5px;">{badges_html}</div>', unsafe_allow_html=True)
                else:
                    st.write("None registered in database")
            with cn3:
                st.markdown("##### Connected Concepts")
                if connected_concepts:
                    badges_html = "".join([f'<span class="badge badge-concept" style="margin-right:6px; margin-bottom:6px;">{c}</span>' for c in sorted(connected_concepts)])
                    st.markdown(f'<div style="display:flex; flex-wrap:wrap; margin-top:5px;">{badges_html}</div>', unsafe_allow_html=True)
                else:
                    st.write("None registered in database")
            
            st.write("")
            
            # Tabs for Chunks and Recommendations
            tab_rec, tab_chk = st.tabs(["Related Papers", "Chunk Explorer"])
            
            with tab_rec:
                st.markdown("#### Related Papers Recommendation Engine (Top 10)")
                
                # Fetch details for recommendations
                ref_emb = embeddings_map[selected_id]
                
                def get_neighbors(aid):
                    return {rel.get("source") or rel.get("target") for rel in papers[aid]["relationships"]}
                    
                def get_methods_and_datasets_ids(aid):
                    nodes = set()
                    for rel in papers[aid]["relationships"]:
                        t_type = rel.get("target_type") or rel.get("source_type")
                        t_id = rel.get("target") or rel.get("source")
                        if t_type in ["Method", "Dataset"]:
                            nodes.add(t_id)
                    return nodes
                
                ref_neighbors = get_neighbors(selected_id)
                ref_md_ids = get_methods_and_datasets_ids(selected_id)
                
                ref_methods = {rel.get("target_name") or rel.get("source_name") for rel in papers[selected_id]["relationships"] if (rel.get("target_type") == "Method" or rel.get("source_type") == "Method")}
                ref_datasets = {rel.get("target_name") or rel.get("source_name") for rel in papers[selected_id]["relationships"] if (rel.get("target_type") == "Dataset" or rel.get("source_type") == "Dataset")}
                
                recommendations = []
                
                for candidate_id in paper_ids:
                    if candidate_id == selected_id:
                        continue
                        
                    cand_data = papers[candidate_id]
                    
                    # 1. Cosine similarity
                    cand_emb = embeddings_map[candidate_id]
                    cos_sim = float(np.dot(ref_emb, cand_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb)))
                    
                    # 2. Graph neighbor overlap
                    cand_neighbors = get_neighbors(candidate_id)
                    union_neighbors = ref_neighbors | cand_neighbors
                    graph_overlap = len(ref_neighbors & cand_neighbors) / len(union_neighbors) if union_neighbors else 0.0
                    
                    # 3. Shared methods & datasets Jaccard
                    cand_md_ids = get_methods_and_datasets_ids(candidate_id)
                    union_md = ref_md_ids | cand_md_ids
                    shared_md = len(ref_md_ids & cand_md_ids) / len(union_md) if union_md else 0.0
                    
                    # Calculate hybrid score
                    hybrid_score = 0.5 * cos_sim + 0.3 * graph_overlap + 0.2 * shared_md
                    
                    # Textual reasons compiling
                    reasons = []
                    if cos_sim >= 0.72:
                        reasons.append("✓ Similar embeddings")
                    
                    # Shares category
                    from utils import classify_paper
                    ref_cat = classify_paper(papers[selected_id])
                    cand_cat = classify_paper(cand_data)
                    if ref_cat == cand_cat:
                        reasons.append(f"✓ Shares {ref_cat} category")
                        
                    cand_methods = {rel.get("target_name") or rel.get("source_name") for rel in papers[candidate_id]["relationships"] if (rel.get("target_type") == "Method" or rel.get("source_type") == "Method")}
                    shared_m_names = ref_methods & cand_methods
                    for m in sorted(list(shared_m_names)):
                        reasons.append(f"✓ Shares {m} method")
                        
                    cand_datasets = {rel.get("target_name") or rel.get("source_name") for rel in papers[candidate_id]["relationships"] if (rel.get("target_type") == "Dataset" or rel.get("source_type") == "Dataset")}
                    shared_d_names = ref_datasets & cand_datasets
                    for d in sorted(list(shared_d_names)):
                        reasons.append(f"✓ Shares {d} dataset")
                        
                    ref_concepts = {rel.get("target_name") or rel.get("source_name") for rel in papers[selected_id]["relationships"] if (rel.get("target_type") == "Concept" or rel.get("source_type") == "Concept")}
                    cand_concepts = {rel.get("target_name") or rel.get("source_name") for rel in papers[candidate_id]["relationships"] if (rel.get("target_type") == "Concept" or rel.get("source_type") == "Concept")}
                    shared_c_names = ref_concepts & cand_concepts
                    for c in sorted(list(shared_c_names)):
                        reasons.append(f"✓ Connected through {c} node")
                        
                    if not reasons:
                        reasons.append("✓ General topic similarity")
                    
                    recommendations.append({
                        "arxiv_id": candidate_id,
                        "title": cand_data["title"],
                        "hybrid_score": hybrid_score,
                        "reasons": reasons
                    })
                    
                recommendations.sort(key=lambda x: x["hybrid_score"], reverse=True)
                
                for idx, rec in enumerate(recommendations[:10], 1):
                    reasons_html = "".join([f"<li style='margin-bottom: 3px; color: #059669; font-weight: 500;'>{r}</li>" for r in rec["reasons"]])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom: 12px; border-left: 4px solid #059669; padding: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h5 style="margin: 0; color: #0f172a; font-family: 'Outfit', sans-serif;">{idx}. {rec['title']}</h5>
                                <span style="font-size: 1.15rem; font-weight: 700; color: #059669;">{rec['hybrid_score']:.4f}</span>
                            </div>
                            <span class="stat-label" style="font-size: 0.75rem;">ArXiv ID: {rec['arxiv_id']}</span>
                            <div style="margin-top: 8px; font-size: 0.85rem; color: #334155;">
                                <strong style="color: #64748b;">Recommendation Reasons:</strong>
                                <ul style="margin: 4px 0 0 0; padding-left: 15px; list-style-type: none;">
                                    {reasons_html if reasons_html else "<li>&bull; General topic similarity</li>"}
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            with tab_chk:
                st.markdown(f"#### Retrievable Text Chunks ({len(p_data['chunks'])} chunks)")
                for idx, chunk in enumerate(p_data["chunks"]):
                    with st.expander(f"Chunk {idx+1} — Section: {chunk['section']} | Page Range: {chunk['page_start']}-{chunk['page_end']}"):
                        st.markdown(
                            f"""
                            <div style="font-size: 0.95rem; line-height: 1.5; color: #0f172a; text-align: justify; padding: 8px 0;">
                                {chunk['text']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# Footer
render_footer()
