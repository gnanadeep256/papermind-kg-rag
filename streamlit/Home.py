import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import load_css, get_project_insights, load_json_cache, classify_paper, render_footer, get_methods_cache, get_datasets_cache

# Set Streamlit Page Config
st.set_page_config(
    page_title="PaperMind GraphRAG Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load layout styles
load_css()

# Main Banner Layout
st.markdown(
    """
    <div style="padding: 10px 0 20px 0; margin-bottom: 10px;">
        <h1 style="margin: 0;">PaperMind GraphRAG</h1>
        <p style="font-size: 1.15rem; color: #64748b; margin: 5px 0 0 0; font-family: 'Outfit', sans-serif;">
            Next-generation scientific research assistant powered by a hybrid Vector and Knowledge Graph retrieval pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load Data Caches
papers_cache = load_json_cache("papers_cache.json") or {}
graph_stats = load_json_cache("graph_stats.json") or {}
stats_meta = graph_stats.get("metadata", {})
node_metrics = graph_stats.get("node_metrics", [])
chunk_meta = load_json_cache("chunk_metadata.json") or []
methods_cache = get_methods_cache()
datasets_cache = get_datasets_cache()
insights = get_project_insights()

# Calculate stats
total_papers = len(papers_cache)
total_chunks = len(chunk_meta)
total_nodes = stats_meta.get("total_nodes", 0)
total_edges = stats_meta.get("total_edges", 0)
total_methods = len(methods_cache)
total_datasets = len(datasets_cache)
total_authors = sum(1 for n in node_metrics if n.get("entity_type") == "Author")

# Metric cards
st.markdown("### Repository Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_papers}</div>
            <div class="stat-label">Total Indexed Papers</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_chunks}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_authors}</div>
            <div class="stat-label">Total Authors</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_nodes}</div>
            <div class="stat-label">Knowledge Graph Nodes</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_edges}</div>
            <div class="stat-label">Graph Relationships</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">BAAI/bge-small-en-v1.5</div>
            <div class="stat-label">Embedding Model</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_methods}</div>
            <div class="stat-label">Total Methods</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">{total_datasets}</div>
            <div class="stat-label">Total Datasets</div>
        </div>
        <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
            <div class="stat-val">Gemini 2.5 Flash / Groq Llama-3</div>
            <div class="stat-label">LLM Provider</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Architecture Section
st.markdown("### System Architecture")
st.markdown(
    """
    <div class="glass-card" style="margin-bottom: 20px; padding: 20px;">
        <h5 style="margin-top:0; color: #4f46e5;">End-to-End Execution Pipeline</h5>
        <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 20px;">
            How PaperMind ingests scientific PDFs, constructs the hybrid indices, retrieves grounding context, and synthesizes citation-backed answers:
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; margin-top: 15px;">
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">PDFs</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">Chunking</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">Embedding</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">FAISS + Neo4j</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">Hybrid Retrieval</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">Evidence Packing</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">LLM</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #0f172a;">Citation Validation</div>
            <div style="color: #94a3b8; font-weight: bold;">&rarr;</div>
            <div style="background: #e0e7ff; border: 1px solid #6366f1; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-weight: 600; font-size: 0.85rem; color: #4f46e5;">Grounded Answer</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Helper to format HTML Tables
def make_html_table(headers, rows):
    header_html = "".join([f"<th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #4f46e5; font-family: \"Outfit\", sans-serif;'>{h}</th>" for h in headers])
    rows_html = ""
    for r in rows:
        cells = "".join([f"<td style='padding: 10px 8px; border-bottom: 1px solid #f1f5f9; color: #0f172a;'>{c}</td>" for c in r])
        rows_html += f"<tr>{cells}</tr>"
    return f"<table style='width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 10px;'><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"

# Example Questions Section
st.markdown("### Example Questions to Try")
st.markdown(
    """
    <div class="glass-card" style="margin-bottom: 20px; padding: 20px;">
        <p style="font-size: 0.9rem; color: #64748b; margin-top: 0; margin-bottom: 15px;">
            Copy and paste these queries into the <strong>Search</strong> page to test the hybrid RAG execution pipeline:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.9rem; font-family: 'Outfit', sans-serif; color: #0f172a;">
                <strong style="color: #4f46e5;">Conceptual Queries</strong><br/>
                &bull; What is LoRA?<br/>
                &bull; Explain GraphRAG
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.9rem; font-family: 'Outfit', sans-serif; color: #0f172a;">
                <strong style="color: #4f46e5;">Comparative Queries</strong><br/>
                &bull; Compare BERT and Transformer<br/>
                &bull; Compare Prefix Tuning and LoRA
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.9rem; font-family: 'Outfit', sans-serif; color: #0f172a;">
                <strong style="color: #4f46e5;">Topological Queries</strong><br/>
                &bull; Which papers introduced PEFT?<br/>
                &bull; What datasets evaluate USAD?
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Corpus Insights Block
st.markdown("### Corpus Insights")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Topological Insights")
    
    # 1. Top Methods by PageRank
    methods_list = [n for n in node_metrics if n.get("entity_type") == "Method"]
    methods_list.sort(key=lambda x: x.get("pagerank", 0), reverse=True)
    m_rows = [[idx, f"<strong>{item['name']}</strong>", f"{item['pagerank']:.5f}"] for idx, item in enumerate(methods_list[:5], 1)]
    st.markdown("##### Top Methods (PageRank)")
    st.markdown(make_html_table(["Rank", "Method Name", "PageRank Score"], m_rows), unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Top Datasets by PageRank
    datasets_list = [n for n in node_metrics if n.get("entity_type") == "Dataset"]
    datasets_list.sort(key=lambda x: x.get("pagerank", 0), reverse=True)
    d_rows = [[idx, f"<strong>{item['name']}</strong>", f"{item['pagerank']:.5f}"] for idx, item in enumerate(datasets_list[:5], 1)]
    st.markdown("##### Top Datasets (PageRank)")
    st.markdown(make_html_table(["Rank", "Dataset Name", "PageRank Score"], d_rows), unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 3. Most Connected Papers
    papers_list = [n for n in node_metrics if n.get("entity_type") == "Paper"]
    papers_list.sort(key=lambda x: x.get("raw_degree", 0), reverse=True)
    p_rows = [[idx, f"<strong>{item['name']}</strong>", item.get("raw_degree", 0)] for idx, item in enumerate(papers_list[:5], 1)]
    st.markdown("##### Most Connected Papers (Degree)")
    st.markdown(make_html_table(["Rank", "Paper Title", "Connections (Degree)"], p_rows), unsafe_allow_html=True)

with col_right:
    st.markdown("#### Chronological Insights")
    
    # Sort papers by date
    papers_sorted = []
    for aid, p in papers_cache.items():
        pub = p.get("published", "")
        year = p.get("year", 2023)
        papers_sorted.append({
            "arxiv_id": aid,
            "title": p.get("title", ""),
            "published": pub,
            "year": year
        })
    
    # 4. Newest Papers
    newest = sorted(papers_sorted, key=lambda x: x["published"], reverse=True)
    new_rows = [[idx, f"<strong>{item['title'][:45]}...</strong>", item["published"][:10], item["arxiv_id"]] for idx, item in enumerate(newest[:5], 1)]
    st.markdown("##### Newest Additions")
    st.markdown(make_html_table(["Index", "Paper Title", "Published Date", "arXiv ID"], new_rows), unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 5. Oldest Papers
    oldest = sorted(papers_sorted, key=lambda x: x["published"])
    old_rows = [[idx, f"<strong>{item['title'][:45]}...</strong>", item["published"][:10], item["arxiv_id"]] for idx, item in enumerate(oldest[:5], 1)]
    st.markdown("##### Foundation Papers (Oldest)")
    st.markdown(make_html_table(["Index", "Paper Title", "Published Date", "arXiv ID"], old_rows), unsafe_allow_html=True)

# Charts Section
st.markdown("<br/>### Repository Analysis Charts", unsafe_allow_html=True)

col_c1, col_c2 = st.columns(2)

with col_c1:
    # 1. Timeline Chart
    years_count = {}
    for aid, p in papers_cache.items():
        year = p.get("year")
        if not year and p.get("published"):
            year = int(p["published"][:4])
        if year:
            years_count[year] = years_count.get(year, 0) + 1
            
    df_years = pd.DataFrame(list(years_count.items()), columns=["Year", "Papers Count"]).sort_values("Year")
    fig_timeline = px.bar(
        df_years, x="Year", y="Papers Count",
        title="Publication Timeline Distribution",
        color_discrete_sequence=["#4f46e5"],
        labels={"Year": "Publication Year", "Papers Count": "Number of Papers"}
    )
    fig_timeline.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Plus Jakarta Sans",
        xaxis=dict(gridcolor="#e2e8f0", type='category'),
        yaxis=dict(gridcolor="#e2e8f0"),
        title_font_family="Outfit",
        title_font_size=16
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_c2:
    # 2. Category Distribution Chart (using primary_category + classify_paper)
    cat_count = {}
    for aid, p in papers_cache.items():
        cat = classify_paper(p)
        cat_count[cat] = cat_count.get(cat, 0) + 1
        
    df_cats = pd.DataFrame(list(cat_count.items()), columns=["Category", "Papers Count"]).sort_values("Papers Count", ascending=False)
    fig_cats = px.pie(
        df_cats, names="Category", values="Papers Count",
        title="Research Category Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_cats.update_layout(
        font_family="Plus Jakarta Sans",
        title_font_family="Outfit",
        title_font_size=16
    )
    st.plotly_chart(fig_cats, use_container_width=True)

# Render unified footer
render_footer()
