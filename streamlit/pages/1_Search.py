import streamlit as st
import os
import sys

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    load_css, render_footer, render_llm_answer, renumber_citations,
    render_citations, render_confidence, render_abstention,
    render_metrics, render_chunks
)
from search_modules.search_service import execute_search_query

# Set Streamlit Page Config
st.set_page_config(
    page_title="GraphRAG Search",
    layout="wide"
)

# Load layout styles
load_css()

# Banner
st.markdown("<h1>GraphRAG Search</h1>", unsafe_allow_html=True)
st.write("Query the hybrid GraphRAG pipeline to retrieve vector passages and subgraphs, and synthesize grounded answers with verified citations.")

query = st.text_input("Enter your research question...", placeholder="e.g. How does Prefix Tuning differ from LoRA?")

if query:
    with st.spinner("Processing query..."):
        try:
            res, latency = execute_search_query(query)
            
            # Renumber citations sequentially
            renumbered_answer, renumbered_citations = renumber_citations(res.answer, res.citations)
            
            # 1. Answer display
            st.markdown("### Answer")
            border_color = "#ea580c" if res.abstained else "#4f46e5"
            st.markdown(f'<div class="glass-card" style="border-left: 5px solid {border_color}; padding: 24px; margin-bottom: 25px;">', unsafe_allow_html=True)
            render_llm_answer(renumbered_answer)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if res.abstained:
                render_abstention(res.confidence, 0.45, res.metadata.get("abstention_reason"))
            
            # 2. Metrics and Stats grid
            c_conf, c_stats, c_retrieved = st.columns([1.5, 1.5, 2.0])
            
            with c_conf:
                render_confidence(res.confidence, res.abstained)
                
            with c_stats:
                metrics_data = {
                    "provider_used": res.metadata.get("provider_used", "N/A"),
                    "generation_model": res.metadata.get("generation_model", "N/A"),
                    "total_execution_time_ms": res.metadata.get("total_execution_time_ms", latency * 1000.0),
                    "tokens": res.metadata.get("tokens", {}),
                    "cost": res.metadata.get("cost", 0.0),
                    "fallback_used": res.metadata.get("fallback_used", False),
                    "fallback_reason": res.metadata.get("fallback_reason")
                }
                render_metrics(metrics_data)
                
            with c_retrieved:
                chunks_count = res.provenance.get("chunks_used", 0)
                nodes_count = res.provenance.get("graph_nodes_used", 0)
                papers_count = res.provenance.get("papers_used", 0)
                citations_count = len(renumbered_citations)
                
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 16px; min-height: 140px;">
                        <div style="font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">Retrieved Context</div>
                        <div style="font-size: 0.85rem; color: #0f172a; line-height: 1.6;">
                            <div style="display: flex; justify-content: space-between;"><span>Vector Chunks</span><span style="font-weight: bold;">{chunks_count}</span></div>
                            <div style="display: flex; justify-content: space-between;"><span>Knowledge Graph Nodes</span><span style="font-weight: bold;">{nodes_count}</span></div>
                            <div style="display: flex; justify-content: space-between;"><span>Source Papers</span><span style="font-weight: bold;">{papers_count}</span></div>
                            <div style="display: flex; justify-content: space-between;"><span>Inline Citations</span><span style="font-weight: bold;">{citations_count}</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # 3. Sources Section
            st.markdown("### Sources")
            render_citations(renumbered_citations)
            
            # 4. Show Retrieval Details Expander
            with st.expander("Show Retrieval Details"):
                st.markdown("#### Execution Metadata")
                ret_meta = res.metadata.get("retrieval_metadata", {})
                
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.write(f"**Intent Class**: `{res.metadata.get('category', 'N/A')}`")
                    st.write(f"**Retrieval Policy**: `{ret_meta.get('routing_strategy', 'N/A')}`")
                    st.write(f"**Evidence Budget**: `{ret_meta.get('token_budget', 0)}` tokens")
                    st.write(f"**Budget Used**: `{ret_meta.get('token_used', 0)}` tokens")
                with col_det2:
                    st.write(f"**Vector Candidates**: `{ret_meta.get('vector_candidates', 0)}` chunks")
                    st.write(f"**Graph Expansion Nodes**: `{ret_meta.get('graph_nodes', 0)}` nodes")
                    st.write(f"**Graph Expansion Edges**: `{ret_meta.get('graph_relationships', 0)}` relationships")
                    st.write(f"**Search Latency**: `{ret_meta.get('fusion_time_ms', 0.0):.2f}ms` (retrieval only)")
                    
                st.markdown("##### Confidence Breakdown Details")
                breakdown = res.metadata.get("retrieval_metrics", {})
                st.write(f"- **Average Vector Similarity**: `{breakdown.get('avg_similarity', 0.0):.4f}`")
                st.write(f"- **Graph Connectivity**: `{breakdown.get('graph_connectivity', 0.0):.4f}`")
                st.write(f"- **Citation Coverage**: `{breakdown.get('citation_coverage', 0.0):.4f}`")
                
                st.markdown("##### Retrieved Context Passages (Highlighted)")
                eval_ctx = res.evaluation_context
                if eval_ctx and eval_ctx.vector_context:
                    render_chunks(eval_ctx.vector_context, query)
                else:
                    st.info("No retrieved chunks loaded in evaluation context.")
                    
        except Exception as e:
            st.error(f"Error executing search query: {e}")
            st.info("Ensure that Neo4j and FAISS databases are fully loaded.")

# Footer
render_footer()
