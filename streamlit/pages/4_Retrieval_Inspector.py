import streamlit as st
import os
import sys
import gzip
import json
import time
import uuid
import plotly.graph_objects as go

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_css, render_footer, markdown_to_html, render_chunks, render_llm_answer, render_citations, renumber_citations
from src.hybrid_retriever import HybridRetriever
from src.answer_generator import GroundedAnswerGenerator
from src.observability.trace_models import TraceContext

# Set Page Config
st.set_page_config(
    page_title="Retrieval Inspector",
    layout="wide"
)

# Load layout styles
load_css()

st.markdown("<h1>Retrieval Inspector</h1>", unsafe_allow_html=True)
st.write("Flagship debug interface. Visualize the step-by-step GraphRAG execution pipeline, inspect retrieval scores, and analyze citation alignment.")

# Initialize generator
@st.cache_resource
def get_generator():
    retriever = HybridRetriever()
    retriever.load()
    return GroundedAnswerGenerator(retriever)

try:
    generator = get_generator()
    generator_loaded = True
except Exception as e:
    st.error(f"Failed to load GroundedAnswerGenerator: {e}")
    generator_loaded = False

def render_trace_context(q_trace: dict):
    """Renders the detailed TraceContext schema in a beautiful, structured layout."""
    query_info = q_trace.get("query", {}) or {}
    ret_info = q_trace.get("retrieval", {}) or {}
    gen_info = q_trace.get("generation", {}) or {}
    conf_info = q_trace.get("confidence", {}) or {}
    policy_info = q_trace.get("policy", {}) or {}
    cit_info = q_trace.get("citation", {}) or {}
    
    # 1. Metric Overview Row
    st.markdown("### Trace Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Detected Category", query_info.get("detected_intent", "N/A").upper())
    with c2:
        st.metric("Overall Confidence", f"{conf_info.get('final_confidence', 0.0) * 100:.1f}%")
    with c3:
        st.metric("LLM Model Used", f"{gen_info.get('model_id', 'N/A')} ({gen_info.get('provider_used', 'N/A')})")
    with c4:
        st.metric("Execution Cost", f"${gen_info.get('estimated_cost', 0.0):.6f}")
        
    # 2. Gauges & Waterfall side-by-side
    st.markdown("#### Confidence Breakdown & Performance Waterfall")
    col_g, col_wf = st.columns([1, 1])
    
    with col_g:
        fig_gauges = go.Figure()
        breakdown = conf_info.get("confidence_breakdown", {}) or {}
        
        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=breakdown.get("semantic", 0.0) * 100,
            title={'text': "Semantic Match", 'font': {'size': 11, 'family': 'Outfit'}},
            domain={'x': [0, 0.45], 'y': [0.55, 1.0]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#4f46e5"}, 'borderwidth': 1}
        ))
        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=breakdown.get("graph", 0.0) * 100,
            title={'text': "Graph Connection", 'font': {'size': 11, 'family': 'Outfit'}},
            domain={'x': [0.55, 1.0], 'y': [0.55, 1.0]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#10b981"}, 'borderwidth': 1}
        ))
        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=cit_info.get("citation_precision", 0.0) * 100 if cit_info.get("citation_precision") is not None else breakdown.get("citation", 0.0) * 100,
            title={'text': "Citation Precision", 'font': {'size': 11, 'family': 'Outfit'}},
            domain={'x': [0, 0.45], 'y': [0, 0.45]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#f97316"}, 'borderwidth': 1}
        ))
        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=conf_info.get("final_confidence", 0.0) * 100,
            title={'text': "Overall Confidence", 'font': {'size': 11, 'family': 'Outfit'}},
            domain={'x': [0.55, 1.0], 'y': [0, 0.45]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#db2777"}, 'borderwidth': 1}
        ))
        fig_gauges.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Plus Jakarta Sans"
        )
        st.plotly_chart(fig_gauges, use_container_width=True)
        
    with col_wf:
        wf = q_trace.get("latency_waterfall_ms", {}) or {}
        if wf:
            wf_x = list(wf.keys())
            wf_y = [v if v is not None else 0 for v in wf.values()]
            measures = ["relative"] * (len(wf_x) - 1) + ["total"]
            
            fig_wf = go.Figure(go.Waterfall(
                name="Stage Latency",
                orientation="v",
                measure=measures,
                x=wf_x,
                textposition="outside",
                text=[f"+{int(v)}ms" for v in wf_y[:-1]] + [f"{int(wf_y[-1])}ms"],
                y=wf_y,
                connector={"line":{"color":"#cbd5e1"}},
                decreasing={"marker":{"color":"#dc2626"}},
                increasing={"marker":{"color":"#4f46e5"}},
                totals={"marker":{"color":"#db2777"}}
            ))
            fig_wf.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Plus Jakarta Sans",
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("No latency waterfall profile available in this trace.")
            
    # 3. Stage-by-Stage Details Expanders
    st.markdown("#### Pipeline Stage Execution Details")
    
    with st.expander("Stage 1: Intent Classification & Routing"):
        st.write(f"**User Query**: `{query_info.get('user_query', '')}`")
        st.write(f"**Intent Class**: `{query_info.get('detected_intent', 'N/A')}`")
        st.write(f"**Policy Selected**: `{query_info.get('selected_policy', 'N/A')}`")
        st.write(f"**Timestamp**: `{query_info.get('timestamp', 'N/A')}`")
        st.write(f"**Routing Strategy**: `{policy_info.get('policy_selected', 'N/A')}`")
        
    with st.expander("Stage 2: Hybrid Context Retrieval"):
        st.write(f"**Source Papers Met**: `{ret_info.get('retrieved_papers', [])}`")
        st.write(f"**Vector Chunks Retrieved**: `{len(ret_info.get('retrieved_chunks', []))}` chunks")
        st.write(f"**Knowledge Graph Nodes Expanded**: `{ret_info.get('graph_nodes_count', 0)}` nodes")
        st.write(f"**Knowledge Graph Edges Expanded**: `{ret_info.get('graph_edges_count', 0)}` relationships")
        
        chunks = ret_info.get('retrieved_chunks', [])
        if chunks:
            st.markdown("##### Retrieved Chunk Summaries:")
            for idx, c in enumerate(chunks, 1):
                st.markdown(
                    f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 8px; font-size:0.85rem;">
                        <strong>[{idx}] Section: {c.get('section')}</strong> &bull; Pages: {c.get('page_start')}-{c.get('page_end')} &bull; Paper ID: {c.get('arxiv_id')}<br/>
                        Semantic: <code>{c.get('semantic_score', 0.0):.4f}</code> | Reranker: <code>{c.get('reranker_score', 0.0):.4f}</code> | Graph Bonus: <code>{c.get('graph_bonus', 0.0):.4f}</code> | Combined: <code>{c.get('combined_score', 0.0):.4f}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
    with st.expander("Stage 3: Evidence Packing & Budget Allocation"):
        st.write(f"**Fallback Policy Triggered**: `{policy_info.get('fallback_used', False)}`")
        st.write(f"**Evidence Token Budget**: `{policy_info.get('budget_limit_words', 0)}` words/tokens")
        st.write(f"**Evidence Token Used**: `{policy_info.get('budget_used_words', 0)}` words/tokens")
        st.write(f"**Budget Utilization**: `{policy_info.get('budget_utilization', 0.0) * 100:.1f}%`")
        st.write(f"**Graph Context Overlap Ratio**: `{policy_info.get('graph_overlap_ratio', 0.0):.4f}`")
        
    with st.expander("Stage 4: Prompt Construction & LLM Generation"):
        st.write(f"**LLM Model Used**: `{gen_info.get('model_id', 'N/A')} ({gen_info.get('provider_used', 'N/A')})`")
        st.write(f"**Fallback Model Activated**: `{gen_info.get('fallback_used', False)}` (Reason: `{gen_info.get('fallback_reason')}`)")
        st.write(f"**System Prompt Template Hash**: `{gen_info.get('system_template_hash', 'N/A')}`")
        st.write(f"**Prompt Content Hash**: `{gen_info.get('prompt_hash', 'N/A')}`")
        st.write(f"**Prompt / Completion Tokens**: `{gen_info.get('prompt_tokens_actual')} / {gen_info.get('completion_tokens_actual')}`")
        st.write(f"**Estimated Cost**: `${gen_info.get('estimated_cost', 0.0):.6f}`")
        
        p_text = gen_info.get("prompt_text")
        if p_text:
            st.markdown("##### Assembled LLM Prompt:")
            st.code(p_text)
            
    with st.expander("Stage 5: Citation Validation & Renumbering"):
        st.write(f"**Generated Citations count**: `{cit_info.get('generated_citations_count', 0)}`")
        validated = cit_info.get('validated_citations', [])
        st.write(f"**Validated citations count**: `{len(validated)}`")
        st.write(f"**Citation Precision**: `{cit_info.get('citation_precision', 0.0) * 100:.1f}%`")
        st.write(f"**Citation Coverage**: `{cit_info.get('citation_coverage', 0.0) * 100:.1f}%`")

        rejected = cit_info.get("rejected_citations", [])
        if rejected:
            st.markdown("<span style='color: #dc2626; font-weight: 600;'>Pruned Hallucinated Citations:</span>", unsafe_allow_html=True)
            for r in rejected:
                st.code(r)

        if validated:
            st.markdown("<span style='color: #059669; font-weight: 600;'>Validated Citations:</span>", unsafe_allow_html=True)
            for v in validated:
                arxiv_id = v.get('arxiv_id', '') if isinstance(v, dict) else ''
                section = v.get('section', '?') if isinstance(v, dict) else '?'
                pages = f"{v.get('page_start','?')}-{v.get('page_end','?')}" if isinstance(v, dict) else '?'
                title = v.get('paper_title', arxiv_id) if isinstance(v, dict) else str(v)
                sim = v.get('similarity_score', 0.0) if isinstance(v, dict) else 0.0
                if arxiv_id:
                    st.markdown(f"- **[{title[:50]}](https://arxiv.org/abs/{arxiv_id})** · §{section} · pp.{pages} · sim:{sim:.3f}")
                else:
                    st.markdown(f"- {title[:60]} · §{section} · pp.{pages}")
                
    with st.expander("Stage 6: Hallucination Guard & Semantic Drift"):
        st.write(f"**Semantic Drift Score (Similarity of Answer to Context)**: `{conf_info.get('drift_score', 1.0):.4f}`")
        st.write(f"**Abstention Guard Triggered**: `{conf_info.get('abstention_trigger', False)}`")

# Layout Tabs
tab_live, tab_history = st.tabs(["Live Query Inspector", "Historical Run Inspector"])

if tab_live:
    with tab_live:
        if not generator_loaded:
            st.info("Live query inspector unavailable (Generator failed to load).")
        else:
            st.markdown("### Run Live Trace")
            l_query = st.text_input("Enter query to trace:", placeholder="e.g. How does Prefix Tuning differ from LoRA?", key="live_query_input")
            
            if l_query:
                with st.spinner("Executing query and capturing trace..."):
                    try:
                        # Instantiate our TraceContext directly to capture live traces
                        trace_ctx = TraceContext(
                            experiment_id="standalone",
                            run_id=f"run_{int(time.time())}",
                            query_id=f"q_{int(time.time())}",
                            trace_id=f"trace_{uuid.uuid4().hex[:8]}"
                        )
                        res = generator.generate_answer(l_query, trace_context=trace_ctx)
                        
                        st.markdown("#### Generated Response")
                        border_color = "#ea580c" if res.abstained else "#4f46e5"
                        st.markdown(f'<div class="glass-card" style="border-left: 5px solid {border_color}; padding: 18px 24px; margin-bottom: 20px;">', unsafe_allow_html=True)
                        render_llm_answer(res.answer)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Renumber and show citations as references
                        renumbered_answer, renumbered_citations = renumber_citations(res.answer, res.citations)
                        if renumbered_citations:
                            st.markdown("#### References")
                            render_citations(renumbered_citations)
                        
                        if res.abstained:
                            st.warning(f"**Pipeline Abstained**: {res.metadata.get('abstention_reason')}")
                            
                        # Render trace context details
                        render_trace_context(trace_ctx.model_dump())
                        
                    except Exception as e:
                        st.error(f"Error generating pipeline trace: {e}")
                        st.info("Ensure Neo4j and FAISS are running locally.")

if tab_history:
    with tab_history:
        st.markdown("### Load Historical Traces")
        exp_dir = "reports/experiments"
        if not os.path.exists(exp_dir):
            st.info("No historical experiments directory found.")
        else:
            runs = []
            if os.path.exists(os.path.join(exp_dir, "standalone")):
                runs.append("standalone")
            runs.extend(sorted([d for d in os.listdir(exp_dir) if d.startswith("experiment_")], reverse=True))
            
            if not runs:
                st.info("No runs found in experiments directory.")
            else:
                selected_run = st.selectbox("Select Experiment Run", runs)
                run_path = os.path.join(exp_dir, selected_run)
                
                trace_file = os.path.join(run_path, "traces.jsonl.gz")
                if not os.path.exists(trace_file):
                    st.warning(f"No compressed trace file found in {selected_run}.")
                else:
                    st.success(f"Loaded compressed trace file from {selected_run}.")
                    
                    queries_traces = []
                    try:
                        with gzip.open(trace_file, "rt", encoding="utf-8") as f:
                            for line in f:
                                queries_traces.append(json.loads(line))
                    except Exception as e:
                        st.error(f"Error reading trace file: {e}")
                        
                    if not queries_traces:
                        st.info("No individual query traces found inside this run.")
                    else:
                        query_list = [t.get("query", {}).get("user_query", t.get("query", "Unknown query")) for t in queries_traces]
                        selected_query_idx = st.selectbox(
                            "Select Query to Inspect",
                            range(len(query_list)),
                            format_func=lambda idx: f"{idx+1}. {query_list[idx]}"
                        )
                        
                        q_trace = queries_traces[selected_query_idx]
                        
                        st.markdown(f"### Trace for query: *\"{query_list[selected_query_idx]}\"*")
                        render_trace_context(q_trace)

# Footer
render_footer()
