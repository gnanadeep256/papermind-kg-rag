import streamlit as st
import json
import os
import sys
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Ensure project root and local directory are in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_css, load_json_cache, render_footer

# Set Streamlit Page Config
st.set_page_config(
    page_title="Knowledge Graph",
    page_icon=None,
    layout="wide"
)

# Load layout styles
load_css()

st.markdown("<h1>Interactive Knowledge Graph</h1>", unsafe_allow_html=True)
st.write("Explore the topological structure of the PaperMind knowledge base. Visualize nodes, edges, communities, and PageRank centralities natively.")

# Validation
stats_exists = os.path.exists("data/processed/graph_stats.json")
ents_exists = os.path.exists("data/processed/entities.json")
rels_exists = os.path.exists("data/processed/relationships.json")

if not (stats_exists and ents_exists and rels_exists):
    st.markdown(
        """
        <div class="glass-card" style="border-left: 5px solid #dc2626; padding: 20px;">
            <h5 style="color: #dc2626; margin-top: 0;">Graph data unavailable</h5>
            <p>Please run the graph precomputation script to generate topological analytics files:</p>
            <code>python scripts/generate_graph_stats.py</code>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Load data
    with open("data/processed/entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open("data/processed/relationships.json", "r", encoding="utf-8") as f:
        relationships = json.load(f)
    stats = load_json_cache("graph_stats.json") or {}
    
    stats_meta = stats.get("metadata", {})
    node_metrics = stats.get("node_metrics", [])
    
    # Display loaded counts
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 16px; font-size: 0.85rem; color: #64748b; margin-bottom: 20px;">
            <strong>Cache Status</strong>: Loaded {len(entities):,} entities &bull; {len(relationships):,} semantic relationships from precomputed store.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sort entities by PageRank
    pagerank_lookup = {item["entity_id"]: item["pagerank"] for item in node_metrics}
    
    col_g, col_t = st.columns([1.8, 1.2])
    
    with col_g:
        st.markdown("### Network Visualization")
        
        # Configuration Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            max_nodes = st.select_slider("Maximum nodes to render", options=[100, 250, 500], value=250)
        with col_ctrl2:
            search_node = st.text_input("Search and highlight node...", placeholder="e.g. LoRA, MMLU, RepoPeftBench")
            
        # Build NetworkX Graph with filtered entities
        allowed_types = {"Paper", "Method", "Dataset", "Concept"}
        filtered_entities = [e for e in entities if e.get("entity_type") in allowed_types]
        filtered_entities.sort(key=lambda x: pagerank_lookup.get(x["entity_id"], 0), reverse=True)
        
        # Resolve searched node match from entire dataset
        match_id = None
        match_name = None
        match_type = None
        if search_node:
            q = search_node.lower().strip()
            for ent in entities:
                if q in ent["name"].lower() or q in ent["entity_id"].lower():
                    match_id = ent["entity_id"]
                    match_name = ent["name"]
                    match_type = ent["entity_type"]
                    break
                    
        scope_option = "Global Graph View"
        if match_id:
            scope_option = st.radio(
                "Graph View Scope",
                options=["Global Graph View", "Focus: 1-Hop Neighbors", "Focus: 2-Hop Neighbors"],
                horizontal=True
            )
            
        # Build G based on scope
        G = nx.Graph()
        
        if match_id and scope_option != "Global Graph View":
            hops = 1 if scope_option == "Focus: 1-Hop Neighbors" else 2
            
            # Build full graph first
            Full_G = nx.Graph()
            for ent in entities:
                if ent.get("entity_type") in allowed_types:
                    Full_G.add_node(
                        ent["entity_id"],
                        name=ent["name"],
                        type=ent["entity_type"],
                        pagerank=pagerank_lookup.get(ent["entity_id"], 0.0)
                    )
            for rel in relationships:
                if rel["source"] in Full_G and rel["target"] in Full_G:
                    Full_G.add_edge(rel["source"], rel["target"], relation=rel["relation"])
            
            if match_id in Full_G:
                G = nx.ego_graph(Full_G, match_id, radius=hops)
                st.success(f"Focused sub-graph: Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} relationships within {hops} hop(s) of '{match_name}'.")
            else:
                G.add_node(
                    match_id,
                    name=match_name,
                    type=match_type,
                    pagerank=pagerank_lookup.get(match_id, 0.0)
                )
                st.info(f"Searched node '{match_name}' has no registered edges in database.")
        else:
            top_entities = filtered_entities[:max_nodes]
            top_ids = {e["entity_id"] for e in top_entities}
            top_rels = [
                r for r in relationships
                if r["source"] in top_ids and r["target"] in top_ids
            ]
            
            for ent in top_entities:
                G.add_node(
                    ent["entity_id"],
                    name=ent["name"],
                    type=ent["entity_type"],
                    pagerank=pagerank_lookup.get(ent["entity_id"], 0.0)
                )
                
            for rel in top_rels:
                G.add_edge(rel["source"], rel["target"], relation=rel["relation"])
                
        # Calculate positions using NetworkX spring layout
        if G.number_of_nodes() == 0:
            st.info("No nodes to render.")
            pos = {}
        else:
            pos = nx.spring_layout(G, k=0.15, seed=42)
            
        # Build Plotly traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='#cbd5e1'),
            hoverinfo='none',
            mode='lines'
        )
        
        colors = {
            "Paper": "#6366f1",      # Indigo
            "Method": "#10b981",     # Green
            "Dataset": "#f97316",    # Orange
            "Concept": "#ef4444"     # Red
        }
        
        node_traces = []
        for group, color in colors.items():
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                if G.nodes[node]["type"] == group:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    label = G.nodes[node]["name"]
                    pr = G.nodes[node].get("pagerank", 0.0)
                    node_text.append(f"Name: {label}<br/>Type: {group}<br/>PageRank: {pr:.5f}")
                    
                    # Scaling nodes by PageRank
                    size = 8 + pr * 150
                    node_size.append(min(size, 32))
                    
            if node_x:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    name=group,
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=False,
                        color=color,
                        size=node_size,
                        line=dict(width=1.5, color='#ffffff')
                    )
                )
                node_traces.append(node_trace)
                
        # Draw base figure
        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                showlegend=True,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"
                )
            )
        )
        
        # Highlight node search logic
        if match_id and match_id in pos:
            hx, hy = pos[match_id]
            highlight_trace = go.Scatter(
                x=[hx], y=[hy],
                mode='markers+text',
                name='Highlighted Node',
                text=[match_name],
                textposition="top center",
                textfont=dict(color='#db2777', size=12, family="Plus Jakarta Sans"),
                hoverinfo='text',
                marker=dict(
                    color='rgba(219, 39, 119, 0.25)',
                    size=38,
                    line=dict(color='#db2777', width=2)
                )
            )
            fig.add_trace(highlight_trace)
            if scope_option == "Global Graph View":
                st.success(f"Highlighted node: {match_name}")
        elif search_node and not match_id:
            st.warning("No matching node found in entire graph.")
        elif search_node and match_id not in pos:
            st.warning("Node found in database but not present in the current top PageRank nodes layout. Select 1-Hop or 2-Hop focus scope to pull it.")
            
        if G.number_of_nodes() > 0:
            st.plotly_chart(fig, use_container_width=True)
            
        # Direct Neighbors Table if node is searched
        if match_id:
            direct_neighbors = []
            for rel in relationships:
                if rel["source"] == match_id:
                    direct_neighbors.append({"node_id": rel["target"], "relation": rel["relation"], "type": "target"})
                elif rel["target"] == match_id:
                    direct_neighbors.append({"node_id": rel["source"], "relation": rel["relation"], "type": "source"})
            
            if direct_neighbors:
                ent_name_map = {e["entity_id"]: e for e in entities}
                rows = []
                for idx, nbr in enumerate(direct_neighbors[:15], 1):
                    nbr_ent = ent_name_map.get(nbr["node_id"])
                    if nbr_ent:
                        rel_str = f"-[{nbr['relation']}]->" if nbr["type"] == "target" else f"<-[{nbr['relation']}]-"
                        rows.append([idx, f"<strong>{match_name}</strong>", rel_str, f"<strong>{nbr_ent['name']}</strong> ({nbr_ent['entity_type']})"])
                
                st.markdown(f"##### Neighbors of '{match_name}' (Top {len(rows)})")
                
                def local_html_table(headers, rows):
                    header_html = "".join([f"<th style='padding: 8px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #4f46e5;'>{h}</th>" for h in headers])
                    rows_html = ""
                    for r in rows:
                        cells = "".join([f"<td style='padding: 8px; border-bottom: 1px solid #f1f5f9; color: #0f172a;'>{c}</td>" for c in r])
                        rows_html += f"<tr>{cells}</tr>"
                    return f"<table style='width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 5px;'><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
                
                st.markdown(local_html_table(["#", "Node A", "Relationship", "Node B"], rows), unsafe_allow_html=True)
        
    with col_t:
        st.markdown("### Graph Analytics & Leaderboards")
        
        tab_metrics, tab_papers, tab_methods, tab_datasets = st.tabs([
            "Metrics", "Top Papers", "Top Methods", "Top Datasets"
        ])
        
        with tab_metrics:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-top: 10px;">
                    <h5 style="margin-top:0; color:#4f46e5; font-family: 'Outfit', sans-serif;">Topological Metrics</h5>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05);">
                            <td style="padding: 10px 0; color:#64748b;">Graph Density</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color:#0f172a;">{stats_meta.get('density', 0.0) * 1000:.4f} &times; 10-3</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05);">
                            <td style="padding: 10px 0; color:#64748b;">Connected Components</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color:#0f172a;">{stats_meta.get('connected_components_count', 1)}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05);">
                            <td style="padding: 10px 0; color:#64748b;">Average Node Degree</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color:#0f172a;">{stats_meta.get('average_degree', 0.0):.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; color:#64748b;">Detected Communities</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color:#0f172a;">{stats_meta.get('communities_count', 0)}</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        def render_leaderboard(entity_type: str, limit: int = 15):
            filtered = [n for n in node_metrics if n.get("entity_type") == entity_type]
            filtered.sort(key=lambda x: x["pagerank"], reverse=True)
            
            if not filtered:
                st.write(f"No nodes of type {entity_type} found in metrics.")
                return
                
            st.markdown(
                f"""
                <div style="max-height: 400px; overflow-y: auto; margin-top: 10px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                        <thead>
                            <tr style="border-bottom: 2px solid rgba(0,0,0,0.08); text-align: left; color:#4f46e5; font-family: 'Outfit', sans-serif;">
                                <th style="padding: 6px 0;">Rank</th>
                                <th style="padding: 6px 0;">Name</th>
                                <th style="padding: 6px 0; text-align: right;">PageRank</th>
                            </tr>
                        </thead>
                        <tbody>
                """,
                unsafe_allow_html=True
            )
            
            for idx, item in enumerate(filtered[:limit], 1):
                name = item["name"]
                if len(name) > 32:
                    name = name[:29] + "..."
                st.markdown(
                    f"""
                            <tr style="border-bottom: 1px solid rgba(0,0,0,0.05);">
                                <td style="padding: 6px 0; font-weight: bold; color: #64748b;">{idx}</td>
                                <td style="padding: 6px 0; color: #0f172a;" title="{item['name']}">{name}</td>
                                <td style="padding: 6px 0; text-align: right; font-weight: bold; color: #059669;">{item['pagerank']:.5f}</td>
                            </tr>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)
            
        with tab_papers:
            st.markdown("##### Most Central Papers")
            render_leaderboard("Paper")
            
        with tab_methods:
            st.markdown("##### Most Central Methods")
            render_leaderboard("Method")
            
        with tab_datasets:
            st.markdown("##### Most Central Datasets")
            render_leaderboard("Dataset")

# Footer
render_footer()
