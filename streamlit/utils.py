import os
import json
import re
import streamlit as st
from dotenv import load_dotenv

# Initialize environment variables once
load_dotenv()

def parse_inline_markdown(text: str) -> str:
    """Parses inline bold, italic, code, and citation brackets to HTML."""
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    # Inline code: `code` -> <code>code</code>
    text = re.sub(r'`(.*?)`', r'<code style="background-color: #f1f5f9; padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.85em; color: #db2777;">\1</code>', text)
    # Renumbered Citations: [Y] -> highlighted span badge
    text = re.sub(
        r'\[(\d+)\]', 
        r'<span style="background-color: rgba(99, 102, 241, 0.1); color: #4f46e5; border: 1px solid rgba(99, 102, 241, 0.2); padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; cursor: default; white-space: nowrap;">[\1]</span>', 
        text
    )
    # Citations: [Citation X] -> highlighted span badge (safety fallback)
    text = re.sub(
        r'\[Citation\s+(\d+)\]', 
        r'<span style="background-color: rgba(99, 102, 241, 0.1); color: #4f46e5; border: 1px solid rgba(99, 102, 241, 0.2); padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; cursor: default; white-space: nowrap;">Citation \1</span>', 
        text
    )
    # Invalid citations: silently remove [Invalid Citation Removed] to keep text readable.
    # These are citations the validator rejected; we don't surface the internal label to users.
    text = re.sub(r'\[Invalid\s+Citation\s+Removed\]', '', text)

    return text

def markdown_to_html(md_text: str) -> str:
    """Converts basic markdown (headers, tables, nested lists, ordered lists, bold/italic, code blocks, horizontal rules) to clean HTML."""
    if not md_text:
        return ""
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    in_code_block = False
    list_stack = []  # Stack of tuples: (list_type, indent_level)
    
    for line in lines:
        stripped = line.strip()
        
        # 0. Handle Code Blocks
        if stripped.startswith("```"):
            if list_stack:
                while list_stack:
                    lt, _ = list_stack.pop()
                    html_lines.append(f"</{lt}>")
            if in_table:
                html_lines.append('  </tbody>')
                html_lines.append('</table>')
                in_table = False
            if in_code_block:
                html_lines.append('</pre>')
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                html_lines.append(f'<pre style="background-color: #f1f5f9; padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.85em; color: #0f172a; overflow-x: auto; border: 1px solid #e2e8f0; line-height: 1.4; margin: 12px 0;">')
                in_code_block = True
            continue
            
        if in_code_block:
            escaped_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(escaped_line)
            continue
            
        # 0.5 Handle Horizontal Rules
        if stripped in ["---", "***", "___"]:
            if list_stack:
                while list_stack:
                    lt, _ = list_stack.pop()
                    html_lines.append(f"</{lt}>")
            if in_table:
                html_lines.append('  </tbody>')
                html_lines.append('</table>')
                in_table = False
            html_lines.append('<hr style="border: 0; border-top: 1px solid #cbd5e1; margin: 20px 0;" />')
            continue
            
        # 0.6 Handle Headers
        header_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if header_match:
            if list_stack:
                while list_stack:
                    lt, _ = list_stack.pop()
                    html_lines.append(f"</{lt}>")
            if in_table:
                html_lines.append('  </tbody>')
                html_lines.append('</table>')
                in_table = False
            level = len(header_match.group(1))
            content = header_match.group(2)
            margin_top = "20px" if level <= 3 else "14px"
            margin_bottom = "10px" if level <= 3 else "6px"
            font_size = f"{2.0 - 0.2 * level}rem"
            html_lines.append(f'<h{level} style="font-family: \'Outfit\', sans-serif; font-weight: 600; color: #0f172a; margin-top: {margin_top}; margin-bottom: {margin_bottom}; font-size: {font_size};">{parse_inline_markdown(content)}</h{level}>')
            continue
            
        # 1. Handle Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            if list_stack:
                while list_stack:
                    lt, _ = list_stack.pop()
                    html_lines.append(f"</{lt}>")
            # Check if it's a separator line like |---|---|
            if re.match(r'^\|[\s\-\|:]+\|$', stripped):
                continue
            
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                in_table = True
                html_lines.append('<table style="width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9rem; background: #ffffff;">')
                html_lines.append('  <thead>')
                html_lines.append('    <tr style="border-bottom: 2px solid #cbd5e1; background-color: #f8fafc;">')
                for cell in cells:
                    html_lines.append(f'      <th style="padding: 10px 8px; text-align: left; font-weight: 600; color: #4f46e5;">{parse_inline_markdown(cell)}</th>')
                html_lines.append('    </tr>')
                html_lines.append('  </thead>')
                html_lines.append('  <tbody>')
            else:
                html_lines.append('    <tr style="border-bottom: 1px solid #e2e8f0;">')
                for cell in cells:
                    html_lines.append(f'      <td style="padding: 10px 8px; text-align: left; color: #0f172a;">{parse_inline_markdown(cell)}</td>')
                html_lines.append('    </tr>')
            continue
        elif in_table:
            html_lines.append('  </tbody>')
            html_lines.append('</table>')
            in_table = False
            
        # 2. Handle Lists (nested)
        ul_match = re.match(r'^(\s*)([\*\-\+])\s+(.*)$', line)
        ol_match = re.match(r'^(\s*)(\d+)\.\s+(.*)$', line)
        
        if ul_match or ol_match:
            if in_table:
                html_lines.append('  </tbody>')
                html_lines.append('</table>')
                in_table = False
                
            indent = len(ul_match.group(1)) if ul_match else len(ol_match.group(1))
            list_type = "ul" if ul_match else "ol"
            content = ul_match.group(3) if ul_match else ol_match.group(3)
            
            # Reconcile stack with indent level
            if not list_stack:
                list_stack.append((list_type, indent))
                style = 'list-style-type: disc;' if list_type == "ul" else 'list-style-type: decimal;'
                html_lines.append(f'<{list_type} style="margin: 10px 0; padding-left: 20px; {style} color: #0f172a;">')
            else:
                top_type, top_indent = list_stack[-1]
                if indent > top_indent:
                    list_stack.append((list_type, indent))
                    style = 'list-style-type: circle;' if list_type == "ul" else 'list-style-type: lower-alpha;'
                    html_lines.append(f'<{list_type} style="margin: 5px 0; padding-left: 20px; {style} color: #0f172a;">')
                elif indent < top_indent:
                    while list_stack and indent < list_stack[-1][1]:
                        lt, _ = list_stack.pop()
                        html_lines.append(f"</{lt}>")
                    if not list_stack or list_stack[-1][0] != list_type:
                        if list_stack:
                            list_stack.pop()
                            html_lines.append(f"</{lt}>")
                        list_stack.append((list_type, indent))
                        style = 'list-style-type: disc;' if list_type == "ul" else 'list-style-type: decimal;'
                        html_lines.append(f'<{list_type} style="margin: 10px 0; padding-left: 20px; {style} color: #0f172a;">')
                elif list_type != top_type:
                    list_stack.pop()
                    html_lines.append(f"</{top_type}>")
                    list_stack.append((list_type, indent))
                    style = 'list-style-type: disc;' if list_type == "ul" else 'list-style-type: decimal;'
                    html_lines.append(f'<{list_type} style="margin: 10px 0; padding-left: 20px; {style} color: #0f172a;">')
                    
            html_lines.append(f'<li style="margin-bottom: 6px; line-height: 1.5;">{parse_inline_markdown(content)}</li>')
            continue
        else:
            if list_stack:
                while list_stack:
                    lt, _ = list_stack.pop()
                    html_lines.append(f"</{lt}>")
                    
        # 3. Handle normal paragraphs
        if stripped:
            html_lines.append(f'<p style="margin: 12px 0; text-align: justify; line-height: 1.6; color: #0f172a;">{parse_inline_markdown(stripped)}</p>')
        else:
            html_lines.append('<div style="height: 6px;"></div>')
            
    if in_table:
        html_lines.append('  </tbody>')
        html_lines.append('</table>')
    if list_stack:
        while list_stack:
            lt, _ = list_stack.pop()
            html_lines.append(f"</{lt}>")
    if in_code_block:
        html_lines.append('</pre>')
        
    return "\n".join(html_lines)

def load_css():
    """Injects high-quality HSL-tailored light theme styling."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        
        <style>
        /* Main page overrides */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
        }
        
        /* Headers styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #0f172a;
        }
        
        h1 {
            background: linear-gradient(135deg, #4f46e5 0%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem !important;
            padding-bottom: 0.5rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Custom Cards styling */
        .glass-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            color: #0f172a;
        }
        
        .glass-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 12px 24px -10px rgba(99, 102, 241, 0.15);
            background: #ffffff;
        }
        
        .stat-val {
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
            margin-bottom: 4px;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500;
        }
        
        /* Metric Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        
        .badge-paper { background-color: rgba(99, 102, 241, 0.1); color: #4f46e5; border: 1px solid rgba(99, 102, 241, 0.2); }
        .badge-method { background-color: rgba(16, 185, 129, 0.1); color: #059669; border: 1px solid rgba(16, 185, 129, 0.2); }
        .badge-dataset { background-color: rgba(249, 115, 22, 0.1); color: #ea580c; border: 1px solid rgba(249, 115, 22, 0.2); }
        .badge-concept { background-color: rgba(239, 68, 68, 0.1); color: #dc2626; border: 1px solid rgba(239, 68, 68, 0.2); }
        
        /* Search Box Overrides */
        div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #0f172a !important;
        }
        
        div[data-baseweb="input"]:focus-within {
            border-color: #4f46e5 !important;
        }
        
        /* Streamlit default tab override */
        button[data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            color: #64748b !important;
        }
        button[aria-selected="true"] {
            color: #0f172a !important;
            border-color: #4f46e5 !important;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 40px 0 20px 0;
            font-size: 0.8rem;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_json_cache(filename: str):
    """Utility to load cached JSON files from data/ui_cache or data/processed."""
    paths = [
        os.path.join("data/ui_cache", filename),
        os.path.join("data/processed", filename)
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading {p}: {e}")
    return None

def get_project_insights():
    return load_json_cache("project_insights.json") or {}

def get_papers_cache():
    return load_json_cache("papers_cache.json") or {}

def get_methods_cache():
    return load_json_cache("methods_cache.json") or {}

def get_datasets_cache():
    return load_json_cache("datasets_cache.json") or {}

def get_timeline():
    return load_json_cache("timeline.json") or []

def get_global_graph_pyvis():
    return load_json_cache("global_graph_pyvis.json") or {"nodes": [], "edges": []}

def classify_paper(p_data: dict) -> str:
    """Classifies a paper into one of 8 topics based on primary category metadata and title/abstract keywords."""
    primary = p_data.get("primary_category", "")
    title = p_data.get("title", "").lower()
    abstract = p_data.get("abstract", "").lower()
    text = title + " " + abstract
    
    # Check RAG keywords first
    if any(k in text for k in ["rag", "retrieval-augmented", "retrieval augmented", "lightrag", "graphrag"]):
        return "RAG"
    # Check Agent keywords
    elif any(k in text for k in ["agent", "agentic", "multi-agent", "handoff"]):
        return "Agents"
    
    # Map primary category
    if primary == "cs.CL":
        return "LLM"
    elif primary == "cs.CV":
        return "Vision"
    elif primary in ["cs.SE", "cs.PL"]:
        return "Code Intelligence"
    
    # Check RL keywords
    if any(k in text for k in ["reinforcement learning", " rl ", "rlhf", "ppo", "actor-critic", "policy gradient", "atari"]):
        return "Reinforcement Learning"
        
    # Check Graph ML keywords
    if any(k in text for k in ["graph", "gnn", "gcn", "gat", "node2vec", "deepwalk"]):
        return "Graph ML"
        
    # Fallback mappings for other primary categories
    if primary in ["cs.AI", "cs.LG", "stat.ML"]:
        if "agent" in text:
            return "Agents"
        elif "graph" in text:
            return "Graph ML"
        else:
            return "LLM"
            
    return "Others"

def render_footer():
    """Renders a consistent, professional footer at the bottom of the page."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #e2e8f0; font-size: 0.8rem; color: #64748b; font-family: 'Outfit', sans-serif;">
            <strong>PaperMind GraphRAG</strong><br/>
            Hybrid Retrieval &bull; FAISS &bull; Neo4j &bull; BGE Embeddings &bull; Gemini/Groq &bull; Grounded Citations
        </div>
        """,
        unsafe_allow_html=True
    )


def highlight_keywords(text: str, query: str) -> str:
    """Highlights query terms in text using a high-contrast span."""
    if not query:
        return text
    # Stop words list to filter out
    stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "in", "out", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "does", "what", "differ", "difference", "explain", "describe", "show"}
    
    words = re.findall(r"\b\w+\b", query.lower())
    keywords = {w for w in words if w not in stopwords and len(w) > 2}
    
    if not keywords:
        return text
        
    pattern = re.compile(rf"\b({'|'.join(re.escape(k) for k in keywords)})\b", re.IGNORECASE)
    return pattern.sub(r'<span style="background-color: rgba(99, 102, 241, 0.15); color: #4f46e5; padding: 2px 4px; border-radius: 4px; font-weight: 500;">\1</span>', text)


def renumber_citations(answer_text: str, citations: list) -> tuple[str, list]:
    """
    Renumbers inline citation tags sequentially (e.g. [Citation 5] becomes [1])
    as they appear in the answer_text, and returns the modified text and
    the list of corresponding Citation objects in the new order.
    """
    matches = re.findall(r'\[Citation\s+(\d+)\]', answer_text)
    citation_map = {}
    new_citations = []
    new_num = 1
    
    for match in matches:
        orig_idx = int(match) - 1
        if orig_idx not in citation_map:
            if 0 <= orig_idx < len(citations):
                citation_map[orig_idx] = new_num
                new_citations.append(citations[orig_idx])
                new_num += 1
                
    def replace_citation(match_obj):
        num = int(match_obj.group(1))
        orig_idx = num - 1
        if orig_idx in citation_map:
            return f"[{citation_map[orig_idx]}]"
        return "[Invalid Citation Removed]"
        
    renumbered_text = re.sub(r'\[Citation\s+(\d+)\]', replace_citation, answer_text)
    return renumbered_text, new_citations


def render_llm_answer(answer_text: str) -> str:
    """
    Unifies answer rendering by converting markdown to clean HTML and displaying it.
    Returns the parsed HTML string.
    """
    html = markdown_to_html(answer_text)
    st.markdown(html, unsafe_allow_html=True)
    return html


def render_citations(citations: list):
    """Displays citations in a beautiful, structured card expander list."""
    if not citations:
        st.info("No citation sources referenced in this answer.")
        return
    
    # Group citations by paper to deduplicate listings
    grouped_citations = {}
    for idx, cit in enumerate(citations, 1):
        aid = getattr(cit, "arxiv_id", cit.get("arxiv_id") if isinstance(cit, dict) else "")
        paper_title = getattr(cit, "paper_title", cit.get("paper_title") if isinstance(cit, dict) else "Unknown Title")
        sec = getattr(cit, "section", cit.get("section") if isinstance(cit, dict) else "Unknown Section")
        page_start = getattr(cit, "page_start", cit.get("page_start") if isinstance(cit, dict) else 1)
        page_end = getattr(cit, "page_end", cit.get("page_end") if isinstance(cit, dict) else 1)
        sim_score = getattr(cit, "similarity_score", cit.get("similarity_score") if isinstance(cit, dict) else 0.0)
        graph_bonus = getattr(cit, "graph_bonus", cit.get("graph_bonus") if isinstance(cit, dict) else 0.0)
        combined_score = getattr(cit, "combined_score", cit.get("combined_score") if isinstance(cit, dict) else 0.0)
        
        if aid not in grouped_citations:
            grouped_citations[aid] = {
                "title": paper_title,
                "citations": []
            }
        grouped_citations[aid]["citations"].append({
            "idx": idx,
            "section": sec,
            "page_start": page_start,
            "page_end": page_end,
            "arxiv_id": aid,
            "similarity_score": sim_score,
            "graph_bonus": graph_bonus,
            "combined_score": combined_score
        })
        
    for aid, gp in grouped_citations.items():
        paper_title = gp["title"]
        citations_list = gp["citations"]
        display_title = paper_title[:90] + "..." if len(paper_title) > 90 else paper_title
        
        expander_label = f"{display_title} (Local PDF)" if (aid.startswith("temp_") or not aid or len(aid) > 45) else f"{display_title} (arXiv: {aid})"
        with st.expander(expander_label):
            st.markdown("**Used for:**")
            for c in citations_list:
                arxiv_link = ""
                if aid and not aid.startswith("temp_") and not len(aid) > 40:
                    arxiv_link = f'<a href="https://arxiv.org/abs/{aid}" target="_blank" style="color: #4f46e5; text-decoration: none; font-weight: 500;">Open on arXiv</a> &bull; '
                st.markdown(
                    f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <strong>[{c['idx']}] Section: {c['section']}</strong> &bull; Pages: {c['page_start']} - {c['page_end']} &bull; 
                        {arxiv_link}
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px; border-top: 1px solid #f1f5f9; padding-top: 6px;">
                            Similarity Score: <code>{c['similarity_score']:.4f}</code> &bull; 
                            Graph Bonus: <code>{c['graph_bonus']:.4f}</code> &bull; 
                            Combined Score: <code>{c['combined_score']:.4f}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Check if PDF exists locally and show download button
            import os as _os
            import time as _time
            pdf_path_raw = f"data/raw/pdfs/{aid}.pdf"
            pdf_path_temp1 = f"data/temp_uploads/{aid}"
            pdf_path_temp2 = f"data/temp_uploads/{aid}.pdf"
            
            local_pdf_path = None
            for p in [pdf_path_raw, pdf_path_temp1, pdf_path_temp2]:
                if _os.path.exists(p) and _os.path.isfile(p):
                    local_pdf_path = p
                    break
                    
            if local_pdf_path:
                try:
                    with open(local_pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    button_key = f"dl_ref_{aid}_{int(_time.time() * 1000) % 100000}"
                    st.download_button(
                        label="📥 Download / Open Local PDF",
                        data=pdf_data,
                        file_name=_os.path.basename(local_pdf_path),
                        mime="application/pdf",
                        key=button_key
                    )
                except Exception as e:
                    pass


def render_confidence(confidence: float, abstained: bool = False):
    """Renders confidence scores in a beautiful card layout with colored progress indicators."""
    conf_pct = int(confidence * 100)
    if abstained:
        conf_label = "Abstained"
        conf_color = "#94a3b8"
    elif confidence >= 0.75:
        conf_label = "High Confidence"
        conf_color = "#059669"
    elif confidence >= 0.50:
        conf_label = "Medium Confidence"
        conf_color = "#ea580c"
    else:
        conf_label = "Low Confidence"
        conf_color = "#dc2626"
        
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; min-height: 140px;">
            <div style="font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Confidence Score</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {conf_color}; margin-top: 8px;">{conf_label} ({conf_pct}%)</div>
            <div style="margin-top: 15px; background-color: #e2e8f0; border-radius: 9999px; height: 8px; overflow: hidden;">
                <div style="background-color: {conf_color}; width: {conf_pct}%; height: 100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_abstention(confidence: float, threshold: float, reason: str = None):
    """Renders abstention reasons and diagnostic details instead of generic text."""
    st.markdown(
        f"""
        <div style="background-color: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 20px; margin-bottom: 25px;">
            <h4 style="color: #dc2626; margin-top: 0; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Search Abstained</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #7f1d1d; line-height: 1.5;">
                The RAG pipeline decided to abstain from answering this query because it did not find sufficient high-quality context.
            </p>
            <div style="margin-top: 14px; font-size: 0.85rem; color: #991b1b; background-color: rgba(239, 68, 68, 0.05); padding: 10px; border-radius: 6px; border: 1px dashed rgba(239, 68, 68, 0.15);">
                <strong>System Diagnostics:</strong><br/>
                &bull; Confidence Score: <code>{confidence:.4f}</code> (Min Threshold required: <code>{threshold:.2f}</code>)<br/>
                &bull; Reason: <code>{reason or "No details provided."}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sources(source_papers: list):
    """Displays standard, clean source paper badges."""
    if not source_papers:
        st.info("No source papers metadata found.")
        return
    
    st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 8px;'>Retrieved Source Papers</div>", unsafe_allow_html=True)
    for p in source_papers:
        title = p.get("title", "Unknown Title")
        arxiv_id = p.get("arxiv_id", "")
        primary_cat = p.get("primary_category", "")
        st.markdown(
            f"""
            <div style="padding: 8px 12px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 8px; font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 500; color: #0f172a; margin-right: 10px;">{title}</div>
                <div style="white-space: nowrap;">
                    <span class="badge badge-paper">{primary_cat}</span>
                    <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" style="color: #4f46e5; text-decoration: none; font-weight: 600;">arXiv:{arxiv_id}</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_metrics(metrics: dict):
    """Renders execution metrics including provider, model, latencies, tokens, and cost estimation."""
    provider = metrics.get("provider_used", "N/A")
    model = metrics.get("generation_model", "N/A")
    total_time = metrics.get("total_execution_time_ms", 0.0) / 1000.0
    tokens = metrics.get("tokens", {})
    cost = metrics.get("cost", 0.0)
    fallback_used = metrics.get("fallback_used", False)
    fallback_reason = metrics.get("fallback_reason", None)
    
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; min-height: 140px;">
            <div style="font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 10px;">Performance & Cost Metrics</div>
            <div style="font-size: 0.85rem; color: #0f172a; line-height: 1.6;">
                <div style="display: flex; justify-content: space-between;"><span>LLM Model</span><span style="font-weight: bold;">{model} ({provider})</span></div>
                <div style="display: flex; justify-content: space-between;"><span>Total Time</span><span style="font-weight: bold;">{total_time:.2f}s</span></div>
                <div style="display: flex; justify-content: space-between;"><span>Prompt / Completion Tokens</span><span style="font-weight: bold;">{tokens.get('prompt', 0)} / {tokens.get('completion', 0)}</span></div>
                <div style="display: flex; justify-content: space-between;"><span>Estimated Cost</span><span style="font-weight: bold; color: #059669;">${cost:.6f}</span></div>
                {"<div style='color: #ea580c; margin-top: 4px; font-size: 0.8rem;'>* API Fallback triggered: " + str(fallback_reason) + "</div>" if fallback_used else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_chunks(chunks: list, query: str = ""):
    """Displays chunks/passages sequentially with keywords highlighted and collapse/expand details."""
    if not chunks:
        st.info("No chunks were retrieved for the context.")
        return
        
    for idx, chunk in enumerate(chunks, 1):
        title = getattr(chunk, "title", chunk.get("title") if isinstance(chunk, dict) else "Unknown")
        sec = getattr(chunk, "section", chunk.get("section") if isinstance(chunk, dict) else "Unknown")
        text = getattr(chunk, "text", chunk.get("text") if isinstance(chunk, dict) else "")
        p_start = getattr(chunk, "page_start", chunk.get("page_start") if isinstance(chunk, dict) else 1)
        p_end = getattr(chunk, "page_end", chunk.get("page_end") if isinstance(chunk, dict) else 1)
        sim = getattr(chunk, "similarity_score", chunk.get("similarity_score") if isinstance(chunk, dict) else None)
        
        preview = text[:150] + "..." if len(text) > 150 else text
        
        st.markdown(f"**Chunk {idx}**")
        score_str = f"Similarity: `{sim:.4f}` | " if sim is not None else ""
        st.markdown(f"*{score_str}Pages: `{p_start}-{p_end}` | Section: `{sec}` | Paper: **{title}**")
        
        highlighted = highlight_keywords(text, query)
        with st.expander("Expand full text"):
            st.markdown(
                f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; font-size: 0.9rem; line-height: 1.5; color: #0f172a; text-align: justify;">
                    {highlighted}
                </div>
                """,
                unsafe_allow_html=True
            )


