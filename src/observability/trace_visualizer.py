import os
import gzip
import json
from typing import Dict, Any, List

def generate_ascii_bar(val: float, max_val: float = 1.0, length: int = 20) -> str:
    if max_val <= 0:
        return "|" + " " * length + "|"
    val = max(0.0, min(val, max_val))
    filled_length = int(round(length * (val / max_val)))
    return "[" + "#" * filled_length + " " * (length - filled_length) + "]"

def build_trace_summary_markdown(traces: List[Dict[str, Any]]) -> str:
    total_queries = len(traces)
    if total_queries == 0:
        return "# Execution Trace Summary\n\nNo trace records found."

    # Identity and Metadata
    first_trace = traces[0]
    experiment_id = first_trace.get("experiment_id", "unknown")
    run_id = first_trace.get("run_id", "unknown")
    
    meta = first_trace.get("metadata") or {}
    git_sha = meta.get("git_sha", "unknown")
    py_version = meta.get("python_version", "unknown")
    plat_os = meta.get("platform_os", "unknown")
    emb_model = meta.get("embedding_model", "unknown")
    judge_model = meta.get("judge_model", "unknown")
    
    # Cost and Tokens
    total_estimated_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_actual_prompt_tokens = 0
    total_actual_completion_tokens = 0
    
    # Latencies
    waterfalls: Dict[str, List[float]] = {}
    stage_latencies: Dict[str, List[float]] = {
        "retrieval": [],
        "policy": [],
        "cache": [],
        "generation": [],
        "citation": [],
        "confidence": []
    }
    
    # Policy routing stats
    policies_selected: Dict[str, int] = {}
    fallback_policies_used = 0
    
    # Cache stats
    cache_hits = 0
    cache_misses = 0
    cache_lookup_latencies = []
    cache_insert_latencies = []
    
    # Confidence breakdown
    confidences: List[float] = []
    answered_confidences: List[float] = []
    abstention_confidences: List[float] = []
    abstentions_count = 0
    query_confidence_list = []
    generation_models = set()
    
    # Process all traces
    for tr in traces:
        # Latency waterfall
        wf = tr.get("latency_waterfall_ms") or {}
        for k, v in wf.items():
            if k not in waterfalls:
                waterfalls[k] = []
            waterfalls[k].append(v)
            
        # Generation trace info
        gen = tr.get("generation") or {}
        total_estimated_cost += gen.get("estimated_cost", 0.0)
        total_prompt_tokens += gen.get("prompt_tokens_estimated", 0)
        total_completion_tokens += gen.get("completion_tokens_estimated", 0)
        
        mod_id = gen.get("model_id")
        if mod_id:
            generation_models.add(mod_id)
            
        act_prompt = gen.get("prompt_tokens_actual")
        if act_prompt is not None:
            total_actual_prompt_tokens += act_prompt
        else:
            total_actual_prompt_tokens += gen.get("prompt_tokens_estimated", 0)
            
        act_comp = gen.get("completion_tokens_actual")
        if act_comp is not None:
            total_actual_completion_tokens += act_comp
        else:
            total_actual_completion_tokens += gen.get("completion_tokens_estimated", 0)
            
        # Stage timings directly from inner traces
        ret = tr.get("retrieval") or {}
        if "execution_time_ms" in ret: # Or if we profile stages manually
            pass
            
        # Policy trace
        pol = tr.get("policy") or {}
        p_sel = pol.get("policy_selected", "unknown")
        policies_selected[p_sel] = policies_selected.get(p_sel, 0) + 1
        if pol.get("fallback_used", False):
            fallback_policies_used += 1
            
        # Cache trace
        c_tr = tr.get("cache") or {}
        cache_hits += c_tr.get("hits", 0)
        cache_misses += c_tr.get("misses", 0)
        if "lookup_latency_ms" in c_tr:
            cache_lookup_latencies.append(c_tr["lookup_latency_ms"])
        if "insert_latency_ms" in c_tr:
            cache_insert_latencies.append(c_tr["insert_latency_ms"])
            
        # Confidence trace
        conf = tr.get("confidence") or {}
        final_conf = conf.get("final_confidence", 0.0)
        confidences.append(final_conf)
        if conf.get("abstention_trigger", False):
            abstentions_count += 1
            abstention_confidences.append(final_conf)
        else:
            answered_confidences.append(final_conf)
            
        query_text = (tr.get("query") or {}).get("user_query", "unknown")
        query_id = tr.get("query_id", "unknown")
        query_confidence_list.append((query_id, final_conf, query_text))

    # Aggregating averages
    avg_wf = {}
    for k, v_list in waterfalls.items():
        avg_wf[k] = sum(v_list) / len(v_list) if v_list else 0.0
        
    avg_cache_lookup = sum(cache_lookup_latencies) / len(cache_lookup_latencies) if cache_lookup_latencies else 0.0
    avg_cache_insert = sum(cache_insert_latencies) / len(cache_insert_latencies) if cache_insert_latencies else 0.0
    total_cache_ops = cache_hits + cache_misses
    cache_hit_rate = cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
    
    avg_confidence_all = sum(confidences) / len(confidences) if confidences else 0.0
    avg_confidence_answered = sum(answered_confidences) / len(answered_confidences) if answered_confidences else 0.0
    avg_confidence_abstention = sum(abstention_confidences) / len(abstention_confidences) if abstention_confidences else 0.0

    # Build Markdown
    lines = []
    lines.append(f"# Execution Trace Summary: {experiment_id}")
    lines.append("")
    lines.append("## Experiment Metadata")
    lines.append(f"- **Experiment ID**: {experiment_id}")
    lines.append(f"- **Run ID**: {run_id}")
    lines.append(f"- **Git Commit SHA**: {git_sha}")
    lines.append(f"- **Python Version**: {py_version}")
    lines.append(f"- **Platform OS**: {plat_os}")
    lines.append(f"- **Embedding Model**: {emb_model}")
    lines.append(f"- **LLM Judge**: {judge_model}")
    lines.append(f"- **Generation Models**: {', '.join(sorted(list(generation_models))) if generation_models else 'unknown'}")
    lines.append(f"- **Total Queries Processed**: {total_queries}")
    lines.append("")
    
    # 1. Cost & Token Accounting
    lines.append("## Cost & Token Accounting")
    lines.append("| Metric | Estimated Value | Actual / Calibrated Value |")
    lines.append("|---|---|---|")
    lines.append(f"| Prompt Tokens | {total_prompt_tokens} | {total_actual_prompt_tokens} |")
    lines.append(f"| Completion Tokens | {total_completion_tokens} | {total_actual_completion_tokens} |")
    lines.append(f"| Total Tokens | {total_prompt_tokens + total_completion_tokens} | {total_actual_prompt_tokens + total_actual_completion_tokens} |")
    lines.append(f"| Estimated API Cost (USD) | ${total_estimated_cost:.5f} | N/A |")
    lines.append(f"| Average Cost per Query | ${total_estimated_cost / total_queries:.5f} | N/A |")
    lines.append("")
    
    # 2. Latency Waterfall
    lines.append("## Latency Waterfall")
    lines.append("This section lists the average latencies recorded for each stage of execution.")
    lines.append("")
    lines.append("| Stage / Operation | Avg Latency (ms) |")
    lines.append("|---|---|")
    for stage, avg_val in sorted(avg_wf.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {stage} | {avg_val:.2f} ms |")
    lines.append("")
    
    # 3. Policy Routing & Cache Observability
    lines.append("## Policy Routing & Cache Stats")
    lines.append("")
    lines.append("### Policy Selection Distribution")
    lines.append("| Policy | Count | Percentage |")
    lines.append("|---|---|---|")
    for policy, count in policies_selected.items():
        pct = (count / total_queries) * 100
        lines.append(f"| {policy} | {count} | {pct:.1f}% |")
    lines.append(f"| Fallback Triggered | {fallback_policies_used} | {(fallback_policies_used / total_queries)*100:.1f}% |")
    lines.append("")
    
    lines.append("### SQLite Embedding Cache Analytics")
    lines.append(f"- **Total Cache Queries**: {total_cache_ops}")
    lines.append(f"- **Cache Hits**: {cache_hits}")
    lines.append(f"- **Cache Misses**: {cache_misses}")
    lines.append(f"- **Cache Hit Rate**: {cache_hit_rate:.2%}")
    lines.append(f"- **Avg Lookup Latency**: {avg_cache_lookup:.2f} ms")
    lines.append(f"- **Avg Insert Latency**: {avg_cache_insert:.2f} ms")
    lines.append("")
    
    # 4. Confidence and Abstention analysis
    lines.append("## Confidence & Abstention Summary")
    lines.append(f"- **Average Confidence (All Queries)**: {avg_confidence_all:.3f}")
    lines.append(f"- **Average Confidence (Answered Queries)**: {avg_confidence_answered:.3f}")
    lines.append(f"- **Average Confidence (Abstention Queries)**: {avg_confidence_abstention:.3f}")
    lines.append(f"- **Abstention Rate**: {abstentions_count} / {total_queries} ({abstentions_count / total_queries:.2%})")
    lines.append("")
    lines.append("### Query Confidence ASCII Plot")
    lines.append("```")
    for q_id, score, q_text in query_confidence_list:
        bar = generate_ascii_bar(score, 1.0, 20)
        short_query = q_text[:40] + "..." if len(q_text) > 40 else q_text
        lines.append(f"{q_id:<12} {score:.3f} {bar} {short_query}")
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)

class TraceVisualizer:
    @staticmethod
    def visualize(experiment_dir: str) -> str:
        """Parses traces from directory and writes trace_summary.md."""
        traces = []
        
        # Determine files
        jsonl_gz = os.path.join(experiment_dir, "traces.jsonl.gz")
        jsonl_raw = os.path.join(experiment_dir, "traces.jsonl")
        
        if os.path.exists(jsonl_gz):
            with gzip.open(jsonl_gz, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
        elif os.path.exists(jsonl_raw):
            with open(jsonl_raw, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
        else:
            return ""
            
        summary_md = build_trace_summary_markdown(traces)
        summary_path = os.path.join(experiment_dir, "trace_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
            
        return summary_path
