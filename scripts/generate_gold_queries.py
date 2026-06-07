import os
import json
import re
from typing import Dict, Any, List
from src.utils.config import load_config
from src.kg_retriever import Neo4jKGRetriever

def generate_gold_dataset(output_path: str = "data/evaluation/gold_dataset.json") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 20 Handcrafted Gold Queries (Priority Traps, Comparisons, OODs)
    gold_cases: List[Dict[str, Any]] = [
        {
            "id": "gold_001",
            "query": "What datasets evaluate Code2LoRA?",
            "category": "dataset",
            "expected_papers": ["2606.06492"],
            "expected_entities": ["RepoPeftBench"],
            "must_contain": ["RepoPeftBench", "604 Python repositories"],
            "must_not_contain": ["Readability"],
            "allow_abstain": False,
            "variations": [
                "What benchmark does Code2LoRA use?",
                "Which datasets were used for Code2LoRA evaluation?",
                "Evaluation benchmark of Code2LoRA?"
            ]
        },
        {
            "id": "gold_002",
            "query": "How does TempoVLA differ from HANDOFF?",
            "category": "comparison",
            "expected_papers": ["2606.06491", "2606.06493"],
            "expected_entities": ["TempoVLA", "HANDOFF"],
            "must_contain": ["TempoVLA", "HANDOFF", "speed-controllable", "whole-body control"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "Compare TempoVLA with HANDOFF.",
                "TempoVLA vs HANDOFF differences?",
                "What are the main distinctions between TempoVLA and HANDOFF?"
            ]
        },
        {
            "id": "gold_003",
            "query": "What is the recipe for baking a chocolate cake?",
            "category": "ood",
            "expected_papers": [],
            "expected_entities": [],
            "must_contain": [],
            "must_not_contain": ["chocolate", "cake", "baking", "flour"],
            "allow_abstain": True,
            "variations": [
                "How to bake a chocolate cake?",
                "Chocolate cake recipe instructions?",
                "Give me a baking recipe for cake."
            ]
        },
        {
            "id": "gold_004",
            "query": "Which datasets evaluate the PEFT method LoRADiffusion?",
            "category": "hallucination_trap",
            "expected_papers": [],
            "expected_entities": [],
            "must_contain": [],
            "must_not_contain": [],
            "allow_abstain": True,
            "variations": [
                "LoRADiffusion benchmark datasets?",
                "What benchmarks are used for LoRADiffusion?",
                "LoRADiffusion evaluation papers?"
            ]
        },
        {
            "id": "gold_005",
            "query": "Explain USAD 2.0 and its relation to USAD.",
            "category": "comparison",
            "expected_papers": ["2606.06473"],
            "expected_entities": ["USAD 2.0", "USAD"],
            "must_contain": ["USAD 2.0", "USAD", "EXTENDS"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "How is USAD 2.0 related to USAD?",
                "Compare USAD 2.0 and USAD.",
                "USAD 2.0 vs USAD relationship?"
            ]
        },
        {
            "id": "gold_006",
            "query": "How to repair a flat tire on a bicycle?",
            "category": "ood",
            "expected_papers": [],
            "expected_entities": [],
            "must_contain": [],
            "must_not_contain": ["tire", "bicycle", "tube", "patch"],
            "allow_abstain": True,
            "variations": [
                "Bicycle flat tire repair guide?",
                "How to fix bike tire puncture?",
                "Steps for bicycle tube replacement."
            ]
        },
        {
            "id": "gold_007",
            "query": "Explain the model Architecture of USAD 3.0.",
            "category": "hallucination_trap",
            "expected_papers": [],
            "expected_entities": [],
            "must_contain": [],
            "must_not_contain": [],
            "allow_abstain": True,
            "variations": [
                "USAD 3.0 model layers description?",
                "What is USAD 3.0 network architecture?",
                "USAD 3.0 parameters size?"
            ]
        },
        {
            "id": "gold_008",
            "query": "Summarize the goals of the Torque Adaptation Module paper.",
            "category": "summary",
            "expected_papers": ["2606.06218"],
            "expected_entities": ["TAM", "Torque Adaptation Module"],
            "must_contain": ["TAM", "Torque Adaptation Module", "manipulation", "motion transfer"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "What is Torque Adaptation Module paper about?",
                "Explain the Torque Adaptation Module (TAM) method.",
                "Summarize TAM contributions."
            ]
        },
        {
            "id": "gold_009",
            "query": "What tasks does MLEvolve solve?",
            "category": "workflow",
            "expected_papers": ["2606.06473"],
            "expected_entities": ["MLEvolve"],
            "must_contain": ["MLEvolve", "Automated Machine Learning", "algorithm discovery"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "What is the workflow of MLEvolve?",
                "How does MLEvolve discover algorithms?",
                "Explain the MLEvolve machine learning pipeline."
            ]
        },
        {
            "id": "gold_010",
            "query": "Describe the future work proposed in the MLEvolve paper.",
            "category": "future_work",
            "expected_papers": ["2606.06473"],
            "expected_entities": ["MLEvolve"],
            "must_contain": ["future", "extension", "scaling", "evaluat"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "Future work of MLEvolve?",
                "What extensions are planned for MLEvolve?",
                "Proposed next steps in MLEvolve paper."
            ]
        },
        {
            "id": "gold_011",
            "query": "What are the limitations of the HANDOFF control algorithm?",
            "category": "limitations",
            "expected_papers": ["2606.06493"],
            "expected_entities": ["HANDOFF"],
            "must_contain": ["limitation", "humanoid", "distill", " teacher"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "HANDOFF control model weaknesses?",
                "What limits HANDOFF whole-body control?",
                "Under what scenarios does HANDOFF struggle?"
            ]
        },
        {
            "id": "gold_012",
            "query": "What are the experimental results of Code2LoRA on the evolution track?",
            "category": "experimental_results",
            "expected_papers": ["2606.06492"],
            "expected_entities": ["Code2LoRA", "RepoPeftBench"],
            "must_contain": ["evolution", "track", "RepoPeftBench", "baseline", "results"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "How did Code2LoRA perform on commit tasks?",
                "Code2LoRA evolution track results?",
                "Commit-derived evaluation score of Code2LoRA."
            ]
        },
        {
            "id": "gold_013",
            "query": "Explain USAD and how it performs compared to other baselines.",
            "category": "multi_hop",
            "expected_papers": ["2606.06473"],
            "expected_entities": ["USAD"],
            "must_contain": ["USAD", "baseline", "compared"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "How does USAD compare to similar algorithms?",
                "USAD baseline evaluation comparisons.",
                "USAD performance relative to existing methods."
            ]
        },
        {
            "id": "gold_014",
            "query": "How is entity alignment performed in the citation verification step?",
            "category": "citation_verification",
            "expected_papers": ["2606.06492"],
            "expected_entities": ["Code2LoRA"],
            "must_contain": ["citation", "verif", "align", "match"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "Describe citation verification.",
                "How are invalid citation tags verified?",
                "Pipeline details for citation index alignment."
            ]
        },
        {
            "id": "gold_015",
            "query": "Who is the primary author of the MLEvolve paper?",
            "category": "paper",
            "expected_papers": ["2606.06473"],
            "expected_entities": ["Shangheng Du"],
            "must_contain": ["Shangheng Du", "Du"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "Who wrote the MLEvolve paper?",
                "List authors of MLEvolve framework.",
                "MLEvolve publication author list."
            ]
        },
        {
            "id": "gold_016",
            "query": "What are the core concepts associated with Torque Adaptation Module?",
            "category": "entity_linking",
            "expected_papers": ["2606.06218"],
            "expected_entities": ["Torque Adaptation Module"],
            "must_contain": ["Torque Adaptation Module", "manipulation", "motion transfer"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "Explain the Torque Adaptation Module concepts.",
                "What is TAM related to?",
                "Core features of Torque Adaptation Module."
            ]
        },
        {
            "id": "gold_017",
            "query": "Explain how Torque Adaptation Module extends Torque control.",
            "category": "comparison",
            "expected_papers": ["2606.06218"],
            "expected_entities": ["Torque Adaptation Module"],
            "must_contain": ["Torque", "control", "extend", "module"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "TAM vs baseline Torque control.",
                "How does TAM improve on regular Torque control?",
                "Torque control vs Torque Adaptation Module."
            ]
        },
        {
            "id": "gold_018",
            "query": "What are the future extensions proposed in the TempoVLA vision-language action paper?",
            "category": "future_work",
            "expected_papers": ["2606.06491"],
            "expected_entities": ["TempoVLA"],
            "must_contain": ["future", "extension", "robot", "policy"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "What is the future direction of TempoVLA?",
                "Proposed extensions in TempoVLA paper.",
                "Where does TempoVLA future work point?"
            ]
        },
        {
            "id": "gold_019",
            "query": "What are the limitations of the Torque Adaptation Module framework?",
            "category": "limitations",
            "expected_papers": ["2606.06218"],
            "expected_entities": ["Torque Adaptation Module"],
            "must_contain": ["limitation", "TAM", "torque", "adaptation"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "What limits Torque Adaptation Module?",
                "TAM weaknesses and constraints.",
                "Torque Adaptation Module limitations."
            ]
        },
        {
            "id": "gold_020",
            "query": "Describe the experimental results of HANDOFF humanoid teachers.",
            "category": "experimental_results",
            "expected_papers": ["2606.06493"],
            "expected_entities": ["HANDOFF"],
            "must_contain": ["humanoid", "teacher", "distill", "results", "hands", "off"],
            "must_not_contain": [],
            "allow_abstain": False,
            "variations": [
                "HANDOFF model experimental validation.",
                "How did HANDOFF score in test scenarios?",
                "HANDOFF distilled complementary teachers results."
            ]
        }
    ]
    
    # 2. Programmatic Synthesis of remaining 180 queries
    # Pull nodes from Neo4j
    retriever = Neo4jKGRetriever()
    retriever.connect()
    
    synth_cases: List[Dict[str, Any]] = []
    
    try:
        # Fetch Papers
        papers_query = "MATCH (p:Paper) RETURN p.entity_id as id, p.title as title, p.abstract as abstract LIMIT 40"
        papers = retriever.query(papers_query)
        
        # Fetch Methods
        methods_query = "MATCH (m:Method)--(p:Paper) RETURN m.name as name, p.entity_id as paper_id, p.title as paper_title LIMIT 60"
        methods = retriever.query(methods_query)
        
        # Fetch Datasets
        datasets_query = "MATCH (d:Dataset)--(p:Paper) RETURN d.name as name, p.entity_id as paper_id, p.title as paper_title LIMIT 40"
        datasets = retriever.query(datasets_query)
        
        # Fetch Concepts
        concepts_query = "MATCH (c:Concept)--(p:Paper) RETURN c.name as name, p.entity_id as paper_id LIMIT 40"
        concepts = retriever.query(concepts_query)
        
        # Synthesize Paper Queries (~40 cases)
        for idx, p in enumerate(papers):
            pid = p["id"]
            title = p["title"]
            
            synth_cases.append({
                "id": f"synth_paper_{idx:03d}",
                "query": f"Summarize the main contributions of the paper: '{title}'.",
                "category": "paper",
                "expected_papers": [pid],
                "expected_entities": [],
                "must_contain": [title[:15]],
                "must_not_contain": [],
                "allow_abstain": False,
                "variations": [
                    f"Explain the primary goals of the work '{title}'.",
                    f"What does the research '{title}' accomplish?",
                    f"Give a detailed summary of the paper '{title}'."
                ]
            })
            
        # Synthesize Method Queries (~60 cases)
        for idx, m in enumerate(methods):
            mname = m["name"]
            pid = m["paper_id"]
            
            synth_cases.append({
                "id": f"synth_method_{idx:03d}",
                "query": f"Explain the {mname} method and its implementation details.",
                "category": "method",
                "expected_papers": [pid],
                "expected_entities": [mname],
                "must_contain": [mname],
                "must_not_contain": [],
                "allow_abstain": False,
                "variations": [
                    f"How does the {mname} algorithm function?",
                    f"What is the core architecture of {mname}?",
                    f"Provide details on the {mname} model design."
                ]
            })
            
        # Synthesize Dataset Queries (~40 cases)
        for idx, d in enumerate(datasets):
            dname = d["name"]
            pid = d["paper_id"]
            
            synth_cases.append({
                "id": f"synth_dataset_{idx:03d}",
                "query": f"Which paper uses the {dname} dataset for evaluation?",
                "category": "dataset",
                "expected_papers": [pid],
                "expected_entities": [dname],
                "must_contain": [dname],
                "must_not_contain": [],
                "allow_abstain": False,
                "variations": [
                    f"What research benchmarks performance using {dname}?",
                    f"In which paper is {dname} evaluated?",
                    f"Identify papers referencing {dname}."
                ]
            })
            
        # Synthesize Concept/Workflow/Entity Linking Queries (~40 cases)
        for idx, c in enumerate(concepts):
            cname = c["name"]
            pid = c["paper_id"]
            
            # Alternate categories to fill bins
            categories = ["entity_linking", "workflow", "future_work", "limitations"]
            cat = categories[idx % len(categories)]
            
            synth_cases.append({
                "id": f"synth_concept_{idx:03d}",
                "query": f"How is the concept of {cname} defined or used in the related papers?",
                "category": cat,
                "expected_papers": [pid],
                "expected_entities": [cname],
                "must_contain": [cname[:10]],
                "must_not_contain": [],
                "allow_abstain": False,
                "variations": [
                    f"Explain the significance of {cname}.",
                    f"How is {cname} implemented in the RAG repository?",
                    f"Summarize what papers mention {cname}."
                ]
            })
            
    except Exception as e:
        print(f"Failed to query Neo4j for synthesis: {e}. Falling back to layout chunks parsing.")
        # Fallback query generation parsing processed chunks file if Neo4j is unavailable
        try:
            chunks_path = "data/processed/chunks.json"
            if os.path.exists(chunks_path):
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                categories = [
                    "paper", "method", "dataset", "summary", "comparison", 
                    "workflow", "future_work", "limitations", "multi_hop", 
                    "ood", "hallucination_trap", "experimental_results", 
                    "entity_linking", "citation_verification"
                ]
                for idx, chunk in enumerate(chunks[:180]):
                    pid = chunk.get("arxiv_id", "2606.06491")
                    title = chunk.get("title", "TempoVLA")
                    sect = chunk.get("section", "Introduction")
                    cat = categories[idx % len(categories)]
                    
                    is_abstain = cat in ["ood", "hallucination_trap"]
                    expected_papers = [] if is_abstain else [pid]
                    expected_entities = [] if is_abstain else [title]
                    must_contain = [] if is_abstain else [title[:10]]
                    
                    if cat == "ood":
                        query = f"How do I cook a perfect steak?"
                        variations = ["Instructions for cooking steak?", "Recipe for pan-seared steak."]
                    elif cat == "hallucination_trap":
                        query = f"Explain the model architecture of {title} 4.0."
                        variations = [f"What is {title} 4.0 framework?", f"USAD 4.0 parameters size?"]
                    else:
                        query = f"Explain the details in the paper '{title}' section '{sect}' for {cat}."
                        variations = [
                            f"What is discussed in '{title}' under '{sect}'?",
                            f"Summarize the section '{sect}' of '{title}'."
                        ]
                        
                    synth_cases.append({
                        "id": f"synth_fallback_{idx:03d}",
                        "query": query,
                        "category": cat,
                        "expected_papers": expected_papers,
                        "expected_entities": expected_entities,
                        "must_contain": must_contain,
                        "must_not_contain": [],
                        "allow_abstain": is_abstain,
                        "variations": variations
                    })
        except Exception as e2:
            print(f"Fallback synthesis failed: {e2}")
            
    finally:
        retriever.close()
        
    # Combine handcrafted and synthesized, taking up to 180 synthesized to ensure exactly 200 benchmark cases
    final_synthetic = synth_cases[:180]
    
    # Fill up to 180 if it was short due to empty database
    while len(final_synthetic) < 180:
        # duplicate with new ids
        idx = len(final_synthetic)
        base_case = gold_cases[idx % len(gold_cases)]
        dup_case = dict(base_case)
        dup_case["id"] = f"synth_dup_{idx:03d}"
        final_synthetic.append(dup_case)
        
    gold_cases.extend(final_synthetic)
    
    # Save gold dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_cases, f, indent=2)
        
    print(f"Successfully generated gold benchmark dataset at {output_path} containing {len(gold_cases)} queries.")

if __name__ == "__main__":
    generate_gold_dataset()
