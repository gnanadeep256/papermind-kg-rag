import os
import json
import time
import requests
from loguru import logger
from src.utils.config import load_config
from src.fetch_papers import parse_feed, normalize_papers

# Hardcoded verified arXiv IDs for foundational papers
FOUNDATIONAL_PAPERS = {
    "transformers": {
        "Attention Is All You Need": "1706.03762",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": "1810.04805",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach": "1907.11692",
        "ALBERT: A Fast Light BERT for Self-supervised Learning of Language Representations": "1909.11942",
        "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer": "1910.10683",
        "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators": "2003.10555",
        "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension": "1910.13461",
        "DeBERTa: Decoding-enhanced BERT with Disentangled Attention": "2006.03654"
    },
    "rag": {
        "REALM: Retrieval-Augmented Language Model Pre-Training": "2002.08909",
        "Dense Passage Retrieval for Open-Domain Question Answering": "2004.04906",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": "2005.11401",
        "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering": "2007.01282",
        "Improving language models by retrieving from trillions of tokens": "2112.04426",
        "Few-shot learning with retrieval augmented language models": "2208.03299",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection": "2310.11511",
        "From Local to Global: A Graph RAG Approach to Query-Focused Summarization": "2404.16130",
        "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval": "2401.18059",
        "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models": "2405.14496",
        "LightRAG: Simple and Fast Retrieval-Augmented Generation": "2410.05779",
        "KG2RAG: Towards Trustworthy Relationship Extraction with Knowledge Graph and Retrieval-Augmented Generation": "2405.11195"
    },
    "llm": {
        "Training language models to follow instructions with human feedback": "2203.02155",
        "LLaMA: Open and Efficient Foundation Language Models": "2302.13971",
        "Llama 2: Open Foundation and Fine-Tuned Chat Models": "2307.09288",
        "Llama 3 Technical Report": "2407.21783",
        "Mistral 7B": "2310.06825",
        "Mixtral of Experts": "2401.04088",
        "Qwen Technical Report": "2309.16609",
        "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model": "2405.04434",
        "Gemini: A Family of Highly Capable Multimodal Models": "2312.11805"
    },
    "gnn": {
        "Semi-Supervised Classification with Graph Convolutional Networks": "1609.02907",
        "Inductive Representation Learning on Large Graphs": "1706.02216",
        "Graph Attention Networks": "1710.10903",
        "How Powerful are Graph Neural Networks?": "1810.00826",
        "DeepWalk: Online Learning of Social Representations": "1403.6652",
        "node2vec: Scalable Feature Learning for Networks": "1607.00653",
        "Generalization of Transformer Networks to Graphs": "2012.09699"
    },
    "vision": {
        "Very Deep Convolutional Networks for Large-Scale Image Recognition": "1409.1556",
        "Going Deeper with Convolutions": "1409.4842",
        "Deep Residual Learning for Image Recognition": "1512.03385",
        "Densely Connected Convolutional Networks": "1608.06993",
        "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks": "1905.11946",
        "A ConvNet for the 2020s": "2201.03545"
    },
    "vit": {
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale": "2010.11929",
        "Training data-efficient image transformers & distillation through attention": "2012.12877",
        "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows": "2103.14030",
        "Masked Autoencoders Are Scalable Vision Learners": "2111.06377",
        "Emerging Properties in Self-Supervised Vision Transformers": "2104.14294",
        "DINOv2: Learning Robust Visual Features without Supervision": "2304.07193"
    },
    "diffusion": {
        "Denoising Diffusion Probabilistic Models": "2006.11239",
        "Denoising Diffusion Implicit Models": "2010.02502",
        "High-Resolution Image Synthesis with Latent Diffusion Models": "2112.10752"
    },
    "rl": {
        "Playing Atari with Deep Reinforcement Learning": "1312.5602",
        "Asynchronous Methods for Deep Reinforcement Learning": "1602.01783",
        "Proximal Policy Optimization Algorithms": "1707.06347",
        "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor": "1801.01290",
        "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm": "1712.01815",
        "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model": "1911.08265"
    },
    "xai": {
        "\"Why Should I Trust You?\": Explaining the Predictions of Any Classifier": "1602.04938",
        "A Unified Approach to Interpreting Model Predictions": "1705.07874",
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization": "1610.02391",
        "Axiomatic Attribution for Deep Networks": "1703.01365"
    },
    "recommendation": {
        "Wide & Deep Learning for Recommender Systems": "1606.07792",
        "DeepFM: A Factorization-Machine based Neural Network for Predictor Recommendation": "1703.04247",
        "Deep Learning Recommendation Model for Personalization and Recommendation Systems": "1906.00091"
    },
    "kge": {
        "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space": "1903.01011",
        "Complex Embeddings for Simple Link Prediction": "1606.06357",
        "Convolutional 2D Knowledge Graph Embeddings": "1707.01476"
    }
}

# Non-arXiv foundational papers to log as skipped/unresolved
NON_ARXIV_FOUNDATIONAL = [
    {"title": "Improving Language Understanding by Generative Pre-Training", "category": "transformers", "reason": "OpenAI preprint, not on arXiv"},
    {"title": "Language Models are Unsupervised Multitask Learners", "category": "transformers", "reason": "OpenAI preprint, not on arXiv"},
    {"title": "ImageNet Classification with Deep Convolutional Neural Networks", "category": "vision", "reason": "NIPS publication, not on arXiv"},
    {"title": "Mastering the game of Go with deep neural networks and tree search", "category": "rl", "reason": "Nature publication, not on arXiv"},
    {"title": "Translating Embeddings for Modeling Multi-relational Data", "category": "kge", "reason": "NIPS publication, not on arXiv"},
    {"title": "Knowledge Graph Embedding by Translating on Hyperplanes", "category": "kge", "reason": "AAAI publication, not on arXiv"},
    {"title": "Learning Entity and Relation Embeddings for Knowledge Graph Completion", "category": "kge", "reason": "AAAI publication, not on arXiv"}
]

def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Queries arXiv API for metadata of a single arXiv ID."""
    url = "http://export.arxiv.org/api/query"
    params = {"id_list": arxiv_id}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for attempt in range(3):
        try:
            time.sleep(1.0)
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 200:
                parsed_records = parse_feed(response.text)
                if parsed_records:
                    return parsed_records[0]
            logger.warning(f"ArXiv query status {response.status_code} for {arxiv_id}, retrying...")
        except Exception as e:
            logger.warning(f"Error fetching metadata for {arxiv_id}: {e}, retrying...")
            time.sleep(2.0 * (attempt + 1))
            
    return {}

def main():
    logger.info("Initializing Foundational Papers Downloader...")
    
    # Check configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
        
    corpus_config = config.get("corpus", {})
    if not corpus_config.get("include_foundational", True):
        logger.info("Configuration has include_foundational set to False. Aborting download.")
        return
        
    # Directories setup
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    pdf_dir = os.path.join(raw_dir, "pdfs")
    manifest_path = os.path.join(pdf_dir, "download_manifest.json")
    papers_json_path = os.path.join(raw_dir, "papers.json")
    foundational_dir = os.path.join("data", "foundational")
    
    os.makedirs(foundational_dir, exist_ok=True)
    
    # Load download manifest
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}. Starting fresh.")
            
    # Load papers.json
    papers_payload = {"papers": []}
    if os.path.exists(papers_json_path):
        try:
            with open(papers_json_path, "r", encoding="utf-8") as f:
                papers_payload = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load papers.json: {e}")
            
    existing_arxiv_ids = {p.get("arxiv_id") for p in papers_payload.get("papers", []) if p.get("arxiv_id")}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    failed_log = []
    
    # Log non-arXiv papers as skipped/failures
    for paper in NON_ARXIV_FOUNDATIONAL:
        failed_log.append({
            "title": paper["title"],
            "category": paper["category"],
            "arxiv_id": None,
            "status": "failed",
            "reason": paper["reason"]
        })
        
    for category, papers_dict in FOUNDATIONAL_PAPERS.items():
        cat_dir = os.path.join(foundational_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for title, arxiv_id in papers_dict.items():
            local_filename = f"{arxiv_id}.pdf"
            local_path = os.path.join(cat_dir, local_filename).replace("\\", "/")
            
            # Check if already present in manifest and on disk
            if arxiv_id in manifest and manifest[arxiv_id].get("status") == "success":
                registered_path = manifest[arxiv_id].get("local_path")
                if registered_path and os.path.exists(registered_path):
                    logger.debug(f"Paper {arxiv_id} already downloaded at {registered_path}.")
                    skipped_count += 1
                    
                    # Ensure it exists in papers.json metadata
                    if arxiv_id not in existing_arxiv_ids:
                        logger.info(f"Re-fetching metadata for missing papers.json record: {arxiv_id}...")
                        metadata = fetch_arxiv_metadata(arxiv_id)
                        if metadata:
                            papers_payload["papers"].append(metadata)
                            existing_arxiv_ids.add(arxiv_id)
                            # Save papers.json updates
                            with open(papers_json_path, "w", encoding="utf-8") as f:
                                json.dump(papers_payload, f, indent=2, ensure_ascii=False)
                    continue
            
            logger.info(f"Processing foundational paper: {arxiv_id} ({title})")
            
            # Query ArXiv for metadata
            metadata = fetch_arxiv_metadata(arxiv_id)
            if not metadata:
                logger.error(f"Failed to fetch metadata for {arxiv_id}. Skipping PDF download.")
                failed_count += 1
                failed_log.append({
                    "title": title,
                    "category": category,
                    "arxiv_id": arxiv_id,
                    "status": "failed",
                    "reason": "Failed to fetch metadata from arXiv API"
                })
                continue
                
            pdf_url = metadata.get("pdf_url")
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
            # Download PDF
            success = False
            for attempt in range(1, 4):
                try:
                    time.sleep(2.0)
                    response = requests.get(pdf_url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        with open(local_path, "wb") as pdf_file:
                            pdf_file.write(response.content)
                        success = True
                        break
                    elif response.status_code == 403:
                        logger.warning(f"Access forbidden (403) on attempt {attempt} for {arxiv_id}. Applying backoff...")
                        time.sleep(5 * attempt)
                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed for {arxiv_id}: {e}")
                    time.sleep(3.0)
                    
            if success:
                logger.info(f"Successfully downloaded {arxiv_id} to {local_path}.")
                manifest[arxiv_id] = {
                    "status": "success",
                    "local_path": local_path,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "title": metadata.get("title", title)
                }
                
                # Append to papers.json
                if arxiv_id not in existing_arxiv_ids:
                    papers_payload["papers"].append(metadata)
                    existing_arxiv_ids.add(arxiv_id)
                    
                downloaded_count += 1
            else:
                logger.error(f"Failed to download PDF for {arxiv_id}.")
                failed_count += 1
                failed_log.append({
                    "title": title,
                    "category": category,
                    "arxiv_id": arxiv_id,
                    "status": "failed",
                    "reason": "PDF download failed after all retries"
                })
                manifest[arxiv_id] = {
                    "status": "failed",
                    "error": "PDF download failed",
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "title": title
                }
                
            # Save files
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            with open(papers_json_path, "w", encoding="utf-8") as f:
                json.dump(papers_payload, f, indent=2, ensure_ascii=False)
                
    # Save failures log
    failures_log_path = os.path.join(foundational_dir, "failures.json")
    with open(failures_log_path, "w", encoding="utf-8") as f:
        json.dump(failed_log, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("FOUNDATIONAL DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Downloaded: {downloaded_count}")
    logger.info(f"Skipped   : {skipped_count}")
    logger.info(f"Failed    : {failed_count}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
