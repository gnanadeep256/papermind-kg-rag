import os
import json
from dotenv import load_dotenv
from src.llm import query_groq_json
from src.models.graph_models import GroqPayload
from src.utils.config import load_config
from src.utils.logger import logger

load_dotenv()

def main():
    logger.info("Running Groq Extraction Proof of Concept")
    
    # 1. Setup API key validation
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not defined in environment variables.")
        print("Extraction failed")
        return

    # 2. Load extraction prompt template
    prompt_path = "src/prompts/extraction_prompt.txt"
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found at: {prompt_path}")
        print("Extraction failed")
        return
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # 3. Load extraction config
    try:
        config = load_config()
        model_name = config.get("llm", {}).get("extraction_model", "qwen/qwen3-32b")
    except Exception as e:
        logger.warning(f"Failed to load config, defaulting to qwen/qwen3-32b. Error: {e}")
        model_name = "qwen/qwen3-32b"

    # 4. Use a sample abstract (from the HANDOFF paper)
    sample_abstract = (
        "For a humanoid robot to be deployed in the real world, the choice of command space "
        "(i.e., the interface between task planning and whole-body control) is crucial. "
        "Existing whole-body controllers typically demand dense kinematic or spatial references "
        "that planners struggle to synthesize from task semantics. We instead propose a compact, "
        "explicit interface that is intuitive, general, modular, and expressive enough for diverse "
        "manipulation skills. To this end, we introduce HANDOFF, a single humanoid whole-body "
        "controller that follows this interface and is distilled via multi-teacher KL distillation "
        "under a context-conditioned gating scheme into a mixture-of-experts student from three "
        "complementary specialists: whole-body motion tracking with safety-filtered data, locomotion, "
        "and fall-recovery. On the Unitree G1, HANDOFF matches state-of-the-art velocity tracking "
        "and offers one of the largest robust manipulation workspaces. We further demonstrate hardware "
        "feasibility through multiple natural-language-driven task roll-outs, powered by a VLM-driven "
        "agentic planner with no task-specific data or controller fine-tuning."
    )

    full_prompt = f"{prompt_template}\n\nAbstract to analyze:\n{sample_abstract}"
    
    logger.info(f"Target extraction model: {model_name}")

    try:
        raw_response = query_groq_json(full_prompt, model_name=model_name, temperature=0.0)
        logger.info(f"Raw response received:\n{raw_response}")
        
        # Validate JSON using Pydantic GroqPayload schema
        payload = GroqPayload.model_validate_json(raw_response)
        logger.info(f"Successfully validated JSON payload. Extracted {len(payload.entities)} entities and {len(payload.relationships)} relationships.")
        print("Extraction successful")
        
    except Exception as e:
        logger.error(f"Extraction failed with error: {e}")
        print("Extraction failed")

if __name__ == "__main__":
    main()
