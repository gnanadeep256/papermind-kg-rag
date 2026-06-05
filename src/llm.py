import os
from groq import Groq
from dotenv import load_dotenv
from src.utils.logger import logger

load_dotenv()

_client = None

def get_groq_client() -> Groq:
    """
    Initializes and returns a singleton Groq client instance.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not defined in the environment variables.")
        _client = Groq(api_key=api_key)
    return _client

def query_groq_json(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """
    Queries the Groq API for a chat completion, enforcing JSON output.
    If the requested model fails (e.g. because it's unavailable), 
    it falls back to 'llama-3.3-70b-versatile'.

    Args:
        prompt: Prompt text to send.
        model_name: The configured model identifier.
        temperature: Sampling temperature.

    Returns:
        The raw JSON string from the model response.
    """
    client = get_groq_client()
    
    active_model = model_name
        
    try:
        logger.info(f"Querying Groq with model: {active_model}")
        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq API call failed for model {active_model}: {e}")
        
        # Select fallback model dynamically based on what was tried
        if active_model == "qwen/qwen3-32b":
            fallback_model = "llama-3.3-70b-versatile"
        elif active_model == "llama-3.3-70b-versatile":
            fallback_model = "llama-3.1-8b-instant"
        else:
            fallback_model = "llama-3.3-70b-versatile"
            
        logger.info(f"Falling back to model: {fallback_model}")
        try:
            response = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as fe:
            logger.error(f"Fallback to {fallback_model} failed: {fe}")
            raise fe
