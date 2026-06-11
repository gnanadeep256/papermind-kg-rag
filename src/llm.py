import os
import sys
import time
import requests
from typing import Any, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from functools import lru_cache

# Standard python singletons for CLI / non-streamlit contexts
_groq_client = None

def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

def _load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not defined in the environment variables.")
    logger.info("Initializing Groq client singleton.")
    return Groq(api_key=api_key)

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    return _load_embedding_model(model_name)

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = _load_groq_client()
    return _groq_client

def query_gemini_api(prompt: str, system_instruction: str, model: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Makes direct HTTP request to the Google Gemini API using standard key variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not defined.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    
    if system_instruction:
        data["systemInstruction"] = {
            "parts": [
                {"text": system_instruction}
            ]
        }
        
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    res_json = response.json()
    
    try:
        text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        usage = res_json.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        return text, prompt_tokens, completion_tokens
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse Gemini API response: {res_json}. Error: {e}")
        raise ValueError("Invalid response format from Gemini API.")

def query_groq_api(prompt: str, system_instruction: str, model: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Makes client request to the centralized Groq SDK."""
    client = get_groq_client()
    
    # Map generic model name to Groq model ID
    if model in ["llama-3", "llama3-70b-8192"]:
        model = "llama-3.3-70b-versatile"
    elif model in ["llama-3.1-8b", "llama3-8b-8192"]:
        model = "llama-3.1-8b-instant"
        
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    text = response.choices[0].message.content
    prompt_tokens = None
    completion_tokens = None
    if hasattr(response, "usage") and response.usage:
        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)
    return text, prompt_tokens, completion_tokens

def query_groq_json(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Queries the Groq API for a chat completion, enforcing JSON output."""
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
        
        # Select fallback model dynamically
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
