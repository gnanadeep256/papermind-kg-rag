import streamlit as st
from src.llm import get_groq_client
from groq import Groq

@st.cache_resource
def get_llm() -> Groq:
    """Wraps get_groq_client with Streamlit's cache_resource to prevent API key re-creation."""
    return get_groq_client()
