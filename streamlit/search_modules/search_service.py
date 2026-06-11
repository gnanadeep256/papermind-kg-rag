import streamlit as st
import time
from src.hybrid_retriever import HybridRetriever
from src.answer_generator import GroundedAnswerGenerator

@st.cache_resource
def get_generator():
    """Initializes and caches the GroundedAnswerGenerator with HybridRetriever."""
    retriever = HybridRetriever()
    retriever.load()
    return GroundedAnswerGenerator(retriever)

def execute_search_query(query: str):
    """Executes a search query and returns the RAG result and execution latency."""
    generator = get_generator()
    t0 = time.time()
    res = generator.generate_answer(query)
    latency = time.time() - t0
    return res, latency
