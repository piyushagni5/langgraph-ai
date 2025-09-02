import streamlit as st
import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_gemini_key() -> Tuple[str, bool]:
    """Load Gemini API key from .env file or user input"""
    key = ""
    key_provided = False
    
    # Try to load from .env file
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        st.sidebar.success("✅ Using API key from .env file")
        key_provided = True
    else:
        # Fallback to user input
        key = st.sidebar.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio or add it to .env file"
        )
        
        if key:
            st.sidebar.success("✅ API key provided")
            key_provided = True
        else:
            st.sidebar.error("❌ No API key provided")
    
    return key, key_provided

def get_node_color(entity_class: str) -> str:
    """Assign colors based on entity class"""
    color_map = {
        "company": "#FF6B6B",
        "person": "#4ECDC4", 
        "location": "#45B7D1",
        "financial_metric": "#96CEB4",
        "date": "#FFEAA7",
        "character": "#DDA0DD",
        "emotion": "#FFB347",
        "relationship": "#98D8C8"
    }
    return color_map.get(entity_class, "#BDC3C7")