from typing import List, TypedDict


class GraphState(TypedDict):
    """State object for workflow containing query, documents, and control flags."""
    question: str           # User's original query
    generation: str         # LLM-generated response  
    web_search: bool       # Control flag for web search requirement
    documents: List[str]   # Retrieved document context
