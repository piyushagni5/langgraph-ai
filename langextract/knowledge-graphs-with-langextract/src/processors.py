import streamlit as st
from typing import List, Dict, Any
from src.extractors import extract_entities, extract_relationships
from src.graph_builder import build_graph_data
from src.query_processor import answer_query

def process_documents(documents: List[str], query: str = "") -> Dict[str, Any]:
    """Main pipeline that processes documents and builds knowledge graph"""
    
    # Extract entities and relationships
    st.info("🔍 Extracting entities...")
    entities = extract_entities(documents, query or "business entities and relationships")
    
    st.info("🔗 Extracting relationships...")
    relationships = extract_relationships(documents, entities)
    
    # Debug information
    st.success(f"✅ Found {len(entities)} entities and {len(relationships)} relationships")
    
    # Build graph data
    st.info("📊 Building knowledge graph...")
    nodes, edges = build_graph_data(entities, relationships)
    
    st.success(f"📈 Created graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Process query if provided
    query_results = None
    if query:
        st.info("🔍 Processing your query...")
        query_results = answer_query(entities, relationships, query)
    
    return {
        "entities": entities,
        "relationships": relationships,
        "graph_data": {"nodes": nodes, "edges": edges},
        "query_results": query_results
    }