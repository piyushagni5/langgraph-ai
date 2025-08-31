import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from typing import List, Dict, Any
from src.utils import get_node_color

def display_agraph(nodes: List[Node], edges: List[Edge]):
    """Display interactive graph using streamlit-agraph"""
    config = Config(
        width=1200,    # Increased width for better full-screen usage
        height=700,    # Increased height for better visibility
        directed=True,
        physics=True,
        hierarchical=False,
        highlight_color="#F7A7A6",
        collapsible=False,
        node_label_property="label"
    )
    
    return agraph(nodes=nodes, edges=edges, config=config)

def format_output_for_graph(data: Dict[str, Any]) -> tuple:
    """Convert extraction results to nodes and edges for visualization"""
    nodes = []
    edges = []
    
    # Create nodes from entities
    entity_to_id = {}
    for i, entity in enumerate(data.get("extractions", [])):
        node_id = f"entity_{i}"
        entity_to_id[entity["text"]] = node_id
        
        nodes.append(Node(
            id=node_id,
            label=entity["text"],
            size=25,
            color=get_node_color(entity["class"])
        ))
    
    return nodes, edges