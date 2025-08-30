from streamlit_agraph import Node, Edge
from typing import List, Dict, Tuple
from src.utils import get_node_color

def build_graph_data(entities: List[Dict], relationships: List[Dict]) -> Tuple[List[Node], List[Edge]]:
    """Build graph nodes and edges from entities and relationships"""
    nodes = []
    edges = []
    entity_to_id = {}
    
    # Create nodes from entities
    for i, entity in enumerate(entities):
        node_id = f"entity_{i}"
        entity_to_id[entity["text"]] = node_id
        
        nodes.append(Node(
            id=node_id,
            label=entity["text"],
            size=25,
            color=get_node_color(entity["class"])
        ))
    
    # Create edges from relationships
    edge_count = 0
    for relationship in relationships:
        # Try to find entities mentioned in the relationship
        rel_text = relationship["text"].lower()
        connected_entities = []
        
        for entity_text, entity_id in entity_to_id.items():
            if entity_text.lower() in rel_text:
                connected_entities.append(entity_id)
        
        # Create edges between connected entities
        if len(connected_entities) >= 2:
            for i in range(len(connected_entities) - 1):
                edges.append(Edge(
                    source=connected_entities[i],
                    target=connected_entities[i + 1],
                    label=relationship.get("attributes", {}).get("type", "related_to")
                ))
                edge_count += 1
    
    # Fallback: create co-occurrence edges if no explicit relationships found
    if edge_count == 0 and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append(Edge(
                source=nodes[i].id,
                target=nodes[i + 1].id,
                label="co_occurs_with"
            ))
    
    return nodes, edges