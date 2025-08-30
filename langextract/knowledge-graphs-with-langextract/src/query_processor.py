from typing import List, Dict, Any

def answer_query(entities: List[Dict], relationships: List[Dict], query: str) -> Dict:
    """Search extracted data based on user query"""
    query_words = query.lower().split()
    
    relevant_entities = []
    relevant_relationships = []
    
    # Find matching entities
    for entity in entities:
        entity_text = entity["text"].lower()
        entity_attrs = str(entity.get("attributes", {})).lower()
        
        if any(word in entity_text or word in entity_attrs for word in query_words):
            relevant_entities.append(entity)
    
    # Find matching relationships
    for relationship in relationships:
        rel_text = relationship["text"].lower()
        rel_attrs = str(relationship.get("attributes", {})).lower()
        
        if any(word in rel_text or word in rel_attrs for word in query_words):
            relevant_relationships.append(relationship)
    
    return {
        "query": query,
        "entities": relevant_entities,
        "relationships": relevant_relationships,
        "entity_count": len(relevant_entities),
        "relationship_count": len(relevant_relationships)
    }