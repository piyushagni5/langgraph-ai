from typing import List, Dict, Any

def analyze_user_query(entities: List[Dict], relationships: List[Dict], query: str) -> Dict:
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

def find_matching_entities(entities: List[Dict], query: str) -> List[Dict]:
    """Filter entities that match the query"""
    query_words = query.lower().split()
    matching_entities = []
    
    for entity in entities:
        entity_text = entity["text"].lower()
        entity_attrs = str(entity.get("attributes", {})).lower()
        
        if any(word in entity_text or word in entity_attrs for word in query_words):
            matching_entities.append(entity)
    
    return matching_entities

def score_relevance(entities: List[Dict], query: str) -> List[Dict]:
    """Score and rank entities by relevance to query"""
    query_words = [word.lower() for word in query.split()]
    scored_entities = []
    
    for entity in entities:
        score = 0
        entity_text = entity["text"].lower()
        entity_attrs = str(entity.get("attributes", {})).lower()
        
        # Calculate relevance score
        for word in query_words:
            if word in entity_text:
                score += 2  # Higher weight for text matches
            if word in entity_attrs:
                score += 1  # Lower weight for attribute matches
        
        if score > 0:
            entity_with_score = entity.copy()
            entity_with_score["relevance_score"] = score
            scored_entities.append(entity_with_score)
    
    # Sort by relevance score (highest first)
    return sorted(scored_entities, key=lambda x: x["relevance_score"], reverse=True)
