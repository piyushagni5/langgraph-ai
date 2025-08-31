import langextract as lx
import streamlit as st
import textwrap
from typing import Dict, Any, List
from templates.few_shot_examples import get_dynamic_examples
import os
from dotenv import load_dotenv

load_dotenv()

def document_extractor_tool(text: str, query: str) -> Dict[str, Any]:
    """
    Extract entities and relationships from text using LangExtract
    with dynamic few-shot examples based on query keywords.
    """
    
    # Build clean prompt
    prompt = textwrap.dedent(f"""
        You are an expert information extractor. Your task is to pull out 
        relevant entities and relationships from the given text.
        
        Focus on: {query}
        
        Extract entities with their types and meaningful attributes.
        Identify relationships between entities with clear connection types.
        Use exact text for extractions. Do not paraphrase or overlap entities.
    """)
    
    # Get dynamic examples based on query
    examples = get_dynamic_examples(query)
    
    # Extract using LangExtract
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Log results for debugging
        st.write(f"Debug: Extracted {len(result.extractions)} entities")
        
        # Normalize output
        normalized_extractions = []
        for extraction in result.extractions:
            normalized_extractions.append({
                "text": extraction.extraction_text,
                "class": extraction.extraction_class,
                "attributes": extraction.attributes or {}
            })
        
        return {
            "extractions": normalized_extractions,
            "source_text": text
        }
        
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return {"extractions": [], "source_text": text}

def extract_entities(documents: List[str], query: str) -> List[Dict]:
    """Extract entities from multiple documents"""
    all_entities = []
    
    for doc in documents:
        result = document_extractor_tool(doc, f"{query} - extract entities")
        all_entities.extend(result.get("extractions", []))
    
    return all_entities

def extract_relationships(documents: List[str], entities: List[Dict]) -> List[Dict]:
    """Extract relationships between entities"""
    all_relationships = []
    
    for doc in documents:
        # Create a context-aware prompt for relationship extraction
        entity_names = [e["text"] for e in entities]
        relationship_query = f"Extract relationships and connections between entities. Focus on: founder relationships, CEO positions, company locations, employment, partnerships"
        
        result = document_extractor_tool(doc, relationship_query)
        
        # Process extractions to identify relationships
        for extraction in result.get("extractions", []):
            # More flexible relationship detection
            extraction_class = extraction["class"].lower()
            if any(keyword in extraction_class for keyword in ["relationship", "connection", "founded", "ceo", "employment", "partnership", "located", "headquartered"]):
                all_relationships.append(extraction)
            # Also check if extraction text suggests a relationship
            elif any(keyword in extraction["text"].lower() for keyword in ["founded by", "ceo of", "located in", "headquartered", "worked at", "employed by"]):
                # Convert to relationship format
                extraction["class"] = "relationship"
                all_relationships.append(extraction)
    
    return all_relationships