# src/document_processor.py (Renamed from processors.py)
import streamlit as st
import langextract as lx
import os
from typing import List, Dict, Any
from templates.few_shot_examples import get_dynamic_examples

def extract_entities_from_documents(documents: List[str], query: str = "") -> Dict[str, Any]:
    """Simplified processing pipeline using LangExtract's native features"""
    
    all_results = []
    all_entities = []
    
    st.info("🔍 Processing documents with LangExtract...")
    progress_bar = st.progress(0)
    
    for i, document in enumerate(documents):
        # Use LangExtract directly for better integration
        try:
            # Get dynamic examples based on query
            examples = get_dynamic_examples(query or "business information")
            
            # Create prompt description
            prompt_description = f"""
            Extract entities and information relevant to: {query or 'business and organizational information'}
            
            Focus on:
            - Companies and organizations
            - People and their roles
            - Locations and addresses
            - Financial information
            - Dates and time periods
            - Relationships between entities
            
            Be precise and extract only factual information present in the text.
            """
            
            result = lx.extract(
                text_or_documents=document,
                prompt_description=prompt_description.strip(),
                examples=examples,
                model_id="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
                max_workers=5,  # Reasonable number of workers
                extraction_passes=2  # Two passes for better recall
            )
            
            # Store the complete LangExtract result object
            all_results.append(result)
            
            # Extract entities for summary
            entities = [
                {
                    "text": extraction.extraction_text,
                    "class": extraction.extraction_class,
                    "attributes": extraction.attributes or {},
                    "document_index": i  # Track which document this came from
                }
                for extraction in result.extractions
            ]
            all_entities.extend(entities)
            
            # Update progress
            progress_bar.progress((i + 1) / len(documents))
            
        except Exception as e:
            st.error(f"Error processing document {i+1}: {str(e)}")
            st.error("Please check your API key and try again.")
            continue
    
    progress_bar.progress(1.0)
    st.success(f"✅ Processed {len(documents)} documents, found {len(all_entities)} entities")
    
    return {
        "langextract_results": all_results,
        "entities": all_entities,
        "document_count": len(documents),
        "original_documents": documents
    }

def analyze_documents_simple(documents: List[str], query: str) -> List[Dict]:
    """Simple entity extraction without complex processing"""
    
    all_entities = []
    
    for i, doc in enumerate(documents):
        try:
            examples = get_dynamic_examples(query)
            
            result = lx.extract(
                text_or_documents=doc,
                prompt_description=f"Extract entities related to: {query}",
                examples=examples,
                model_id="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            entities = [
                {
                    "text": extraction.extraction_text,
                    "class": extraction.extraction_class,
                    "attributes": extraction.attributes or {},
                    "document_index": i,
                    "confidence": getattr(extraction, 'confidence', 0.8)  # Default confidence
                }
                for extraction in result.extractions
            ]
            
            all_entities.extend(entities)
            
        except Exception as e:
            st.error(f"Failed to extract from document {i+1}: {str(e)}")
            continue
    
    return all_entities

def filter_entities_by_query(entities: List[Dict], query: str) -> List[Dict]:
    """Filter entities based on query keywords"""
    
    if not query:
        return entities
    
    query_words = [word.lower().strip() for word in query.split() if word.strip()]
    matching_entities = []
    
    for entity in entities:
        entity_text = entity.get("text", "").lower()
        entity_class = entity.get("class", "").lower()
        entity_attrs = str(entity.get("attributes", {})).lower()
        
        # Check if any query word matches entity text, class, or attributes
        if any(
            word in entity_text or 
            word in entity_class or 
            word in entity_attrs 
            for word in query_words
        ):
            matching_entities.append(entity)
    
    return matching_entities
