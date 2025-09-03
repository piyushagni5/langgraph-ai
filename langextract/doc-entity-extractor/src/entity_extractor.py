# src/document_processor.py (Renamed from processors.py)
import streamlit as st
import langextract as lx
import os
from typing import List, Dict, Any
from templates.few_shot_examples import get_dynamic_examples
import textwrap

def extract_entities_from_documents(documents: List[str], query: str = "") -> Dict[str, Any]:
    """Simplified processing pipeline using LangExtract's native features"""
    
    all_results = []
    all_entities = []
    
    st.info("🔍 Processing documents with LangExtract...")
    progress_bar = st.progress(0) 
    
    for i, document in enumerate(documents):
        # Use LangExtract directly for better integration
        try:
            
            # Create prompt description
            prompt_description = textwrap.dedent(f"""
            You are an expert information extractor. Your task is to pull out 
            relevant entities and relationships from the given text.
            
            Focus on: {query}
            
            Extract entities with their types and meaningful attributes.
            Identify relationships between entities with clear connection types.
            Use exact text for extractions. Do not paraphrase or overlap entities.
            """)

            # Get dynamic examples based on query
            examples = get_dynamic_examples(query or "business information")
            
            result = lx.extract(
                text_or_documents=document,
                prompt_description=prompt_description.strip(),
                examples=examples,
                model_id="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
                # max_workers=5,  # Reasonable number of workers
                # extraction_passes=2  # Two passes for better recall
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