# src/display_manager.py
import streamlit as st
import streamlit.components.v1 as components
import langextract as lx
import os
from typing import List, Dict, Any

def create_highlighted_html(extraction_results: List, output_dir: str = "./data/outputs"):
    """Generate LangExtract's native HTML visualization"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save extraction results to JSONL
        output_file = os.path.join(output_dir, "extraction_results.jsonl")
        lx.io.save_annotated_documents(extraction_results, 
                                       output_name="extraction_results.jsonl", 
                                       output_dir=output_dir)
        
        # Generate HTML visualization
        html_content = lx.visualize(output_file)
        
        # Save HTML file
        html_file = os.path.join(output_dir, "visualization.html")
        with open(html_file, "w", encoding='utf-8') as f:
            if hasattr(html_content, 'data'):
                f.write(html_content.data)  # For Jupyter/Colab environments
            else:
                f.write(str(html_content))
        
        return html_file, html_content
    
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None, None

def render_entity_highlights(html_content: str):
    """Display the LangExtract visualization in Streamlit"""
    
    if html_content is None:
        st.error("No visualization content available")
        return
    
    try:
        # Extract just the content if it's wrapped
        if hasattr(html_content, 'data'):
            html_to_display = html_content.data
        else:
            html_to_display = str(html_content)
        
        # Display in Streamlit using components
        components.html(html_to_display, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error displaying visualization: {str(e)}")

def show_extraction_summary(entities: List[Dict]):
    """Display a simple entity summary alongside the visualization"""
    
    if not entities:
        st.warning("No entities found to display")
        return
    
    st.subheader("📊 Extraction Summary")
    
    # Entity count by class
    entity_counts = {}
    for entity in entities:
        class_name = entity.get("class", "unknown")
        entity_counts[class_name] = entity_counts.get(class_name, 0) + 1
    
    # Display metrics
    if entity_counts:
        cols = st.columns(min(len(entity_counts), 4))
        for i, (class_name, count) in enumerate(entity_counts.items()):
            with cols[i % 4]:
                st.metric(class_name.replace('_', ' ').title(), count)
    
    # Entity details in expander
    with st.expander("🔍 View All Entities"):
        for i, entity in enumerate(entities):
            st.write(f"**{i+1}. {entity.get('text', 'N/A')}** ({entity.get('class', 'unknown')})")
            if entity.get('attributes'):
                for key, value in entity['attributes'].items():
                    st.write(f"   - {key}: {value}")
            if i < len(entities) - 1:  # Don't add divider after last item
                st.divider()

def display_simple_text_view(entities: List[Dict], original_texts: List[str]):
    """Fallback: Display entities in a simple text format if visualization fails"""
    
    st.subheader("📝 Extracted Entities (Text View)")
    
    for i, text in enumerate(original_texts):
        with st.expander(f"Document {i+1}"):
            st.text_area(
                f"Original Text {i+1}:",
                text,
                height=150,
                disabled=True
            )
            
            # Show entities found in this document
            doc_entities = [e for e in entities if e.get('document_index') == i]
            if doc_entities:
                st.write("**Entities found:**")
                for entity in doc_entities:
                    st.write(f"• {entity['text']} ({entity['class']})")
            else:
                st.write("*No entities extracted from this document*")
