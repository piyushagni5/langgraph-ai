# app.py (Fixed Version)
import streamlit as st
import os
from typing import Dict, Any
from src.utils import load_gemini_key
from src.entity_extractor import extract_entities_from_documents
from src.display_manager import create_highlighted_html, render_entity_highlights, show_extraction_summary
from data.sample_documents import SAMPLE_DOCUMENTS

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="LangExtract Knowledge Extraction",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 LangExtract Knowledge Extraction")
    st.markdown("Transform unstructured text into highlighted, structured information using Google's LangExtract")
    
    # Load API key
    api_key, key_provided = load_gemini_key()
    
    if not key_provided:
        st.warning("⚠️ Please provide a Gemini API key to continue")
        st.stop()
    
    # Set environment variable for LangExtract
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Display sample documents info
    st.success(f"📚 Ready to process {len(SAMPLE_DOCUMENTS)} documents")
    
    # User query input
    user_query = st.text_input(
        "🔍 Enter your query (optional):",
        placeholder="e.g., 'company founders and locations' or 'financial information'"
    )
    
    # Process documents button
    if st.button("🚀 Process Documents & Extract Information", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                results = extract_entities_from_documents(SAMPLE_DOCUMENTS, user_query)
                
                # Display results in simplified tabs
                tab1, tab2, tab3 = st.tabs(["📋 Highlighted Text", "📊 Entity Summary", "🔍 Search Results"])
                
                with tab1:
                    st.subheader("📋 Text with Highlighted Entities")
                    
                    # Generate and display LangExtract's native visualization
                    try:
                        html_file, html_content = create_highlighted_html(
                            results["langextract_results"]
                        )
                        
                        st.info("💡 Entities are highlighted directly in the source text below:")
                        render_entity_highlights(html_content)
                        
                        # Provide download link for the HTML file
                        with open(html_file, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📥 Download Full Visualization",
                                f.read(),
                                "langextract_visualization.html",
                                "text/html"
                            )
                            
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
                        st.info("Showing entity summary instead:")
                        show_extraction_summary(results["entities"])
                
                with tab2:
                    st.subheader("📊 Extraction Summary")
                    show_extraction_summary(results["entities"])
                    
                    # Optional: Show raw extraction data
                    with st.expander("🔧 Raw Extraction Data"):
                        st.json(results["entities"])
                
                with tab3:
                    st.subheader("🔍 Query-Specific Results")
                    if user_query:
                        # Simple query matching
                        query_words = user_query.lower().split()
                        matching_entities = []
                        
                        for entity in results["entities"]:
                            entity_text = entity["text"].lower()
                            entity_attrs = str(entity.get("attributes", {})).lower()
                            
                            if any(word in entity_text or word in entity_attrs for word in query_words):
                                matching_entities.append(entity)
                        
                        st.metric("Matching Entities", len(matching_entities))
                        
                        if matching_entities:
                            for entity in matching_entities:
                                with st.expander(f"{entity['class'].title()}: {entity['text']}"):
                                    st.write(f"**Type:** {entity['class']}")
                                    if entity.get('attributes'):
                                        st.write("**Attributes:**")
                                        for key, value in entity['attributes'].items():
                                            st.write(f"- {key}: {value}")
                        else:
                            st.info("No entities match your query. Try different keywords.")
                    else:
                        st.info("Enter a query above and reprocess to see filtered results.")
                        
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.error("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()