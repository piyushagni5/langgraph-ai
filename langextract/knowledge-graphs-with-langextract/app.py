import streamlit as st
import os
from typing import Dict, Any
from src.utils import load_gemini_key
from src.visualizer import display_agraph
from src.processors import process_documents
from data.sample_documents import SAMPLE_DOCUMENTS

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="LangExtract Knowledge Graph Builder",
        page_icon="🕸️",
        layout="wide"  # Use wide layout for full screen
    )
    
    # Custom CSS for full-width graphs
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .element-container {
        width: 100% !important;
    }
    iframe {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🕸️ LangExtract Knowledge Graph Builder")
    st.markdown("Transform unstructured text into interactive knowledge graphs using Google's LangExtract")
    
    # Load API key
    api_key, key_provided = load_gemini_key()
    
    if not key_provided:
        st.warning("⚠️ Please provide a Gemini API key to continue")
        st.stop()
    
    # Set environment variable for LangExtract
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Display sample documents info
    st.success(f"📚 Processing {len(SAMPLE_DOCUMENTS)} documents")
    
    # User query input
    user_query = st.text_input(
        "🔍 Enter your query (optional):",
        placeholder="e.g., 'company founders and locations' or 'financial information'"
    )
    
    # Process documents button
    if st.button("🚀 Process Documents & Build Knowledge Graph", type="primary"):
        with st.spinner("Processing documents..."):
            results = process_documents(SAMPLE_DOCUMENTS, user_query)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Knowledge Graph", "🏷️ Entities", "🔗 Relationships", "🔍 Query Results"])
        
        with tab1:
            st.subheader("Interactive Knowledge Graph")
            nodes, edges = results["graph_data"]["nodes"], results["graph_data"]["edges"]
            
            if nodes:
                # Use full width container for the graph
                with st.container():
                    selected_node = display_agraph(nodes, edges)
                    if selected_node:
                        st.info(f"Selected: {selected_node}")
            else:
                st.warning("No entities found to visualize")
        
        with tab2:
            st.subheader("Extracted Entities")
            for i, entity in enumerate(results["entities"]):
                with st.expander(f"{entity['class'].title()}: {entity['text']}"):
                    st.json(entity)
        
        with tab3:
            st.subheader("Extracted Relationships")
            if results["relationships"]:
                for i, relationship in enumerate(results["relationships"]):
                    with st.expander(f"Relationship: {relationship['text'][:50]}..."):
                        st.json(relationship)
            else:
                st.info("No explicit relationships found. Graph uses co-occurrence connections.")
        
        with tab4:
            st.subheader("Query Results")
            if results["query_results"] and user_query:
                query_res = results["query_results"]
                st.metric("Matching Entities", query_res["entity_count"])
                st.metric("Matching Relationships", query_res["relationship_count"])
                
                if query_res["entities"]:
                    st.write("**Relevant Entities:**")
                    for entity in query_res["entities"]:
                        st.write(f"- {entity['text']} ({entity['class']})")
                
                if query_res["relationships"]:
                    st.write("**Relevant Relationships:**")
                    for rel in query_res["relationships"]:
                        st.write(f"- {rel['text']}")
            else:
                st.info("Enter a query above and reprocess to see filtered results")

if __name__ == "__main__":
    main()