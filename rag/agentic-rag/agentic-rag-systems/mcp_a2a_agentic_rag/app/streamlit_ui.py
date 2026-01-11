# app/streamlit_ui.py
import streamlit as st
import asyncio
import tempfile
import os
from uuid import uuid4
from pathlib import Path
import sys

# Ensure project root is on sys.path when running from app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.a2a.agent_connect import AgentConnector
from a2a.client import A2ACardResolver
import httpx

st.set_page_config(
    page_title="MCPxA2AxAgentic-RAG System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 MCPxA2AxAgentic-RAG System")
st.markdown("*Intelligent document processing with A2A agents, MCP tools, and centralized model management*")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_agent_available" not in st.session_state:
    st.session_state.rag_agent_available = False

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
host_agent_url = st.sidebar.text_input("Host Agent URL", value="http://127.0.0.1:10001")
rag_agent_url = st.sidebar.text_input("RAG Agent URL", value="http://127.0.0.1:10002")

# Agent selection
selected_agent = st.sidebar.selectbox(
    "Select Agent",
    ["Host Orchestrator", "Agentic RAG Agent"],
    help="Choose which agent to interact with directly"
)

# Get the appropriate URL based on selection
agent_url = host_agent_url if selected_agent == "Host Orchestrator" else rag_agent_url

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Information")
st.sidebar.info("""
**Current Models:**
- **LLM**: Gemini 2.0 Flash Exp
- **Embeddings**: Nomic AI v1.5
- **Features**: Multi-candidate ranking, web integration
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Document Upload")

# PDF upload for RAG agent
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF for RAG processing",
    type=['pdf'],
    help="Upload a PDF to be processed by the Agentic RAG system using HuggingFace embeddings"
)

if uploaded_file and st.sidebar.button("📚 Ingest PDF"):
    with st.spinner("Processing PDF with HuggingFace embeddings..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Send ingestion request to RAG agent
        try:
            async def ingest_pdf():
                async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                    resolver = A2ACardResolver(
                        base_url=rag_agent_url.rstrip('/'),
                        httpx_client=httpx_client
                    )
                    card = await resolver.get_agent_card()
                    connector = AgentConnector(card)
                    return await connector.send_task(
                        message=f"Please ingest this PDF: {temp_path}",
                        session_id=st.session_state.session_id
                    )
            
            result = asyncio.run(ingest_pdf())
            st.sidebar.success("✅ PDF ingested successfully!")
            st.sidebar.write(result)
            
            # Mark RAG agent as available
            st.session_state.rag_agent_available = True
            
            # Clean up temp file
            os.unlink(temp_path)
            
        except Exception as e:
            st.sidebar.error(f"❌ Error ingesting PDF: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Main chat interface
st.markdown("### 💬 Chat Interface")
st.markdown("*Powered by Gemini 2.0 Flash Exp with multi-candidate RAG ranking*")

# Display chat history
for i, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# Chat input
if prompt := st.chat_input("Ask me anything about your documents or any topic..."):
    # Add user message to history
    st.session_state.chat_history.append(("user", prompt))
    st.chat_message("user").write(prompt)
    
    # Get response from selected agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Processing with Gemini..."):
            try:
                async def get_response():
                    async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                        resolver = A2ACardResolver(
                            base_url=agent_url.rstrip('/'),
                            httpx_client=httpx_client
                        )
                        card = await resolver.get_agent_card()
                        connector = AgentConnector(card)
                        
                        # For RAG agent, ensure we're asking about documents if available
                        if selected_agent == "Agentic RAG Agent" and st.session_state.rag_agent_available:
                            # Modify the prompt to be more specific about document queries
                            enhanced_prompt = prompt
                            if not any(word in prompt.lower() for word in ['document', 'resume', 'pdf', 'file', 'content']):
                                enhanced_prompt = f"Based on the ingested documents, {prompt}"
                            
                            return await connector.send_task(
                                message=enhanced_prompt,
                                session_id=st.session_state.session_id
                            )
                        else:
                            return await connector.send_task(
                                message=prompt,
                                session_id=st.session_state.session_id
                            )
                
                response = asyncio.run(get_response())
                st.write(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append(("assistant", response))
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))

# Sidebar status and controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

if st.sidebar.button("🔄 Check RAG Status"):
    try:
        async def check_status():
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                resolver = A2ACardResolver(
                    base_url=rag_agent_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                card = await resolver.get_agent_card()
                connector = AgentConnector(card)
                return await connector.send_task(
                    message="What is the current RAG system status?",
                    session_id=st.session_state.session_id
                )
        
        status = asyncio.run(check_status())
        st.sidebar.info(status)
        
        # Update RAG agent availability based on status
        if "empty" not in status.lower() and "no documents" not in status.lower():
            st.session_state.rag_agent_available = True
        else:
            st.session_state.rag_agent_available = False
            
    except Exception as e:
        st.sidebar.error(f"Error checking status: {str(e)}")

if st.sidebar.button("⚙️ Check Model Config"):
    try:
        async def check_config():
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                resolver = A2ACardResolver(
                    base_url=rag_agent_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                card = await resolver.get_agent_card()
                connector = AgentConnector(card)
                return await connector.send_task(
                    message="What is the current model configuration?",
                    session_id=st.session_state.session_id
                )
        
        config = asyncio.run(check_config())
        st.sidebar.info(config)
    except Exception as e:
        st.sidebar.error(f"Error checking config: {str(e)}")

# Debug section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🐛 Debug & Testing")

if st.sidebar.button("🧪 Test Function Calling"):
    try:
        async def test_function():
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                resolver = A2ACardResolver(
                    base_url=rag_agent_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                card = await resolver.get_agent_card()
                connector = AgentConnector(card)
                return await connector.send_task(
                    message="What skills are mentioned in my resume?",
                    session_id=st.session_state.session_id
                )
        
        result = asyncio.run(test_function())
        st.sidebar.success("✅ Function calling test completed!")
        st.sidebar.write("**Test Result:**", result)
    except Exception as e:
        st.sidebar.error(f"❌ Function calling test failed: {str(e)}")

# Add direct function calling test
if st.sidebar.button("🔍 Test Direct Function Call"):
    try:
        async def test_direct_function():
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                resolver = A2ACardResolver(
                    base_url=rag_agent_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                card = await resolver.get_agent_card()
                connector = AgentConnector(card)
                
                # Test with a very specific function call request
                test_message = "Please call the get_rag_system_status function and tell me the current status of the RAG system."
                return await connector.send_task(
                    message=test_message,
                    session_id=st.session_state.session_id
                )
        
        result = asyncio.run(test_direct_function())
        st.sidebar.success("✅ Direct function call test completed!")
        st.sidebar.write("**Test Result:**", result)
        
        # Check if the response indicates function calling worked
        if "RAG system" in result and ("empty" in result.lower() or "documents" in result.lower()):
            st.sidebar.success("🎉 Function calling is working correctly!")
        else:
            st.sidebar.warning("⚠️ Function calling may not be working as expected")
            
    except Exception as e:
        st.sidebar.error(f"❌ Direct function call test failed: {str(e)}")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("✅ Chat history cleared!")

# Debug info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Debug Info")
st.sidebar.info(f"""
**Session ID:** {st.session_state.session_id[:8]}...
**Chat Messages:** {len(st.session_state.chat_history)}
**RAG Agent:** {rag_agent_url}
**Host Agent:** {host_agent_url}
**RAG Available:** {'✅ Yes' if st.session_state.rag_agent_available else '❌ No'}
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Session ID:** `" + st.session_state.session_id + "`")
st.sidebar.markdown("**Current Agent:** " + selected_agent)
st.sidebar.markdown("**Models:** Gemini 2.0 + Nomic Embeddings")

# Add helpful tips
if not st.session_state.rag_agent_available:
    st.info("💡 **Tip:** Upload a PDF document first to enable RAG functionality. The agent will then be able to answer questions about your documents.")
else:
    st.success("✅ **RAG Ready:** You can now ask questions about your uploaded documents!")