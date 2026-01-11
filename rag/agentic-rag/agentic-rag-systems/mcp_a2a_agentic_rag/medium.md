# Building a Next-Generation AI System: A2A + MCP + Agentic AI Integration

## Introduction: The Future of Distributed AI Systems

In the rapidly evolving landscape of artificial intelligence, we're witnessing a paradigm shift toward more sophisticated, interconnected AI systems. Today, we'll explore how to build a cutting-edge platform that combines three powerful concepts:

1. **Agent-to-Agent (A2A) Communication** - Enabling AI agents to discover and communicate with each other
2. **Model Context Protocol (MCP)** - Standardizing how AI models interact with external tools and services
3. **Agentic AI with RAG** - Creating autonomous agents with enhanced retrieval-augmented generation capabilities

## Project Overview: MCPxA2AxAgentic-RAG System

Our system is a comprehensive AI platform that demonstrates the synergy between these technologies. It features:

### Core Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    MCPxA2AxAgentic-RAG System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Host Agent    │◄──►│ Agentic RAG     │                    │
│  │  (Port 10001)   │    │    Agent        │                    │
│  │                 │    │  (Port 10002)   │                    │
│  │ • Orchestrates  │    │                 │                    │
│  │ • Coordinates   │    │ • Document RAG  │                    │
│  │ • Integrates    │    │ • Q&A System    │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               MCP Tools Layer                           │    │
│  │                                                         │    │
│  │ ┌─────────────┐ ┌─────────────┐                        │    │
│  │ │Web Search   │ │Terminal     │                        │    │
│  │ │Server       │ │Server       │                        │    │
│  │ │             │ │             │                        │    │
│  │ │• SerpAPI    │ │• Commands   │                        │    │
│  │ │• Real-time  │ │• File Ops   │                        │    │
│  │ │• Search     │ │• System     │                        │    │
│  │ └─────────────┘ └─────────────┘                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features
- **Multi-Agent Communication**: Agents can discover and delegate tasks to each other
- **Unified Tool Access**: All external tools accessible through standardized MCP protocol
- **Advanced RAG Pipeline**: Multi-candidate retrieval with intelligent ranking
- **Real-time Integration**: Web search and system command capabilities
- **Centralized Model Management**: Easy switching between different AI models

## Step-by-Step Implementation Guide

### Phase 1: Project Foundation and Environment Setup

The foundation of any AI system starts with proper project structure and environment configuration.

#### 1.1 Project Structure Setup

First, let's establish the project directory structure:

```bash
mkdir mcp_a2a_agentic_rag
cd mcp_a2a_agentic_rag

# Create main directories
mkdir -p agents/{host_agent,agentic_rag_agent}
mkdir -p mcp/servers/{web_search_server,terminal_server}
mkdir -p utilities/{a2a,mcp,common}
mkdir -p app
```

#### 1.2 Environment Configuration

Create the essential configuration files:

**`.env` file for API keys:**
```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
SERPAPI_KEY=your_serpapi_key_here

# Optional configuration
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

**`requirements.txt` for dependencies:**
```txt
# Core AI libraries
google-adk==0.1.0
langchain-google-genai==2.0.6
langchain-huggingface==0.1.2
sentence-transformers==3.3.1

# MCP and communication
mcp==1.1.0
httpx==0.28.1
pydantic==2.10.3

# RAG and document processing
faiss-cpu==1.9.0.post1
PyPDF2==3.0.1
tiktoken==0.8.0

# UI and utilities
streamlit==1.41.0
python-dotenv==1.0.1
rich==13.9.4

# System utilities
asyncio-compat==0.2.4
numpy==2.2.1
```

### Phase 2: Centralized Model Management

One of the key design principles is centralized model configuration, making it easy to switch between different AI models across the entire system.

#### 2.1 Model Configuration (`model.py`)

```python
#!/usr/bin/env python3
"""
Centralized model configuration for MCPxA2AxAgentic-RAG system.
Uses Google Gemini for LLM and HuggingFace for embeddings.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

def get_llm_model(temperature: float = 0):
    """Get LLM model name string for Google ADK LlmAgent."""
    print("🔄 Using Google Gemini model")
    return "gemini-2.0-flash-exp"

def get_llm_model_instance(temperature: float = 0):
    """Get actual LLM model instance for direct use."""
    print("🔄 Creating Google Gemini model instance")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )

def get_embedding_model(use_huggingface: bool = True):
    """Get embedding model with safer HuggingFace models as default."""
    
    if use_huggingface:
        # Try multiple safe HuggingFace models in order of preference
        safe_models = [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "kwargs": {'device': 'cpu'},
                "encode_kwargs": {'normalize_embeddings': True}
            },
            {
                "name": "sentence-transformers/paraphrase-MiniLM-L6-v2", 
                "kwargs": {'device': 'cpu'},
                "encode_kwargs": {'normalize_embeddings': True}
            }
        ]
        
        for model_config in safe_models:
            try:
                print(f"🔄 Trying HuggingFace model: {model_config['name']}")
                embedding_model = HuggingFaceEmbeddings(
                    model_name=model_config["name"],
                    model_kwargs=model_config["kwargs"],
                    encode_kwargs=model_config["encode_kwargs"]
                )
                print(f"✅ Successfully loaded: {model_config['name']}")
                return embedding_model
                
            except Exception as e:
                print(f"❌ Failed to load {model_config['name']}: {e}")
                continue
    
    # Fallback to Google embeddings
    print("🔄 Using Google Generative AI embeddings as fallback")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# Model information for display
MODEL_INFO = {
    "llm": "Google Gemini 2.0 Flash Exp",
    "embedding": "HuggingFace Sentence Transformers",
    "description": "High-performance models for reasoning and semantic understanding"
}
```

**Key Features of This Approach:**
- **Single Source of Truth**: All model configurations in one place
- **Easy Switching**: Change models system-wide by modifying one file
- **Fallback Mechanisms**: Graceful handling of model loading failures
- **Performance Optimization**: CPU-optimized configurations for embeddings

### Phase 3: MCP (Model Context Protocol) Integration

MCP provides a standardized way for AI models to interact with external tools and services. Let's implement both the client-side integration and server implementations.

#### 3.1 MCP Discovery System (`utilities/mcp/mcp_discovery.py`)

```python
import json
import os
from typing import Dict, Any
from rich import print

class MCPDiscovery:
    """
    Discovers and manages MCP servers from configuration.
    Loads server configurations and provides access to their details.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize MCP discovery with configuration file.
        
        Args:
            config_file: Path to MCP configuration JSON file
        """
        if config_file is None:
            config_file = "mcp_config.json"
        
        self.config_file = config_file
        self.servers = {}
        self._load_config()
    
    def _load_config(self):
        """Load MCP server configurations from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.servers = config.get('mcpServers', {})
                    print(f"✅ Loaded {len(self.servers)} MCP servers from {self.config_file}")
            else:
                print(f"⚠️ MCP config file {self.config_file} not found. Using empty configuration.")
                self.servers = {}
        except Exception as e:
            print(f"❌ Error loading MCP config: {e}")
            self.servers = {}
    
    def list_servers(self) -> Dict[str, Any]:
        """Return dictionary of configured MCP servers."""
        return self.servers
    
    def get_server(self, name: str) -> Dict[str, Any]:
        """Get specific server configuration by name."""
        return self.servers.get(name, {})
```

#### 3.2 MCP Connector (`utilities/mcp/mcp_connect.py`)

```python
import asyncio
import logging
from utilities.mcp.mcp_discovery import MCPDiscovery
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from rich import print

# Configure logging for MCP cleanup issues to reduce noise during shutdown
logging.getLogger("mcp").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class MCPConnector:
    """
    Discovers MCP servers from configuration and loads their tools.
    Provides MCPToolsets compatible with Google's Agent Development Kit.
    """

    def __init__(self, config_file: str = None):
        self.discovery = MCPDiscovery(config_file=config_file)
        self.tools: list[MCPToolset] = []
        
    async def _load_all_tools(self):
        """
        Loads all tools from discovered MCP servers and caches them as MCPToolsets.
        """
        tools = []

        for name, server in self.discovery.list_servers().items():
            try:
                conn = StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=server["command"],
                        args=server["args"]
                    ),
                    timeout=5
                )
                
                try:
                    # Wrap toolset creation with timeout and error handling
                    toolset = await asyncio.wait_for(
                        MCPToolset(connection_params=conn).get_tools(),
                        timeout=10.0
                    )
                    
                    if toolset:
                        tools.extend(toolset)
                        print(f"✅ Loaded {len(toolset)} tools from {name}")
                    else:
                        print(f"⚠️ No tools found in {name}")
                        
                except asyncio.TimeoutError:
                    print(f"⚠️ Timeout loading tools from {name}")
                except Exception as tool_error:
                    print(f"⚠️ Error loading tools from {name}: {tool_error}")
                    
            except Exception as conn_error:
                print(f"⚠️ Error connecting to {name}: {conn_error}")

        self.tools = tools
        print(f"🔧 Total MCP tools loaded: {len(self.tools)}")
        
    async def get_tools(self):
        """Get all loaded MCP tools."""
        if not self.tools:
            await self._load_all_tools()
        return self.tools
```

#### 3.3 MCP Configuration (`mcp_config.json`)

```json
{
  "mcpServers": {
    "web_search": {
      "command": "python",
      "args": ["mcp/servers/web_search_server/web_search_server.py"]
    },
    "terminal": {
      "command": "python", 
      "args": ["mcp/servers/terminal_server/terminal_server.py"]
    }
  }
}
```

#### 3.4 Web Search MCP Server (`mcp/servers/web_search_server/web_search_server.py`)

```python
#!/usr/bin/env python3
"""
Web Search MCP Server using SerpAPI
Provides real-time web search capabilities through MCP protocol
"""

import asyncio
import json
import os
import sys
from typing import Any
import requests
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    EmbeddedResource,
    CallToolRequest,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
)

# Load environment variables
load_dotenv()

class WebSearchServer:
    def __init__(self):
        self.server = Server("web-search-server")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        
        if not self.serpapi_key:
            print("❌ SERPAPI_KEY not found in environment variables")
            sys.exit(1)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="web_search",
                        description="Search the web for current information using SerpAPI. Returns relevant search results with titles, snippets, and links.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to execute"
                                },
                                "num_results": {
                                    "type": "integer", 
                                    "description": "Number of results to return (default: 5, max: 10)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool calls"""
            if name == "web_search":
                return await self._web_search(arguments)
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )
    
    async def _web_search(self, arguments: dict) -> CallToolResult:
        """Perform web search using SerpAPI"""
        try:
            query = arguments.get("query", "")
            num_results = min(arguments.get("num_results", 5), 10)
            
            if not query:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: Query parameter is required")]
                )
            
            # SerpAPI request
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            organic_results = data.get("organic_results", [])
            
            if not organic_results:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"No results found for query: {query}")]
                )
            
            # Format results
            results = []
            for i, result in enumerate(organic_results[:num_results], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "No link")
                
                results.append(f"{i}. **{title}**\n   {snippet}\n   Link: {link}\n")
            
            formatted_results = f"🔍 **Web Search Results for:** {query}\n\n" + "\n".join(results)
            
            return CallToolResult(
                content=[TextContent(type="text", text=formatted_results)]
            )
            
        except Exception as e:
            error_msg = f"Web search error: {str(e)}"
            return CallToolResult(
                content=[TextContent(type="text", text=error_msg)]
            )

async def main():
    """Run the web search MCP server"""
    server_instance = WebSearchServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream, write_stream, server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 4: Agent-to-Agent (A2A) Communication System

A2A communication enables agents to discover and interact with each other dynamically, creating a distributed AI ecosystem.

#### 4.1 Agent Discovery System (`utilities/a2a/agent_discovery.py`)

```python
import json
import os
from typing import List, Optional
from a2a.types import AgentCard
from rich import print

class AgentDiscovery:
    """
    Agent discovery system that manages registration and discovery of A2A agents.
    Maintains a registry of available agents and their capabilities.
    """
    
    def __init__(self, registry_file: str = "utilities/a2a/agent_registry.json"):
        """
        Initialize agent discovery with registry file.
        
        Args:
            registry_file: Path to agent registry JSON file
        """
        self.registry_file = registry_file
        self.agents = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load agent registry from JSON file."""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                # Convert registry data to AgentCard objects
                for agent_id, agent_data in registry_data.get('agents', {}).items():
                    try:
                        self.agents[agent_id] = AgentCard(**agent_data)
                    except Exception as e:
                        print(f"⚠️ Error loading agent {agent_id}: {e}")
                        
                print(f"✅ Loaded {len(self.agents)} agents from registry")
            else:
                print(f"⚠️ Agent registry file {self.registry_file} not found. Creating empty registry.")
                self.agents = {}
                self._save_registry()
                
        except Exception as e:
            print(f"❌ Error loading agent registry: {e}")
            self.agents = {}
    
    def _save_registry(self):
        """Save current agent registry to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            
            registry_data = {
                "agents": {
                    agent_id: agent_card.model_dump(exclude_none=True) 
                    for agent_id, agent_card in self.agents.items()
                }
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving agent registry: {e}")
    
    async def register_agent(self, agent_card: AgentCard) -> bool:
        """
        Register a new agent in the discovery system.
        
        Args:
            agent_card: AgentCard object with agent details
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            self.agents[agent_card.agent_id] = agent_card
            self._save_registry()
            print(f"✅ Registered agent: {agent_card.name} ({agent_card.agent_id})")
            return True
        except Exception as e:
            print(f"❌ Error registering agent: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the discovery system.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                self._save_registry()
                print(f"✅ Unregistered agent: {agent_id}")
                return True
            else:
                print(f"⚠️ Agent {agent_id} not found in registry")
                return False
        except Exception as e:
            print(f"❌ Error unregistering agent: {e}")
            return False
    
    async def find_agent(self, agent_id: str) -> Optional[AgentCard]:
        """
        Find an agent by ID.
        
        Args:
            agent_id: ID of agent to find
            
        Returns:
            AgentCard if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    async def list_agent_cards(self) -> List[AgentCard]:
        """
        Get list of all registered agent cards.
        
        Returns:
            List of AgentCard objects
        """
        return list(self.agents.values())
    
    async def find_agents_by_capability(self, capability: str) -> List[AgentCard]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of AgentCard objects with the capability
        """
        return [
            agent_card for agent_card in self.agents.values()
            if capability in agent_card.capabilities
        ]
```

#### 4.2 Agent Connection System (`utilities/a2a/agent_connect.py`)

```python
import httpx
from a2a.client import A2AClient
from a2a.types import AgentCard

class AgentConnector:
    """
    Simple A2A Agent Connector that facilitates communication with other agents
    """
    
    def __init__(self, agent_card: AgentCard):
        """
        Initialize the connector with an agent card
        
        Args:
            agent_card: The AgentCard object containing agent connection details
        """
        self.agent_card = agent_card
    
    async def send_task(self, message: str, session_id: str) -> str:
        """
        Send a task/message to the connected agent
        
        Args:
            message: The message or task to send to the agent
            session_id: Session ID for conversation tracking
            
        Returns:
            Response from the agent
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                client = A2AClient(
                    base_url=self.agent_card.base_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                
                # Send the message to the agent
                response = await client.send_request(
                    message=message,
                    session_id=session_id
                )
                
                return response
                
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"
```

#### 4.3 Agent Registry Configuration (`utilities/a2a/agent_registry.json`)

```json
{
  "agents": {
    "agentic_rag_agent": {
      "agent_id": "agentic_rag_agent",
      "name": "Agentic RAG Agent",
      "description": "Specialized agent for document processing and question answering using advanced RAG techniques",
      "base_url": "http://localhost:10002",
      "capabilities": [
        "document_processing",
        "question_answering", 
        "pdf_analysis",
        "semantic_search",
        "multi_candidate_retrieval"
      ],
      "version": "1.0.0",
      "status": "active"
    }
  }
}
```

### Phase 5: Agentic RAG System Implementation

The RAG (Retrieval-Augmented Generation) system is the core of our document processing capabilities, featuring multi-candidate retrieval and intelligent ranking.

#### 5.1 RAG Orchestrator (`agents/agentic_rag_agent/rag_orchestrator.py`)

```python
# RAG Orchestrator - Multi-Agent Retrieval-Augmented Generation System
import os
import json
import faiss
import tiktoken
import requests
from dotenv import load_dotenv
from typing import List, Tuple
from PyPDF2 import PdfReader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from model import get_llm_model, get_embedding_model, get_llm_model_instance, get_embed_model_instance

load_dotenv()

# Use the centralized model configuration
ENC = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """Calculate the number of tokens in a text string."""
    return len(ENC.encode(text))

class PDFLoaderAgent:
    """Handles PDF document loading and text chunking for optimal retrieval."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

class EmbeddingAgent:
    """Handles text embeddings and vector indexing using FAISS."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.index = None
        self.texts = []

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return np.array([])

    def build_index(self, texts: List[str]):
        """Build FAISS index from text chunks."""
        self.texts = texts
        embeddings = self.create_embeddings(texts)
        
        if len(embeddings) > 0:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            print(f"✅ Built FAISS index with {len(texts)} documents")
        else:
            print("❌ Failed to create embeddings for index")

class RetrievalAgent:
    """Performs semantic search with diversity sampling."""
    
    def __init__(self, embedding_agent: EmbeddingAgent):
        self.embedding_agent = embedding_agent

    def retrieve_diverse_candidates(self, query: str, k: int = 10, num_candidates: int = 3) -> List[str]:
        """Retrieve diverse context candidates using semantic search."""
        if not self.embedding_agent.index:
            return ["No documents indexed."]

        try:
            # Create query embedding
            query_embedding = self.embedding_agent.create_embeddings([query])
            if len(query_embedding) == 0:
                return ["Error creating query embedding."]
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search for similar documents
            scores, indices = self.embedding_agent.index.search(query_embedding, k)
            
            # Get diverse candidates by clustering results
            candidates = []
            used_indices = set()
            
            for i in range(min(num_candidates, len(indices[0]))):
                if i < len(indices[0]) and indices[0][i] not in used_indices:
                    idx = indices[0][i]
                    if idx < len(self.embedding_agent.texts):
                        candidates.append(self.embedding_agent.texts[idx])
                        used_indices.add(idx)
            
            return candidates if candidates else ["No relevant documents found."]
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return [f"Retrieval error: {str(e)}"]

class QAAgent:
    """Generates answers using retrieved context."""
    
    def __init__(self):
        self.llm = get_llm_model_instance(temperature=0.1)

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on query and context."""
        try:
            prompt = f"""
Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class RankingAgent:
    """Evaluates and ranks multiple answer candidates using LLM."""
    
    def __init__(self):
        self.llm = get_llm_model_instance(temperature=0)

    def rank_answers(self, query: str, candidates: List[str]) -> str:
        """Rank multiple answer candidates and return the best one."""
        if len(candidates) == 1:
            return candidates[0]
        
        try:
            candidates_text = "\n\n".join([f"Answer {i+1}:\n{answer}" for i, answer in enumerate(candidates)])
            
            prompt = f"""
You are an expert evaluator. Given a question and multiple answer candidates, select the BEST answer based on:
1. Accuracy and correctness
2. Completeness and detail
3. Clarity and coherence
4. Relevance to the question

Question: {query}

{candidates_text}

Please respond with only the number (1, 2, 3, etc.) of the best answer, followed by a brief explanation of why you chose it.
"""

            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # Extract the selected answer number
            try:
                lines = result.strip().split('\n')
                first_line = lines[0].strip()
                
                # Try to extract number from first line
                import re
                numbers = re.findall(r'\d+', first_line)
                if numbers:
                    selected_idx = int(numbers[0]) - 1
                    if 0 <= selected_idx < len(candidates):
                        return candidates[selected_idx]
            except:
                pass
            
            # Fallback to first candidate if parsing fails
            return candidates[0]
            
        except Exception as e:
            print(f"Error in ranking: {e}")
            return candidates[0] if candidates else "Error in ranking process."

class RAGOrchestrator:
    """Coordinates all RAG agents for end-to-end document processing and Q&A."""
    
    def __init__(self):
        self.pdf_loader = PDFLoaderAgent()
        self.embedding_agent = EmbeddingAgent()
        self.retrieval_agent = RetrievalAgent(self.embedding_agent)
        self.qa_agent = QAAgent()
        self.ranking_agent = RankingAgent()
        self.is_initialized = False

    def initialize_with_documents(self, pdf_paths: List[str]):
        """Initialize the RAG system with PDF documents."""
        print("🔄 Initializing RAG system with documents...")
        
        all_texts = []
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                print(f"Loading {pdf_path}...")
                text = self.pdf_loader.load_pdf(pdf_path)
                if text:
                    chunks = self.pdf_loader.chunk_text(text)
                    all_texts.extend(chunks)
                    print(f"✅ Added {len(chunks)} chunks from {pdf_path}")
                else:
                    print(f"⚠️ No text extracted from {pdf_path}")
            else:
                print(f"❌ File not found: {pdf_path}")

        if all_texts:
            self.embedding_agent.build_index(all_texts)
            self.is_initialized = True
            print(f"✅ RAG system initialized with {len(all_texts)} total chunks")
        else:
            print("❌ No documents loaded. RAG system not initialized.")

    async def process_query(self, query: str) -> dict:
        """Process a query through the complete RAG pipeline."""
        if not self.is_initialized:
            return {
                "query": query,
                "answer": "RAG system not initialized. Please load documents first.",
                "candidates": [],
                "error": "System not initialized"
            }

        try:
            print(f"🔍 Processing query: {query}")
            
            # Step 1: Retrieve diverse context candidates
            print("📚 Retrieving context candidates...")
            contexts = self.retrieval_agent.retrieve_diverse_candidates(query, k=10, num_candidates=3)
            
            # Step 2: Generate answers for each context in parallel
            print("🤖 Generating answer candidates...")
            with ThreadPoolExecutor(max_workers=3) as executor:
                answer_futures = [
                    executor.submit(self.qa_agent.generate_answer, query, context)
                    for context in contexts
                ]
                candidates = [future.result() for future in answer_futures]
            
            # Step 3: Rank candidates and select best answer
            print("🏆 Ranking answer candidates...")
            best_answer = self.ranking_agent.rank_answers(query, candidates)
            
            print("✅ Query processing complete")
            
            return {
                "query": query,
                "answer": best_answer,
                "candidates": candidates,
                "contexts_used": len(contexts),
                "total_documents": len(self.embedding_agent.texts)
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "query": query,
                "answer": error_msg,
                "candidates": [],
                "error": str(e)
            }

# Global shared instance
_shared_rag_orchestrator = None

def get_shared_rag_orchestrator():
    """Get the shared RAG orchestrator instance (singleton pattern)."""
    global _shared_rag_orchestrator
    if _shared_rag_orchestrator is None:
        _shared_rag_orchestrator = RAGOrchestrator()
        print("[RAGOrchestrator] Initialized shared instance")
    return _shared_rag_orchestrator
```

### Phase 6: Host Agent Implementation

The Host Agent serves as the central orchestrator, coordinating between A2A agents and MCP tools.

#### 6.1 Host Agent Core (`agents/host_agent/agent.py`)

```python
from collections.abc import AsyncIterable
import json
from typing import Any
from uuid import uuid4
from utilities.a2a.agent_connect import AgentConnector
from utilities.a2a.agent_discovery import AgentDiscovery
from utilities.common.file_loader import load_instructions_file
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from rich import print as rprint
from rich.syntax import Syntax
from utilities.mcp.mcp_connect import MCPConnector
from a2a.types import AgentCard
from dotenv import load_dotenv
from model import get_llm_model, MODEL_INFO

load_dotenv()

class HostAgent:
    '''Enhanced orchestrator agent that coordinates A2A agents and MCP tools'''
    
    def __init__(self):
        self.system_instruction = load_instructions_file("agents/host_agent/instructions.txt")
        self.description = load_instructions_file("agents/host_agent/description.txt") 
        self.MCPConnector = MCPConnector()
        self.AgentDiscovery = AgentDiscovery()
        self._agent = None
        self._user_id = "host_agent_user"
        self._runner = None

    async def create(self):
        self._agent = await self._build_agent()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _list_agents(self) -> list[dict]:
        '''A2A tool: returns the list of dictionaries with agent card objects of registered A2A child agents'''
        cards = await self.AgentDiscovery.list_agent_cards()
        return [card.model_dump(exclude_none=True) for card in cards]

    async def _communicate_with_agent(self, agent_id: str, message: str, session_id: str = None) -> str:
        '''A2A tool: sends a message to a specific A2A agent and returns the response'''
        try:
            # Find the agent
            agent_card = await self.AgentDiscovery.find_agent(agent_id)
            if not agent_card:
                return f"Agent with ID '{agent_id}' not found. Available agents: {[card.agent_id for card in await self.AgentDiscovery.list_agent_cards()]}"
            
            # Create connector and send message
            connector = AgentConnector(agent_card)
            if not session_id:
                session_id = str(uuid4())
            
            response = await connector.send_task(message, session_id)
            return response
            
        except Exception as e:
            return f"Error communicating with agent {agent_id}: {str(e)}"

    async def _build_agent(self) -> LlmAgent:
        '''Builds the LLM agent with A2A and MCP tool integration'''
        tools = []
        
        # Add A2A communication tools
        tools.extend([
            FunctionTool(
                name="list_agents",
                description="List all registered A2A agents with their capabilities and status. Use this to discover available agents before delegating tasks.",
                func=self._list_agents
            ),
            FunctionTool(
                name="communicate_with_agent", 
                description="Send a message or task to a specific A2A agent. Use the agent_id from list_agents to target the right agent.",
                func=self._communicate_with_agent
            )
        ])
        
        # Add MCP tools
        try:
            mcp_tools = await self.MCPConnector.get_tools()
            tools.extend(mcp_tools)
            print(f"✅ Added {len(mcp_tools)} MCP tools to Host Agent")
        except Exception as e:
            print(f"⚠️ Error loading MCP tools: {e}")
        
        print(f"🔧 Total tools available to Host Agent: {len(tools)}")
        
        return LlmAgent(
            model_id=get_llm_model(),
            instructions=self.system_instruction,
            description=self.description,
            tools=tools
        )

    async def send(self, message: str, session_id: str = None) -> AsyncIterable[Any]:
        '''Send a message to the host agent and stream the response'''
        if not session_id:
            session_id = str(uuid4())
        
        if not self._agent:
            await self.create()
        
        async for chunk in self._runner.send_message_stream(
            user_id=self._user_id,
            session_id=session_id,
            message_content=message
        ):
            yield chunk

    def get_info(self) -> dict:
        '''Get agent information'''
        return {
            "name": "Host Agent",
            "description": self.description,
            "capabilities": [
                "Agent orchestration",
                "Task delegation", 
                "MCP tool integration",
                "A2A communication",
                "System coordination"
            ],
            "model_info": MODEL_INFO
        }
```

#### 6.2 Host Agent Instructions (`agents/host_agent/instructions.txt`)

```
You are the Host Agent, a sophisticated AI orchestrator responsible for coordinating multiple specialized agents and tools in the MCPxA2AxAgentic-RAG system.

## Your Core Responsibilities:

### 1. Agent Orchestration
- Discover and manage available A2A (Agent-to-Agent) agents
- Delegate tasks to appropriate specialized agents based on their capabilities
- Coordinate complex workflows across multiple agents

### 2. Tool Integration
- Access and utilize MCP (Model Context Protocol) tools for web search and system operations
- Combine agent capabilities with tool functionalities for comprehensive solutions

### 3. Task Analysis & Routing
- Analyze incoming user requests to determine the best approach
- Route document processing tasks to the Agentic RAG Agent
- Handle web search requests using MCP web search tools
- Execute system commands through MCP terminal tools

### 4. Communication Hub
- Serve as the central communication point for all system interactions
- Maintain context across multi-agent conversations
- Provide unified responses that combine insights from multiple sources

## Available Capabilities:

### A2A Agent Communication:
- `list_agents`: Discover available agents and their capabilities
- `communicate_with_agent`: Send tasks to specific agents

### MCP Tools:
- `web_search`: Real-time web search using SerpAPI
- `terminal`: Execute system commands and file operations

## Decision Making Guidelines:

1. **For Document Questions**: Always delegate to the 'agentic_rag_agent' for PDF analysis and document-based Q&A
2. **For Current Information**: Use web_search tool for real-time data and recent events
3. **For System Tasks**: Use terminal tool for file operations and command execution
4. **For Complex Tasks**: Combine multiple agents and tools as needed

## Response Format:
- Be clear about which agents/tools you're using
- Provide comprehensive answers that synthesize information from multiple sources
- Explain your reasoning when delegating tasks or using specific tools
- Always maintain a helpful and professional tone

Remember: You are the intelligent coordinator that makes the entire system work together seamlessly. Think strategically about how to best utilize the available resources to provide the most helpful response to the user.
```

### Phase 7: System Integration and Main Entry Point

Now let's create the main system entry point that coordinates all components.

#### 7.1 Main System Launcher (`main.py`)

```python
"""
Main entry point for the MCPxA2AxAgentic-RAG system.
Starts both the host agent and the agentic RAG agent.
"""
import asyncio
import signal
import sys
import logging
import os
from dotenv import load_dotenv
from agents.host_agent.main import serve_host_agent
from agents.agentic_rag_agent.main import serve_agentic_rag_agent

# Set environment variables to prevent segmentation faults from threading/parallel processing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Load environment variables from .env file
load_dotenv()

# Verify critical environment variables are loaded
required_env_vars = ["GOOGLE_API_KEY", "SERPAPI_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ Missing required environment variables: {missing_vars}")
    print("   Please check your .env file and ensure all required API keys are set.")
    sys.exit(1)

print("✅ Environment variables loaded successfully")

# Configure logging to reduce noise from MCP cleanup issues (Python 3.13 compatibility)
logging.getLogger("mcp.client").setLevel(logging.ERROR)
logging.getLogger("mcp.client.stdio").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

async def start_system():
    """Start both agents concurrently with proper error handling"""
    print("🚀 Starting MCPxA2AxAgentic-RAG System...")
    print("   Web UI will be available at:")
    print("   • Host Agent: http://localhost:10001")
    print("   • RAG Agent: http://localhost:10002")
    print("   • Streamlit UI: streamlit run app/streamlit_ui.py")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host_task = None
    rag_task = None
    
    try:
        print("Starting Host Agent on port 10001...")
        print("Starting Agentic RAG Agent on port 10002...")
        
        # Create tasks for both agents
        host_task = asyncio.create_task(serve_host_agent(host="localhost", port=10001))
        rag_task = asyncio.create_task(serve_agentic_rag_agent(host="localhost", port=10002))
        
        # Wait for both tasks to start
        print("⏳ Waiting for both servers to initialize...")
        await asyncio.sleep(8)
        
        print("✅ Both agents started successfully!")
        print("   System is now running...")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        print("\n🛑 Shutdown signal received...")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if host_task and not host_task.done():
            host_task.cancel()
        if rag_task and not rag_task.done():
            rag_task.cancel()
        
        print("🧹 System shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(start_system())
    except KeyboardInterrupt:
        print("\n🧹 System shutdown complete")
    except Exception as e:
        print(f"❌ System failed: {e}")
        sys.exit(1)
```

### Phase 8: User Interface Implementation

Finally, let's create a Streamlit-based user interface for easy interaction with the system.

#### 8.1 Streamlit UI (`app/streamlit_ui.py`)

```python
import streamlit as st
import asyncio
import httpx
import json
from typing import Dict, Any

class A2ASystemUI:
    """Streamlit UI for MCPxA2AxAgentic-RAG system"""
    
    def __init__(self):
        self.host_agent_url = "http://localhost:10001"
        self.rag_agent_url = "http://localhost:10002"
        
    async def send_to_host_agent(self, message: str, session_id: str = "streamlit_session") -> str:
        """Send message to host agent"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.host_agent_url}/send",
                    json={"message": message, "session_id": session_id}
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            return f"Error connecting to Host Agent: {str(e)}"
    
    async def send_to_rag_agent(self, message: str, session_id: str = "streamlit_session") -> str:
        """Send message to RAG agent"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.rag_agent_url}/send", 
                    json={"message": message, "session_id": session_id}
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            return f"Error connecting to RAG Agent: {str(e)}"

def main():
    st.set_page_config(
        page_title="MCPxA2AxAgentic-RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    ui = A2ASystemUI()
    
    # Title and description
    st.title("🤖 MCPxA2AxAgentic-RAG System")
    st.markdown("""
    This system demonstrates the integration of:
    - **Agent-to-Agent (A2A)** communication
    - **Model Context Protocol (MCP)** tools
    - **Agentic RAG** for document processing
    """)
    
    # Sidebar for system status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check agent status
        try:
            # Simple health check (you can implement actual health endpoints)
            st.success("✅ Host Agent (Port 10001)")
            st.success("✅ RAG Agent (Port 10002)")
        except:
            st.error("❌ Agents not responding")
        
        st.header("📖 Quick Guide")
        st.markdown("""
        **Host Agent Commands:**
        - "List available agents"
        - "Search the web for [topic]"
        - "What's the latest news about AI?"
        
        **RAG Agent Commands:**
        - "Analyze the uploaded document"
        - "What are the key points in the PDF?"
        - "Summarize the document"
        """)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎯 Host Agent")
        st.markdown("*Central orchestrator with web search and agent coordination*")
        
        host_message = st.text_area(
            "Send message to Host Agent:",
            placeholder="Ask me to search the web, coordinate with other agents, or run system commands...",
            height=100,
            key="host_input"
        )
        
        if st.button("Send to Host Agent", key="host_send"):
            if host_message:
                with st.spinner("Processing with Host Agent..."):
                    try:
                        response = asyncio.run(ui.send_to_host_agent(host_message))
                        st.success("Response from Host Agent:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("📚 RAG Agent")
        st.markdown("*Specialized document processing and Q&A*")
        
        rag_message = st.text_area(
            "Send message to RAG Agent:",
            placeholder="Ask questions about your documents, request analysis, or upload PDFs...",
            height=100,
            key="rag_input"
        )
        
        if st.button("Send to RAG Agent", key="rag_send"):
            if rag_message:
                with st.spinner("Processing with RAG Agent..."):
                    try:
                        response = asyncio.run(ui.send_to_rag_agent(rag_message))
                        st.success("Response from RAG Agent:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Document upload section
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF for analysis", type=['pdf'])
    
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info("You can now ask the RAG Agent questions about this document!")

if __name__ == "__main__":
    main()
```

## Deployment and Testing Guide

### Step 1: Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd mcp_a2a_agentic_rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Step 2: System Launch
```bash
# Start the system
python main.py

# In another terminal, start the UI (optional)
streamlit run app/streamlit_ui.py
```

### Step 3: Testing the System

**Test A2A Communication:**
```bash
curl -X POST http://localhost:10001/send \
  -H "Content-Type: application/json" \
  -d '{"message": "List all available agents", "session_id": "test_session"}'
```

**Test MCP Integration:**
```bash
curl -X POST http://localhost:10001/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Search the web for latest AI news", "session_id": "test_session"}'
```

**Test RAG Capabilities:**
```bash
curl -X POST http://localhost:10002/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Initialize with documents and answer questions", "session_id": "test_session"}'
```

## Conclusion

This comprehensive system demonstrates the power of combining:

1. **Agent-to-Agent Communication** - Creating a distributed AI ecosystem where agents can discover and collaborate
2. **Model Context Protocol** - Standardizing tool access for consistent integration
3. **Agentic RAG** - Advanced document processing with multi-candidate retrieval and intelligent ranking

The modular architecture makes it easy to:
- Add new agents with specialized capabilities
- Integrate additional MCP tools
- Scale the system horizontally
- Modify AI models system-wide

This represents the future of AI systems - not monolithic models, but intelligent, communicating agents working together to solve complex problems.

### Key Takeaways for Developers:

1. **Start with Clear Architecture** - Define your agent types and communication patterns early
2. **Centralize Configuration** - Make it easy to change models and settings across the system
3. **Plan for Scale** - Design with multiple agents and tools in mind from the beginning
4. **Error Handling** - Implement robust error handling for network communication and model failures
5. **User Experience** - Provide clear interfaces for both technical and non-technical users

The complete implementation provides a solid foundation for building next-generation AI systems that are more capable, flexible, and maintainable than traditional monolithic approaches.
