# MCPxA2AxAgentic-RAG: Unified AI Agent System

A comprehensive AI agent system that combines **Model Context Protocol (MCP)**, **Agent-to-Agent (A2A)** communication, and **Agentic RAG** capabilities into one powerful platform for intelligent document processing, web search, and task automation.

## ⚠️ Important Notes (Python 3.13 Users)

If you're using **Python 3.13**, you may see some harmless warnings during shutdown:
```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

These are **expected warnings** due to anyio/MCP compatibility with Python 3.13 and do **not** affect system functionality. The system will still:
- ✅ Start successfully  
- ✅ Run all MCP tools correctly
- ✅ Handle agent communication properly
- ✅ Provide full functionality

## 🏗️ System Architecture

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

## ✨ Key Features

### 🤖 Agent-to-Agent (A2A) Communication
- **Host Agent**: Central orchestrator that coordinates all system components
- **Agentic RAG Agent**: Specialized in document processing and question answering
- **Dynamic Discovery**: Agents can discover and communicate with each other
- **Task Delegation**: Intelligent routing of tasks to appropriate specialized agents

### 🔧 Model Context Protocol (MCP) Integration
- **Web Search Tools**: Real-time information retrieval using SerpAPI
- **Terminal Tools**: Safe command execution and file operations
- **Unified Interface**: All tools accessible through standardized MCP protocol

### 📚 Agentic RAG Capabilities
- **Multi-candidate Retrieval**: Generates diverse context candidates using HuggingFace embeddings
- **Parallel Answer Generation**: Creates multiple answers simultaneously with Gemini
- **LLM-based Ranking**: Intelligently selects the best answer using Gemini self-evaluation
- **Document Processing**: Advanced RAG pipeline for document analysis
- **Multi-Source Integration**: Combines document knowledge with real-time web data

### 🧠 Centralized Model Management
- **LLM Model**: Gemini 2.0 Flash Exp for intelligent reasoning and coordination
- **Embedding Model**: Nomic AI nomic-embed-text-v1.5 for semantic similarity
- **Centralized Configuration**: All models managed via `model.py` for consistency
- **Easy Model Switching**: Change models system-wide from one configuration file

## 🚀 Quick Start

### Prerequisites
```bash
# Required API Keys
export SERPAPI_KEY="your_serpapi_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
```

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd mcp_a2a_agentic_rag
pip install -r requirements.txt
```

### Starting the System

**Start the system:**
```bash
python main.py
```

### System URLs
Once started, access the system at:
- **Host Agent API**: http://localhost:10001
- **RAG Agent API**: http://localhost:10002  
- **Streamlit UI**: `streamlit run app/streamlit_ui.py`

### Environment Setup
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key
SERPAPI_KEY=your_serpapi_key
```

## 🛠️ Available Tools

### Web Search Tools
- **web_search**: Search the web for current information
  ```json
  {
    "query": "latest AI developments",
    "max_results": 5
  }
  ```

### Terminal Tools
- **execute_command**: Execute safe terminal commands
  ```json
  {
    "command": "ls",
    "args": ["-la"]
  }
  ```

### Document Processing
- **upload_pdf**: Upload and process PDF documents
- **query_documents**: Ask questions about uploaded documents
- **get_document_summary**: Get summaries of processed documents

## 📁 Project Structure

```
mcp_a2a_agentic_rag/
├── 📄 main.py                    # System entry point
├── 📄 model.py                   # Centralized model configuration
├── 📄 requirements.txt           # Dependencies
├── 📄 mcp_config.json           # MCP server configuration
├── 📄 .env.example              # Environment variables template
├── 📄 kill_services.sh          # Service management script
│
├── 📁 agents/                   # A2A Agent implementations
│   ├── 📁 host_agent/          # Central orchestrator agent
│   │   ├── 📄 main.py          # Host agent server
│   │   ├── 📄 agent.py         # Host agent logic
│   │   ├── 📄 agent_executor.py
│   │   ├── 📄 instructions.txt  # Agent instructions
│   │   └── 📄 description.txt   # Agent description
│   │
│   └── 📁 agentic_rag_agent/   # Specialized RAG agent
│       ├── 📄 main.py          # RAG agent server
│       ├── 📄 agent.py         # RAG agent logic
│       ├── 📄 agent_executor.py
│       ├── 📄 rag_orchestrator.py
│       ├── 📄 instructions.txt
│       └── 📄 description.txt
│
├── 📁 mcp/                     # MCP Server implementations
│   └── 📁 servers/
│       ├── 📁 web_search_server/   # SerpAPI web search
│       │   └── 📄 web_search_server.py
│       └── 📁 terminal_server/     # Terminal command execution
│           └── 📄 terminal_server.py
│
├── 📁 utilities/               # Shared utilities
│   ├── 📁 a2a/                # A2A communication utilities
│   │   ├── 📄 agent_connect.py
│   │   ├── 📄 agent_discovery.py
│   │   └── 📄 agent_registry.json
│   │
│   ├── 📁 mcp/                # MCP utilities
│   │   ├── 📄 mcp_connect.py
│   │   └── 📄 mcp_discovery.py
│   │
│   └── 📁 common/             # Common utilities
│       └── 📄 file_loader.py
│
└── 📁 app/                    # Web interface
    └── 📄 streamlit_ui.py     # Streamlit web UI
```

## 🔧 Configuration

### Model Configuration (`model.py`)
All models are centrally configured for easy switching:
```python
MODEL_INFO = {
    "llm_model": "gemini-2.0-flash-exp",
    "llm_provider": "Google AI",
    "embed_model": "nomic-ai/nomic-embed-text-v1.5",
    "embed_provider": "HuggingFace"
}
```

### MCP Configuration (`mcp_config.json`)
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

## 🚀 Usage Examples

### 1. Basic Document Query
```bash
# Start the system
python main.py

# Use Streamlit UI or direct API calls to:
# 1. Upload a PDF document
# 2. Ask questions about the document
# 3. Get intelligent responses combining document content and web search
```

### 2. Web Search Integration
The system automatically uses web search when:
- Current information is needed
- Document doesn't contain relevant information
- Real-time data is requested

### 3. Agent Communication
The Host Agent automatically delegates tasks to the RAG Agent for:
- Document-related questions
- PDF processing tasks
- Knowledge base queries

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `GOOGLE_API_KEY` and `SERPAPI_KEY` are set in your `.env` file
2. **Port Conflicts**: Make sure ports 10001 and 10002 are available
3. **MCP Server Issues**: Check that all MCP servers start successfully in the logs

### Logs and Debugging
The system provides detailed logging during startup and operation. Monitor the console output for any issues with:
- MCP server connections
- Agent initialization
- API key validation

## 📋 Dependencies

Key dependencies include:
- `google-genai`: For LLM capabilities
- `google-adk`: For agent development
- `a2a-sdk`: For agent-to-agent communication
- `mcp`: Model Context Protocol
- `fastapi`: Web API framework
- `streamlit`: Web UI
- `httpx`: HTTP client for A2A communication

## 🔄 System Lifecycle

1. **Startup**: `main.py` starts both Host and RAG agents
2. **Discovery**: Agents register and discover each other via A2A
3. **Tool Loading**: MCP tools are loaded and cached
4. **Ready**: System accepts requests via API or web UI
5. **Processing**: Requests are routed to appropriate agents/tools
6. **Response**: Results are returned to user

## 📝 Notes

- The system is designed for development and testing environments
- All MCP servers run as child processes of the main system
- Agent communication uses HTTP-based A2A protocol
- Document processing uses local embeddings with FAISS
- Web search requires a valid SerpAPI key for full functionality
