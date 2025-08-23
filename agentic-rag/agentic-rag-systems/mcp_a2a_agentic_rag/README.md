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

The main.py script has been enhanced with logging configuration to minimize these warnings during normal operation.

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
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │    │
│  │ │Web Search   │ │Terminal     │ │HTTP Client  │        │    │
│  │ │Server       │ │Server       │ │Server       │        │    │
│  │ │             │ │             │ │             │        │    │
│  │ │• SerpAPI    │ │• Commands   │ │• HTTP Reqs  │        │    │
│  │ │• Real-time  │ │• File Ops   │ │• Webhooks   │        │    │
│  │ │• Search     │ │• System     │ │• APIs       │        │    │
│  │ └─────────────┘ └─────────────┘ └─────────────┘        │    │
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
- **HTTP Tools**: Web scraping, API interactions, and HTTP requests
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

## 🧪 Testing the System

### 1. Quick Integration Test
Test all MCP servers and connections:
```bash
python test_mcp_integration.py --test
```

This will:
- ✅ Validate MCP server configurations
- ✅ Test connections to all servers
- ✅ Verify tool availability
- ✅ Run sample operations for each tool type

**Expected Output:**
```
Testing MCP Integration...
✓ Web Search Server: Connected successfully
✓ Terminal Server: Connected successfully  
✓ HTTP Server: Connected successfully
✓ All tools accessible and functional
✓ Integration test completed successfully
```

### 2. Interactive Demo
Run a comprehensive demonstration:
```bash
python demo_integration.py

# Or for cleaner output (recommended):
python clean_demo.py
```

This demonstrates:
- 🔍 Web search capabilities (requires SERPAPI_KEY)
- 📁 File system operations
- 🌐 HTTP client functionality
- 🤖 Agent coordination workflows

**What to Expect:**
- Real web search results from SerpAPI (if key is set)
- File listing and reading operations
- HTTP requests to test endpoints
- A2A agent communication examples

### 3. Manual Server Testing
Test individual MCP servers in isolation:
```bash
# Test Web Search Server (requires SERPAPI_KEY)
python mcp/servers/web_search_server/web_search_server.py

# Test Terminal Server  
python mcp/servers/terminal_server/terminal_server.py

# Test HTTP Server
python mcp/servers/streamable_http_server/streamable_http_server.py
```

**Manual Testing Commands:**
Each server accepts JSON-RPC commands via stdin. Example:
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

### 4. System Health Check
Monitor system status and validate environment:
```bash
python test_mcp_integration.py --setup
```

This checks:
- Environment variables (API keys)
- MCP server availability
- Python dependencies
- Model configurations

### 5. Complete Testing Workflow

For a comprehensive test of the entire system:

```bash
# Step 1: Environment check
python test_mcp_integration.py --setup

# Step 2: MCP integration test
python test_mcp_integration.py --test

# Step 3: Interactive demo
python demo_integration.py

# Step 4: Full system test
python test_mcp_integration.py --start
```

### 6. Automated Test Suite

Run all tests automatically:
```bash
# Run all tests with output capturing
python -m pytest tests/ -v --capture=no

# Or use the built-in test runner
python test_structure.py
```

### 7. Performance Testing

Test system performance and load:
```bash
# Test concurrent MCP connections
python -c "
import asyncio
from utilities.mcp.mcp_client import MCPClient

async def test_concurrent():
    clients = [MCPClient() for _ in range(5)]
    tasks = []
    for i, client in enumerate(clients):
        task = client.call_tool('terminal', 'execute_command', 
                               {'command': 'echo', 'args': [f'Test {i}']})
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    print(f'Completed {len(results)} concurrent requests')

asyncio.run(test_concurrent())
"
```

## 🏃‍♂️ Running the Full System

### Option 1: Integrated Startup (Recommended)
```bash
python test_mcp_integration.py --start
```

This automatically:
- Tests MCP integration
- Starts all required servers
- Launches A2A agents
- Provides status monitoring

### Option 2: Manual Startup
```bash
# Terminal 1 - Start the complete system
python main.py
```

This starts:
- Host Orchestrator Agent (port 10001)
- Agentic RAG Agent (port 10002)
- All MCP servers in background

### Option 3: With Web UI
```bash
# Start system
python main.py

# In another terminal - Launch Streamlit UI
streamlit run app/streamlit_ui.py
```

### Access Points
- **Host Agent**: http://localhost:10001
- **Agentic RAG Agent**: http://localhost:10002
- **Web UI**: http://localhost:8501 (if using Streamlit)

## 🛠️ Available Tools & Testing Examples

### Web Search Tools
- **web_search**: Search the web for current information
  ```json
  {
    "query": "latest AI developments",
    "max_results": 5
  }
  ```
  **Test Command**:
  ```bash
  python -c "from utilities.mcp.mcp_client import MCPClient; client = MCPClient(); result = client.call_tool('web_search', 'web_search', {'query': 'AI news', 'max_results': 3}); print(result)"
  ```

### Terminal Tools
- **execute_command**: Execute safe terminal commands
  ```json
  {
    "command": "ls",
    "args": ["-la"],
    "working_directory": "."
  }
  ```
  **Test Command**:
  ```bash
  python -c "from utilities.mcp.mcp_client import MCPClient; client = MCPClient(); result = client.call_tool('terminal', 'execute_command', {'command': 'echo', 'args': ['Hello MCP']}); print(result)"
  ```
- **list_files**: List files and directories
- **read_file**: Read text file contents

### HTTP Tools
- **http_request**: Make custom HTTP requests
- **fetch_webpage**: Fetch and extract text from webpages
- **api_call**: Make JSON API calls with authentication
  
  **Test Command**:
  ```bash
  python -c "from utilities.mcp.mcp_client import MCPClient; client = MCPClient(); result = client.call_tool('http', 'fetch_webpage', {'url': 'https://httpbin.org/json'}); print(result)"
  ```

## 📋 Example Use Cases

### 1. Document Analysis + Web Research
```
User: "Analyze the latest AI research paper and compare it with current industry trends"

System Flow:
1. Host Agent receives request
2. Delegates document analysis to Agentic RAG Agent
3. Uses MCP web search for current trends
4. Synthesizes both sources for comprehensive answer
```

### 2. File Operations + API Integration
```
User: "Read my config file and update the external service"

System Flow:
1. Host Agent uses MCP terminal tools to read config
2. Uses MCP HTTP tools to call external API
3. Provides status update and recommendations
```

### 3. Multi-Modal Research
```
User: "Find information about quantum computing from my documents and latest news"

System Flow:
1. Agentic RAG searches internal documents
2. MCP web search finds recent news
3. Host Agent combines and presents unified results
```

### Enhanced Workflow with Centralized Models:
1. **User uploads PDF** via Streamlit UI
2. **RAG Agent processes** the document:
   - Extracts text and creates ~500 token chunks
   - Generates 768-dimensional embeddings using Nomic AI model
   - Builds searchable FAISS index for efficient similarity search
3. **User asks question** about the document
4. **Host Agent coordinates** using Gemini 2.0:
   - Routes to RAG Agent if document-related
   - Uses MCP tools for web search if needed
5. **RAG Agent provides intelligent answer**:
   - Retrieves 3 sets of diverse candidate contexts
   - Generates parallel answers using Gemini
   - Ranks answers using Gemini self-evaluation
   - Supplements with web search if needed

### Key Technical Improvements:

#### Model Benefits:
- **Consistent Performance**: All agents use the same high-quality models
- **Easy Upgrades**: Change models system-wide from one file
- **Cost Optimization**: Use Google's efficient Gemini models
- **Quality Embeddings**: Nomic AI model optimized for retrieval

#### RAG Enhancements:
- **768-dimensional embeddings** for better semantic understanding
- **Normalized embeddings** for improved similarity calculations
- **Multi-candidate diversity** with embedding perturbations
- **Gemini-based ranking** for intelligent answer selection

## 📁 Project Structure

```
mcp_a2a_agentic_rag/
├── 📄 main.py                    # Main application entry point
├── 📄 model.py                   # Centralized model configuration
├── 📄 requirements.txt           # Python dependencies
├── 📄 test_mcp_integration.py    # MCP integration tests
├── 📄 demo_integration.py        # System demonstration script
├── 📄 kill_services.sh          # Service cleanup script
├── 📄 MCP_INTEGRATION_SUMMARY.md # Integration documentation
│
├── 📁 agents/                    # A2A Agent implementations
│   ├── 📁 host_agent/           # Central orchestrator agent
│   │   ├── 📄 __init__.py
│   │   ├── 📄 agent.py          # Host agent implementation
│   │   └── 📄 instructions.txt   # Agent instructions & prompts
│   └── 📁 agentic_rag_agent/    # Specialized RAG agent
│       ├── 📄 __init__.py
│       ├── 📄 agent.py          # RAG agent implementation
│       └── 📄 instructions.txt   # RAG agent instructions
│
├── 📁 mcp/                      # Model Context Protocol servers
│   └── 📁 servers/
│       ├── 📁 web_search_server/    # SerpAPI web search
│       │   └── 📄 web_search_server.py
│       ├── 📁 terminal_server/      # Safe command execution
│       │   └── 📄 terminal_server.py
│       └── 📁 streamable_http_server/ # HTTP client tools
│           └── 📄 streamable_http_server.py
│
├── 📁 utilities/                # Shared utilities
│   ├── 📁 a2a/                 # Agent-to-Agent communication
│   │   ├── 📄 __init__.py
│   │   ├── 📄 client.py        # A2A client implementation
│   │   └── 📄 agent_registry.json # Agent discovery registry
│   └── 📁 mcp/                 # MCP client utilities
│       ├── 📄 __init__.py
│       ├── 📄 mcp_client.py    # Unified MCP client
│       └── 📄 mcp_discovery.py # MCP server discovery
│
├── 📁 app/                     # User interface components
│   └── 📄 streamlit_ui.py      # Streamlit web interface
│
└── 📁 tests/                   # Test suite
    ├── 📄 __init__.py
    └── 📄 test_chains.py       # Chain and workflow tests
```

### Key Components

#### Core Files
- **`main.py`**: Application entry point, starts all services
- **`model.py`**: Centralized model configuration (Gemini, Nomic AI)
- **`requirements.txt`**: All Python dependencies with versions

#### Testing & Demo
- **`test_mcp_integration.py`**: Comprehensive MCP testing suite
- **`demo_integration.py`**: Interactive system demonstration
- **`test_structure.py`**: Project structure validation

#### Agent System (`agents/`)
- **Host Agent**: Central orchestrator, coordinates all operations
- **RAG Agent**: Document processing and question answering specialist
- Each agent has its own instructions and implementation

#### MCP Servers (`mcp/servers/`)
- **Web Search**: Real-time web search via SerpAPI
- **Terminal**: Safe command execution and file operations  
- **HTTP**: Web scraping, API calls, HTTP requests

#### Utilities (`utilities/`)
- **A2A**: Agent-to-agent communication protocol
- **MCP**: Model Context Protocol client and discovery

## Configuration

### Model Configuration (`model.py`)
```python
# Default models (can be customized)
MODEL_INFO = {
    "llm_model": "gemini-2.0-flash-exp",
    "embed_model": "nomic-ai/nomic-embed-text-v1.5",
    "llm_provider": "Google Generative AI",
    "embed_provider": "HuggingFace"
}
```

### Agent Registry
Edit `utilities/a2a/agent_registry.json` to add/remove A2A agents:
```json
[
    "http://localhost:10002"
]
```

### Environment Variables
```bash
GOOGLE_API_KEY=your_google_api_key  # Required for Gemini
SERPAPI_KEY=your_serpapi_key        # Required for web search
```

## New Features

### Model Management:
- **Centralized Configuration**: All models defined in one place
- **Factory Functions**: Easy model instantiation with custom parameters
- **Model Information**: Built-in model info and debugging functions
- **Environment Integration**: Seamless API key management

### Enhanced RAG:
- **Multi-candidate Retrieval**: 3 diverse context sets per query
- **Parallel Processing**: Simultaneous answer generation
- **LLM Ranking**: Gemini evaluates and selects best answers
- **Web Integration**: Intelligent decision making for supplemental search
- **Better Embeddings**: 768-dimensional Nomic AI embeddings

### UI Improvements:
- **Model Information Display**: Shows current model configuration
- **Enhanced Status Checks**: Detailed system and model status
- **Better Error Handling**: More informative error messages
- **Model Configuration Button**: Check current model settings

## Troubleshooting

### Common Issues and Solutions

#### 1. MCP Server Connection Issues
**Problem**: "Failed to connect to MCP server" or asyncio cleanup errors
**Solutions**:
```bash
# Check if servers are running
ps aux | grep python | grep mcp

# Kill existing servers
python kill_services.sh

# Test with minimal script
python quick_test.py

# Restart with verbose logging
python test_mcp_integration.py --test --verbose
```

**For Python 3.13 asyncio errors**:
- The system now includes proper cleanup handling
- If you still see errors, try: `python quick_test.py` for a minimal test
- Use `python clean_demo.py` for demo without cleanup warnings
- These errors are harmless and don't affect functionality

#### 2. API Key Issues
**Problem**: "Authentication failed" or "Invalid API key"
**Solutions**:
```bash
# Verify environment variables
echo $GOOGLE_API_KEY
echo $SERPAPI_KEY

# Check .env file exists and is properly formatted
cat .env
```

#### 3. Model Loading Issues
**Problem**: "Model not found" or embedding download failures
**Solutions**:
```bash
# Check model configuration
python -c "from model import get_model_info; print(get_model_info())"

# Test model access
python -c "from model import get_llm_model; model = get_llm_model(); print('Model loaded successfully')"

# Clear HuggingFace cache if needed
rm -rf ~/.cache/huggingface/
```

#### 4. Port Conflicts
**Problem**: "Port already in use"
**Solutions**:
```bash
# Check what's using the ports
lsof -i :10001
lsof -i :10002

# Kill processes on specific ports
kill -9 $(lsof -t -i:10001)
kill -9 $(lsof -t -i:10002)
```

#### 5. Memory Issues with Embeddings
**Problem**: "Out of memory" when loading embedding models
**Solutions**:
- Ensure at least 4GB RAM available
- Close other applications
- Consider using smaller embedding models in `model.py`

#### 6. Dependency Issues
**Problem**: Missing dependencies or version conflicts
**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check for conflicts
pip check

# Create fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Debugging Tips

#### Enable Verbose Logging
```bash
# For MCP servers
PYTHONPATH=. python mcp/servers/web_search_server/web_search_server.py --verbose

# For agents
DEBUG=1 python main.py
```

#### Test Individual Components
```bash
# Test just the embedding model
python -c "from model import get_embedding_model; model = get_embedding_model(); print('Embeddings working')"

# Test just the LLM
python -c "from model import get_llm_model; model = get_llm_model(); print('LLM working')"

# Test MCP client without servers
python -c "from utilities.mcp.mcp_client import MCPClient; print('MCP client imports successfully')"
```

#### Check System Resources
```bash
# Check Python version (requires 3.8+)
python --version

# Check available memory
free -h  # Linux
top     # macOS

# Check disk space
df -h
```

### Getting Help

1. **Check logs**: All agents and servers log to console
2. **Run diagnostics**: Use `test_mcp_integration.py --setup` for full system check
3. **Test step by step**: Start with individual components before full system
4. **Environment issues**: Ensure all API keys are set correctly
5. **Model issues**: Verify internet connection for initial model downloads

## Development

### Adding New Models:
1. Update `model.py` with new model configurations
2. Import and use the new models in your agents
3. Update model information in `MODEL_INFO` dictionary

### Model Switching:
```python
# In your agent code
from model import get_llm_model, get_embedding_model

# Use different models
custom_llm = get_llm_model(temperature=0.5, model_name="gemini-1.5-pro")
custom_embed = get_embedding_model(model_name="BAAI/bge-large-en-v1.5")
```

### Adding New MCP Servers

1. **Create Server Directory**:
   ```bash
   mkdir mcp/servers/your_server_name
   ```

2. **Implement Server** (`your_server.py`):
   ```python
   import asyncio
   from mcp.server.stdio import stdio_server
   from mcp.server import Server
   from mcp.types import Tool, TextContent
   
   server = Server("your-server")
   
   @server.list_tools()
   async def list_tools() -> list[Tool]:
       return [
           Tool(
               name="your_tool",
               description="Description of your tool",
               inputSchema={
                   "type": "object",
                   "properties": {
                       "param": {"type": "string", "description": "Parameter description"}
                   },
                   "required": ["param"]
               }
           )
       ]
   
   @server.call_tool()
   async def call_tool(name: str, arguments: dict) -> list[TextContent]:
       if name == "your_tool":
           # Implement your tool logic
           result = f"Tool called with: {arguments}"
           return [TextContent(type="text", text=result)]
   
   if __name__ == "__main__":
       asyncio.run(stdio_server(server))
   ```

3. **Update MCP Config** (`mcp_config.json`):
   ```json
   {
     "your_server": {
       "command": "python",
       "args": ["mcp/servers/your_server_name/your_server.py"],
       "env": {}
     }
   }
   ```

4. **Test New Server**:
   ```bash
   python test_mcp_integration.py --test
   ```

### Adding New Agents

1. **Create Agent Directory**:
   ```bash
   mkdir agents/your_agent_name
   ```

2. **Implement Agent** (`agent.py`):
   ```python
   from fastapi import FastAPI
   from utilities.a2a.client import AgentToAgentClient
   from model import get_llm_model
   
   app = FastAPI()
   llm = get_llm_model()
   
   @app.post("/")
   async def handle_request(request: dict):
       # Process the request using your agent logic
       response = await your_agent_logic(request)
       return {"response": response}
   
   @app.get("/.well-known/agent.json")
   async def agent_card():
       return {
           "name": "Your Agent",
           "description": "Agent description",
           "capabilities": ["your", "capabilities"],
           "endpoints": [{"method": "POST", "path": "/"}]
       }
   ```

3. **Add to Registry** (`utilities/a2a/agent_registry.json`):
   ```json
   [
       "http://localhost:10002",
       "http://localhost:10003"
   ]
   ```

### Customizing Agent Instructions

Edit the `instructions.txt` files in each agent directory to modify agent behavior:

```txt
You are a specialized AI agent that...

Your capabilities include:
- Capability 1
- Capability 2

When receiving requests:
1. Analyze the request type
2. Determine required tools
3. Execute the appropriate workflow
4. Return structured results

Use the available MCP tools:
- web_search: For real-time information
- terminal: For file operations
- http: For web requests and APIs
```

### Environment Variables for Development

```bash
# Required for production
export GOOGLE_API_KEY="your_google_api_key"
export SERPAPI_KEY="your_serpapi_key"

# Optional for development
export DEBUG=1                    # Enable debug logging
export MCP_TIMEOUT=30            # MCP operation timeout
export AGENT_DISCOVERY_TIMEOUT=5 # A2A discovery timeout
export LOG_LEVEL=DEBUG           # Set logging level
```

## 🚀 Deployment & Production

### Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 10001 10002 8501
   
   CMD ["python", "main.py"]
   ```

2. **Docker Compose** (`docker-compose.yml`):
   ```yaml
   version: '3.8'
   services:
     mcp-agents:
       build: .
       ports:
         - "10001:10001"
         - "10002:10002"
         - "8501:8501"
       environment:
         - GOOGLE_API_KEY=${GOOGLE_API_KEY}
         - SERPAPI_KEY=${SERPAPI_KEY}
       volumes:
         - ./data:/app/data
   ```

3. **Deploy**:
   ```bash
   docker-compose up -d
   ```

### Production Considerations

#### Security
- **API Key Management**: Use secrets management (AWS Secrets Manager, Azure Key Vault)
- **Network Security**: Run behind a reverse proxy (nginx)
- **Input Validation**: Validate all user inputs and file uploads
- **Command Execution**: Review terminal server allowed commands

#### Performance
- **Resource Limits**: Set appropriate CPU/memory limits
- **Caching**: Implement caching for embedding operations
- **Load Balancing**: Use multiple agent instances behind a load balancer
- **Database**: Consider persistent storage for RAG indices

#### Monitoring
- **Logging**: Implement structured logging with log aggregation
- **Metrics**: Monitor response times, error rates, resource usage
- **Health Checks**: Implement comprehensive health check endpoints
- **Alerting**: Set up alerts for system failures

#### Scaling
- **Horizontal Scaling**: Deploy multiple agent instances
- **Database**: Use external vector database (Pinecone, Weaviate)
- **Message Queue**: Use Redis/RabbitMQ for agent communication
- **Microservices**: Split components into separate services

### Environment-Specific Configurations

#### Development
```bash
# .env.development
DEBUG=1
LOG_LEVEL=DEBUG
MCP_TIMEOUT=10
AGENT_DISCOVERY_TIMEOUT=5
```

#### Staging
```bash
# .env.staging
DEBUG=0
LOG_LEVEL=INFO
MCP_TIMEOUT=15
AGENT_DISCOVERY_TIMEOUT=3
```

#### Production
```bash
# .env.production
DEBUG=0
LOG_LEVEL=WARNING
MCP_TIMEOUT=30
AGENT_DISCOVERY_TIMEOUT=2
RATE_LIMIT_ENABLED=1
MAX_REQUESTS_PER_MINUTE=100
```

## Performance Notes

- **Embedding Model**: Downloads ~500MB on first use (cached locally)
- **FAISS Index**: In-memory storage, consider persistence for production
- **Gemini API**: Rate limits apply based on your Google Cloud quota
- **Multi-threading**: Parallel answer generation improves response time

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/mcp_a2a_agentic_rag.git
cd mcp_a2a_agentic_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Running Tests
```bash
# Run all tests
python test_mcp_integration.py --test
python -m pytest tests/ -v

# Run specific test categories
python test_mcp_integration.py --test --category mcp
python test_mcp_integration.py --test --category agents
```

### Code Style
```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### Submitting Changes
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and add tests
3. Ensure all tests pass
4. Submit a pull request with a clear description

### Areas for Contribution
- **New MCP Servers**: Implement additional tool servers
- **Agent Enhancements**: Improve existing agents or add new ones
- **UI Improvements**: Enhance the Streamlit interface
- **Documentation**: Improve docs and add examples
- **Testing**: Add more comprehensive tests
- **Performance**: Optimize system performance

## 📄 License

MIT License

Copyright (c) 2024 MCPxA2AxAgentic-RAG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📞 Support

- **Documentation**: Check this README and `MCP_INTEGRATION_SUMMARY.md`
- **Issues**: Report bugs and request features on GitHub
- **Testing**: Use `test_mcp_integration.py` for diagnostics
- **Community**: Join discussions in the Issues section

**Quick Commands Reference:**
```bash
# Test everything
python test_mcp_integration.py --test

# Run demo (clean output)
python clean_demo.py

# Run demo (verbose)
python demo_integration.py

# Quick connection test
python quick_test.py

# Start system
python main.py

# Kill services
./kill_services.sh
```