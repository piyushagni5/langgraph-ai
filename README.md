# LangGraph AI Repository

A comprehensive collection of LangGraph implementations, tutorials, and advanced AI workflows covering Agentic RAG systems, MCP (Model Context Protocol) development, and practical AI application patterns.

## Overview

This repository serves as a implementation guide for building sophisticated AI applications using LangGraph. It contains practical examples, tutorials, and production-ready implementations across multiple domains:

- **Agentic RAG Systems**: Advanced retrieval-augmented generation with adaptive routing and self-correction mechanisms
- **MCP Development**: Complete Model Context Protocol server and client implementations
- **Workflow Patterns**: Orchestration patterns for complex AI workflows
- **Human-in-the-Loop Systems**: Interactive AI systems with human oversight
- **Advanced RAG Agents**: Sophisticated retrieval and generation systems

## Repository Structure

```
langgraph-ai/
├── rag/
│   ├── rag-from-scratch/
│   │   └── 1_rag_overview.ipynb
│   ├── rag-agents/
│   │   ├── Building an Advanced RAG Agent.ipynb
│   │   └── rag-as-tool-in-langgraph-agents.ipynb
│   ├── agentic-rag/
│   │   ├── agentic-rag-systems/
│   │   │   └── building-adaptive-rag/
│   │   └── agentic-workflow-pattern/
│   │       ├── 1-prompting_chaining.ipynb
│   │       ├── 2-routing.ipynb
│   │       ├── 3-parallelization.ipynb
│   │       ├── 4-orchestrator-worker.ipynb
│   │       └── 5-Evaluator-optimizer.ipynb
├── mcp/
│   ├── 01-build-your-own-server-client/
│   ├── 02-build-mcp-client-with-multiple-server-support/
│   ├── 03-build-mcp-server-client-using-sse/
│   └── 04-build-streammable-http-mcp-client/
├── langgraph-cookbook/
│   ├── human-in-the-loop/
│   │   ├── 01-human-in-the-loop.ipynb
│   │   ├── 02-human-in-the-loop.ipynb
│   │   └── 03-human-in-the-loop.ipynb
│   └── tool-calling -vs-react.ipynb
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```


## Prerequisites

Before setting up this repository, ensure you have the following installed:

- Python 3.10 or higher (depends on the project)
- UV package manager (recommended) or pip
- Git

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
cd langgraph-ai
```

### Step 2: Install UV Package Manager

If you haven't installed UV yet, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Create Virtual Environment

Navigate to the specific project directory you want to work with. For example, to work with the Adaptive RAG system:

```bash
cd langgraph-cookbook/agentic-patterns
```

Create a virtual environment using UV:

```bash
uv venv --python 3.10
```

### Step 4: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

**Using UV (Recommended):**
```bash
uv pip install -r requirements.txt
```

**Using pip (Alternative):**
```bash
pip install -r requirements.txt
```

### Step 6: Adding Virtual Environment to Jupyter Kernel
To use your UV virtual environment with Jupyter notebooks, you need to install ipykernel and register the environment as a kernel:
**Install ipykernel in the virtual environment**:
   ```bash
   uv pip install ipykernel
   ```

**Register the virtual environment as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=langgraph-ai --display-name="LangGraph AI"
   ```
When you open a notebook, you can select the "LangGraph AI" kernel from the kernel menu.

### Step 7: Environment Configuration

Create a `.env` file in your project directory with the necessary API keys:

```env
ANTHROPIC_API_KEY="your-anthropic-api-key"
# LANGCHAIN_API_KEY="your-langchain-api-key"  # optional
# LANGCHAIN_TRACING_V2=True                   # optional
# LANGCHAIN_PROJECT="multi-agent-swarm"       # optional
```

**Note**: The `LANGCHAIN_API_KEY` is required if you enable tracing with `LANGCHAIN_TRACING_V2=true`.


## Running Projects

### Adaptive RAG System

```bash
cd agentic-rag/agentic-rag-systems/building-adaptive-rag
uv run main.py
```

### Running Tests

```bash
uv run pytest . -s -v
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes and improvements
- New tutorial implementations
- Documentation enhancements
- Performance optimizations

## License

This project is open source and available under the MIT License.

---

**Note**: This repository contains multiple independent projects. Each project has its own requirements and setup instructions. Please refer to individual project README files for specific details.