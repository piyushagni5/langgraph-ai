# Adaptive RAG
Adaptive RAG is an advanced strategy for RAG that intelligently combines (1) dynamic query analysis with (2) active/self-corrective mechanisms.

Adaptive RAG represents the most sophisticated evolution, addressing a fundamental insight: not all queries are created equal. The research reveals that real-world queries exhibit vastly different complexity levels:

- Simple queries: "Paris is the capital of what?" - Can be answered directly by LLMs
- Multi-hop queries: "When did the people who captured Malakoff come to the region where Philipsburg is located?" - Requires four reasoning steps


![alt text](image.png)

This comprehensive guide presents a refactored approach to the original [LangChain implementations](https://github.com/mistralai/cookbook/tree/main/third_party/langchain), prioritizing enhanced code readability, improved maintainability, and superior developer experience. The implementation is inspired by Marco’s [GitHub repository](https://github.com/emarco177/langgraph-course/tree/project/agentic-rag), which itself references work from mistralai’s [GitHub repository](https://github.com/mistralai/cookbook/tree/main/third_party/langchain). 


## Project Structure

```
building-adaptive-rag/
├── src/                    # Source code
│   ├── workflow/          # Core workflow logic
│   │   ├── chains/       # LLM processing chains
│   │   │   ├── answer_grader.py
│   │   │   ├── generation.py
│   │   │   ├── hallucination_grader.py
│   │   │   ├── retrieval_grader.py
│   │   │   └── router.py
│   │   ├── nodes/        # Workflow nodes
│   │   │   ├── generate.py
│   │   │   ├── grade_documents.py
│   │   │   ├── retrieve.py
│   │   │   └── web_search.py
│   │   ├── consts.py     # Node constants
│   │   ├── graph.py      # Main workflow orchestration
│   │   └── state.py      # State management
│   ├── cli/              # Command line interface
│   │   └── main.py       # Interactive CLI
│   └── models/           # Model configurations
│       └── model.py      # LLM and embedding models
├── data/                 # Data processing
│   └── ingestion.py      # Document ingestion and vector store
├── assets/               # Static files and images
│   ├── LangChain-logo.png
│   └── Langgraph Adaptive Rag.png
├── tests/                # Test files
│   ├── __init__.py
│   └── test_chains.py    # Chain testing suite
├── .env                  # Environment variables
├── .gitignore
├── main.py              # Application entry point
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
```

2. **Navigate to the project directory**

```bash
cd agentic-rag/agentic-rag-systems/building-adaptive-rag/
```

3. **Create and activate virtual environment**

```bash
uv venv --python 3.10
source .venv/bin/activate
```

4. **Install dependencies**

```bash
uv pip install -r requirements.txt
```

## Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file:

```env
GOOGLE_API_KEY=your_tavily_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # For web search capabilities
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true                      # Optional
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com # Optional
LANGCHAIN_PROJECT=agentic-rag                  # Optional
```

**Important Note**: If you enable tracing by setting `LANGCHAIN_TRACING_V2=true`, you must have a valid LangSmith API key set in `LANGCHAIN_API_KEY`. Without a valid API key, the application will throw an error.

## Usage

### Start the Adaptive RAG System

```bash
python main.py
```

Or with uv:

```bash
uv run main.py
```

The system will start an interactive CLI where you can ask questions and receive intelligent responses that combine local knowledge base retrieval with web search when necessary.

### Running Tests

To run the test suite (make sure your virtual environment is activated):

```bash
# If using virtual environment
source .venv/bin/activate
python -m pytest tests/ -v
```

Or with uv:

```bash
uv run pytest tests/ -v
```

## Features

- **Adaptive RAG**: Dynamically routes queries to the most appropriate processing method
- **Self-RAG**: Implements self-reflection mechanisms for improved answer quality
- **Reflective RAG**: Incorporates reflection and grading for enhanced retrieval
- **Web Search Integration**: Fallback to web search when local knowledge is insufficient
- **Document Grading**: Evaluates relevance of retrieved documents
- **Hallucination Detection**: Identifies and handles potential hallucinations in generated responses
- **Professional Architecture**: Clean, modular codebase with industry-standard folder structure
- **Interactive CLI**: User-friendly command-line interface for easy interaction
- **Comprehensive Testing**: Full test suite for reliability and maintainability

## Architecture

The system implements a sophisticated RAG pipeline with a professional, modular architecture:

### Core Components

- **Router** (`src/workflow/chains/router.py`): Intelligently routes queries between vectorstore retrieval and web search
- **Retrieval Grader** (`src/workflow/chains/retrieval_grader.py`): Evaluates the relevance of retrieved documents
- **Generation Chain** (`src/workflow/chains/generation.py`): Produces answers based on retrieved context
- **Hallucination Grader** (`src/workflow/chains/hallucination_grader.py`): Detects potential hallucinations in generated responses
- **Answer Grader** (`src/workflow/chains/answer_grader.py`): Evaluates the quality and relevance of final answers

### Workflow Nodes

- **Retrieve** (`src/workflow/nodes/retrieve.py`): Retrieves documents from the vector store
- **Grade Documents** (`src/workflow/nodes/grade_documents.py`): Filters documents by relevance
- **Generate** (`src/workflow/nodes/generate.py`): Generates natural language answers
- **Web Search** (`src/workflow/nodes/web_search.py`): Performs web search for additional information

### Data Management

- **Ingestion** (`data/ingestion.py`): Handles document loading, processing, and vector store creation
- **Models** (`src/models/model.py`): Centralized LLM and embedding model configuration

## Workflow

The adaptive RAG system follows this intelligent workflow:

1. **Query Analysis**: User question is analyzed by the router
2. **Initial Routing**: Directs to either vector store retrieval or web search
3. **Document Retrieval**: Retrieves relevant documents from the knowledge base
4. **Relevance Grading**: Evaluates and filters documents for relevance
5. **Answer Generation**: Creates responses using filtered context
6. **Quality Assessment**: Checks for hallucinations and answer adequacy
7. **Adaptive Response**: Routes to web search if quality is insufficient
8. **Final Output**: Delivers high-quality, grounded responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
- By Sophia Young from Mistral & Lance Martin from LangChain
- Built with LangGraph
- Marco's refactored [repository](https://github.com/emarco177/langgraph-course/tree/project/agentic-rag)
