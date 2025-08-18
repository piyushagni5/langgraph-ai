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
├── graph/
│   ├── chains/
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_chains.py
│   │   ├── __init__.py
│   │   ├── answer_grader.py
│   │   ├── generation.py
│   │   ├── hallucination_grader.py
│   │   ├── retrieval_grader.py
│   │   └── router.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── grade_documents.py
│   │   ├── retrieve.py
│   │   └── web_search.py
│   ├── __init__.py
│   ├── consts.py
│   ├── graph.py
│   └── state.py
├── static/
│   ├── LangChain-logo.png
│   ├── Langgraph Adaptive Rag.png
│   └── graph.png
├── .env
├── .gitignore
├── ingestion.py
├── main.py
├── model.py
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

### Start the Agentic RAG flow

```bash
uv run main.py
```

### Running Tests

To run tests, execute the following command:

```bash
uv run pytest . -s -v
```

## Features

- **Adaptive RAG**: Dynamically routes queries to the most appropriate processing method
- **Self-RAG**: Implements self-reflection mechanisms for improved answer quality
- **Reflective RAG**: Incorporates reflection and grading for enhanced retrieval
- **Web Search Integration**: Fallback to web search when local knowledge is insufficient
- **Document Grading**: Evaluates relevance of retrieved documents
- **Hallucination Detection**: Identifies and handles potential hallucinations in generated responses

## Architecture

The system implements a sophisticated RAG pipeline with the following components:

- **Router**: Intelligently routes queries between vectorstore retrieval and web search
- **Retrieval Grader**: Evaluates the relevance of retrieved documents
- **Generation Chain**: Produces answers based on retrieved context
- **Hallucination Grader**: Detects potential hallucinations in generated responses
- **Answer Grader**: Evaluates the quality and relevance of final answers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
- By Sophia Young from Mistral & Lance Martin from LangChain
- Built with LangGraph 🦜🕸️
- Marco's refactored repository

