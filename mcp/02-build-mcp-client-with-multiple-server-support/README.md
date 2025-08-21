# MCP Multi-Server Platform

A comprehensive Model Context Protocol (MCP) implementation with multiple server support, featuring terminal execution and web content fetching capabilities.

## Architecture

This platform integrates multiple components from the MCP ecosystem:

- **MCP Client**: LangChain-based client with multi-server orchestration capabilities
- **Terminal Server**: Custom secure command execution within containerized environment  
- **Fetch Server**: Web content retrieval and markdown conversion service

### Integration Benefits
- **Unified Interface**: Single client communicating with multiple specialized servers
- **Docker Orchestration**: All services containerized for consistent deployment
- **Configuration Management**: JSON-based server configuration with environment-specific settings

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for client)
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd langgraph-ai/mcp/02-build-mcp-client-with-multiple-server-support
```

### 2. Install uv (Optional)
We'll use uv for fast and efficient Python package management:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm whether uv has installed in your system:
```bash
uv --version
```
Expected output:
```
uv 0.7.19
```

### 3. Build and Start Servers
```bash
docker-compose up --build -d
```

### 4. Setup and Run Client
Navigate to client directory and run the application:
```bash
cd clients/mcp-client
uv run python mcp_client.py
```

This will automatically create a virtual environment, install dependencies, and run the client.

### 5. Stop Servers
```bash
docker-compose down
```

## Project Structure

```
02-build-mcp-client-with-multiple-server-support/
├── README.md                     # Project documentation
├── docker-compose.yml           # Multi-server orchestration
├── clients/
│   └── mcp-client/               # MCP client implementation
│       ├── mcp_client.py         # Main client application
│       ├── main.py               # Entry point
│       ├── config.json           # Server configuration
│       ├── pyproject.toml        # Python project configuration
│       ├── requirements.txt      # Python dependencies
│       ├── .env                  # Environment variables
│       └── README.md             # Client documentation
├── servers/                      # Server implementations
│   ├── terminal_server/          # Command execution server
│   │   ├── terminal_server.py    # Terminal server implementation
│   │   ├── Dockerfile            # Container configuration
│   │   ├── pyproject.toml        # Project configuration
│   │   └── requirements.txt      # Server dependencies
│   └── fetch_server/             # Web content fetching server
│       ├── src/
│       │   └── mcp_server_fetch/
│       │       ├── __init__.py   # Package initialization
│       │       ├── __main__.py   # Module entry point
│       │       └── server.py     # Fetch server implementation
│       ├── Dockerfile            # Container configuration
│       ├── pyproject.toml        # Project configuration
│       ├── requirements.txt      # Server dependencies
│       └── README.md             # Fetch server documentation
└── workspace/                    # Shared workspace
    └── [dynamic files]           # Runtime files created by operations
```

## Configuration

Server configurations are defined in `clients/mcp-client/config.json`:

```json
{
  "mcpServers": {
    "terminal_server": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--init", "-e", "DOCKER_CONTAINER=true", 
               "-v", "/path/to/workspace:/workspace", "terminal_server_docker"]
    },
    "fetch": {
      "command": "docker", 
      "args": ["run", "-i", "--rm", "mcp_fetch_server_test"]
    }
  }
}
```

## Security Features

- **Containerized Execution**: All server operations run in isolated Docker containers
- **Workspace Isolation**: Terminal operations restricted to mounted workspace directory
- **Robots.txt Compliance**: Fetch server respects website crawling policies by default

## Available Tools

### Terminal Server
- `execute_shell_command`: Execute shell commands in containerized environment

### Fetch Server  
- `fetch`: Retrieve web content and convert to markdown format
- Supports chunked reading with `start_index` parameter
- Configurable content length limits

## License
This project is licensed under the GNU General Public License v3.0.

## Attribution
This project was built with reference to and modifications of code from:
- theailanguage [repositories](https://github.com/theailanguage/mcp_client) by Kartik - GPL v3.0 licensed
- Model Context Protocol [servers](https://github.com/modelcontextprotocol/servers/tree/2ea2c67fd1fdd5cf0e27e2f3f825ba74ffdf1284/src/fetch) - MIT licensed