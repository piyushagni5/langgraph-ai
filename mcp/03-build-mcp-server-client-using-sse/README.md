# MCP Server-Sent Events (SSE) Platform

A comprehensive Model Context Protocol (MCP) implementation using Server-Sent Events (SSE) transport, featuring real-time bidirectional communication between AI agents and MCP servers over HTTP.

## Architecture

This platform demonstrates advanced MCP capabilities with SSE transport:

- **SSE MCP Client**: AI-powered client with Google Gemini integration for intelligent tool calling
- **SSE MCP Server**: FastAPI-based server with secure terminal command execution via SSE transport
- **Real-time Communication**: Persistent HTTP connections enabling streaming responses and real-time tool execution

### Key Benefits
- **Real-time Streaming**: SSE maintains persistent connections for instant response delivery
- **HTTP-based**: Works through firewalls and proxies, simpler than WebSockets
- **AI Integration**: Gemini AI automatically determines when and how to use tools
- **Bidirectional Flow**: Despite SSE being unidirectional, achieves bidirectional communication via HTTP POST + SSE

## Quick Start

### Prerequisites
- Python 3.11+ (for both client and server)
- Google Gemini API key
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd langgraph-ai/mcp/03-build-mcp-server-client-using-sse
```

### 2. Install uv (Recommended)
We use uv for fast and efficient Python package management:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm installation:
```bash
uv --version
```
Expected output: `uv 0.7.19` (or later)

### 3. Setup Environment Variables
Create a `.env` file in the client directory:
```bash
cd clients/mcp-client
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
EOF
```

### 4. Start the MCP Server
Open a terminal and start the SSE server:
```bash
cd servers/sse_server
uv run python terminal_server_sse.py
```

The server will start on `http://localhost:8081` with SSE endpoint at `/sse`.

### 5. Run the MCP Client
In a new terminal, start the client:
```bash
cd clients/mcp-client
uv run python client_sse.py http://localhost:8081/sse
```

### 6. Start Chatting!
The client will automatically:
- Connect to the MCP server via SSE
- Discover available tools
- Integrate tools with Gemini AI
- Enable intelligent tool calling through natural language

Example interaction:
```
You: List the files in the current directory
Assistant: I'll help you list the files in the current directory.

[Tool execution via SSE...]

Here are the files in the current directory:
- client_sse.py
- pyproject.toml
- requirements.txt
- .env
```

## Project Structure

```
03-build-mcp-server-client-using-sse/
├── README.md                     # Project documentation
├── clients/
│   └── mcp-client/               # MCP client with Gemini AI integration
│       ├── client_sse.py         # Main SSE client implementation
│       ├── pyproject.toml        # Python project configuration
│       ├── requirements.txt      # Client dependencies
│       ├── .env                  # Environment variables (API keys)
│       └── uv.lock               # Dependency lock file
├── servers/                      # Server implementations
│   └── sse_server/               # SSE-based MCP server
│       ├── terminal_server_sse.py # SSE server implementation
│       ├── Dockerfile            # Container configuration
│       └── requirements.txt      # Server dependencies
└── workspace/                    # Shared workspace
    └── test.txt                  # Example workspace file
```

## SSE Transport Implementation

### Server Side (terminal_server_sse.py)
- **FastMCP Framework**: Simplified tool definition with decorators
- **SSE Transport**: `SseServerTransport` handles SSE protocol
- **HTTP Endpoints**:
  - `/sse` - SSE connection endpoint for real-time streaming
  - `/messages/` - HTTP POST endpoint for tool requests
- **Tool Execution**: Secure command execution with workspace isolation

### Client Side (client_sse.py)
- **SSE Connection**: Persistent HTTP connection for real-time responses
- **MCP Protocol**: Full MCP client implementation over SSE
- **AI Integration**: Google Gemini for intelligent tool selection
- **Async Architecture**: Non-blocking operations for optimal performance

## Available Tools

### Terminal Server Tools
- `execute_shell_command`: Execute shell commands in a secure environment
  - Parameters: `command` (string) - The shell command to execute
  - Returns: Command output and execution status
  - Security: Restricted to workspace directory, containerizable

## Configuration

### Client Configuration
The client requires a Gemini API key in the `.env` file:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### Server Configuration
The server runs on `localhost:8081` by default. Modify `terminal_server_sse.py` to change:
- Port number
- Host binding
- Workspace directory
- Security settings

## Docker Deployment

### Build and Run Server in Docker
```bash
cd servers/sse_server
docker build -t mcp-sse-server .
docker run -p 8081:8081 -v $(pwd)/../../workspace:/workspace mcp-sse-server
```

### Client with Docker Server
```bash
cd clients/mcp-client
uv run python client_sse.py http://localhost:8081/sse
```

## Security Features

- **Workspace Isolation**: All file operations restricted to mounted workspace directory
- **Containerized Execution**: Server can run in isolated Docker containers
- **Command Validation**: Input sanitization and command restrictions
- **Environment Separation**: Client and server run in separate processes/containers

## Troubleshooting

### Common Issues

**Connection Refused**:
- Ensure server is running on correct port
- Check firewall settings
- Verify URL format: `http://localhost:8081/sse`

**API Key Errors**:
- Verify Gemini API key in `.env` file
- Check API key permissions and quotas
- Ensure `.env` file is in client directory

**Tool Execution Failures**:
- Check workspace permissions
- Verify command syntax
- Review server logs for detailed errors

## License
This project is licensed under the GNU General Public License v3.0.

## Attribution
This project was built with reference to and modifications of code from:
- theailanguage [repositories](https://github.com/theailanguage/mcp_client) by Kartik - GPL v3.0 licensed
- Model Context Protocol [servers](https://github.com/modelcontextprotocol/servers/tree/2ea2c67fd1fdd5cf0e27e2f3f825ba74ffdf1284/src/fetch) - MIT licensed