#!/usr/bin/env python
"""
Multi-Server MCP Client Implementation

A comprehensive LangChain-based Model Context Protocol client that enables
simultaneous orchestration of multiple MCP servers.

This client provides the following capabilities:
  - Dynamic configuration loading from JSON files via environment variables
  - Simultaneous connections to multiple MCP servers as defined in configuration
  - Automatic tool discovery and aggregation from all connected servers
  - Integration with Google Gemini API through LangChain for intelligent agent creation
  - Interactive chat interface with full tool access across all servers

Configuration Details:
  - Retry Logic: Implements max_retries=2 for handling transient API failures
  - Temperature Control: Set to 0 for deterministic responses
  - Environment Configuration: Uses CONFIG variable to specify server definitions
  - Graceful Error Handling: Continues operation even if individual servers fail to connect

Usage:
    Set GOOGLE_API_KEY environment variable and run:
    python mcp_client.py
"""

import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack

# MCP Client Core Components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Agent Framework and LLM Integration
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment Configuration
from dotenv import load_dotenv
load_dotenv()


class CustomEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder for handling LangChain-specific objects.
    
    This encoder properly serializes LangChain message objects (HumanMessage, ToolMessage, etc.)
    by extracting their content attribute and preserving type information for debugging.
    """
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)

# ---------------------------
# Configuration Management
# ---------------------------
def read_config_json():
    """
    Load MCP server configuration with intelligent fallback handling.

    Configuration Priority:
      1. MULTI_SERVER_CONFIG environment variable path
      2. Default 'config.json' in script directory

    Returns:
        dict: Parsed configuration containing MCP server definitions
        
    Raises:
        SystemExit: If configuration file cannot be read or parsed
    """
    config_path = os.getenv("MULTI_SERVER_CONFIG")

    if not config_path:  
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        print(f"Warning: MULTI_SERVER_CONFIG not set. Using default: {config_path}")

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Configuration load failed at '{config_path}': {e}")
        sys.exit(1)


# LLM Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",                     # Latest Gemini model for optimal performance
    temperature=0,                                # Deterministic output for consistent behavior
    max_retries=2,                                # Resilience against transient API failures
    google_api_key=os.getenv("GOOGLE_API_KEY")    # Secure API key retrieval from environment
)

async def run_agent():
    """
    Orchestrate multi-server MCP connections and create a unified agent interface.
    
    This function handles the complete workflow:
    1. Configuration loading and validation
    2. Parallel server connections with error isolation
    3. Tool aggregation from all successful connections
    4. Agent initialization with comprehensive tool access
    5. Interactive chat loop with formatted response handling
    """
    config = read_config_json()
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("Error: No MCP servers found in configuration.")
        return

    tools = []

    async with AsyncExitStack() as stack:
        # Parallel server connection and tool loading
        for server_name, server_info in mcp_servers.items():
            print(f"\nEstablishing connection to: {server_name}...")

            server_params = StdioServerParameters(
                command=server_info["command"],
                args=server_info["args"]
            )

            try:
                # Establish stdio communication channel
                read, write = await stack.enter_async_context(stdio_client(server_params))
                # Create managed client session
                session = await stack.enter_async_context(ClientSession(read, write))
                # Initialize MCP protocol handshake
                await session.initialize()

                # Load and integrate server tools
                server_tools = await load_mcp_tools(session)

                for tool in server_tools:
                    print(f"Registered tool: {tool.name}")
                    tools.append(tool)

                print(f"Successfully loaded {len(server_tools)} tools from {server_name}")
            except Exception as e:
                print(f"Connection failed for {server_name}: {e}")
                # Continue with other servers despite individual failures

        if not tools:
            print("Error: No tools available from any server. Terminating.")
            return

        # Create unified agent with all available tools
        agent = create_react_agent(llm, tools)

        # Interactive chat interface
        print("\nMulti-Server MCP Client Ready! Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break

            # Process query through agent
            response = await agent.ainvoke({"messages": query})

            # Format and display response
            print("\nResponse:")
            try:
                formatted = json.dumps(response, indent=2, cls=CustomEncoder)
                print(formatted)
            except Exception:
                print(str(response))


# Application Entry Point
if __name__ == "__main__":
    asyncio.run(run_agent())