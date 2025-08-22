#!/usr/bin/env python
"""
client_sse.py

This file demonstrates how to build an MCP (Model Context Protocol) client that connects to an MCP server 
using SSE (Server-Sent Events) transport, integrated with Google's Gemini AI for intelligent tool calling.

What makes this architecture powerful:
1. **SSE Connection**: Maintains a persistent HTTP connection to the MCP server for real-time communication
2. **AI Integration**: Uses Gemini's function calling capabilities to intelligently decide when to use tools
3. **Bidirectional Flow**: Despite SSE being "server-to-client", we achieve bidirectional communication:
   - SSE connection: Server streams responses back to client
   - HTTP POST: Client sends tool requests to server's /messages/ endpoint
4. **Async Architecture**: Everything is asynchronous for optimal performance

Complete Request Flow:
┌─────────────────┐    (1) User Query     ┌─────────────────┐
│                 │─────────────────────→ │                 │
│  User Terminal  │                       │  MCP Client     │
│                 │                       │                 │
└─────────────────┘                       └─────────────────┘
                                                   │
                                        (2) Send to Gemini API
                                                   ↓
┌─────────────────┐                       ┌─────────────────┐
│                 │                       │                 │
│  MCP Server     │←─────(4) Tool Call────│  Gemini API     │
│  (via SSE)      │      (if needed)      │                 │
│                 │─────(5) Response─────→│                 │
└─────────────────┘                       └─────────────────┘
        │                                           │
        │                                           │
        └──────────(6) Final Answer─────────────────┘

Key Components:
- **ClientSession**: Manages MCP protocol communication over SSE streams
- **sse_client**: Creates and manages the persistent SSE connection
- **Gemini Integration**: Provides AI-driven tool selection and natural language processing
- **Async Context Managers**: Ensure proper resource cleanup and connection management
"""

import asyncio            # Core async/await functionality for non-blocking operations
import os                 # Environment variable access for API keys and configuration
import sys                # Command-line argument parsing
import json               # JSON serialization for data exchange
from typing import Optional  # Type hints for better code clarity and IDE support

# MCP Core Components
from mcp import ClientSession        # High-level MCP protocol handler
from mcp.client.sse import sse_client # SSE connection manager (async context manager)

# Google Gemini AI Integration
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

# Environment configuration
from dotenv import load_dotenv

# Load environment variables from .env file (API keys, server URLs, etc.)
load_dotenv()


class MCPClient:
    """
    A comprehensive MCP client that bridges AI language models with MCP tools via SSE.
    
    This class handles:
    - SSE connection lifecycle management
    - MCP protocol communication
    - Gemini AI integration for intelligent tool calling
    - Bidirectional message flow coordination
    """
    
    def __init__(self):
        # MCP Session Management
        # These will be populated when we connect to the MCP server
        self.session: Optional[ClientSession] = None  # MCP protocol handler
        self._streams_context = None                   # SSE connection context manager
        self._session_context = None                   # MCP session context manager

        # AI Integration Setup
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please add it to your .env file: GEMINI_API_KEY=your_key_here"
            )

        # Initialize Gemini client for AI-powered tool calling
        # Gemini will analyze user queries and determine when/how to use MCP tools
        self.genai_client = genai.Client(api_key=gemini_api_key)
        
        # Will store converted tool definitions for Gemini function calling
        self.function_declarations = []

    async def connect_to_sse_server(self, server_url: str):
        """
        Establish connection to MCP server using SSE transport.
        
        This method performs the complete SSE handshake:
        1. **SSE Connection**: Opens persistent HTTP connection for server→client streaming
        2. **MCP Session**: Initializes MCP protocol over the SSE streams
        3. **Tool Discovery**: Retrieves available tools from server
        4. **AI Integration**: Converts MCP tools to Gemini-compatible format
        
        The SSE connection pattern:
        - Client connects to server's /sse endpoint
        - Server maintains connection and can push data anytime
        - Client can send requests via separate HTTP POST to /messages/
        - This creates effectively bidirectional communication over HTTP
        
        Args:
            server_url (str): Complete URL to MCP server's SSE endpoint (e.g., http://localhost:8081/sse)
        """
        print(f"Connecting to MCP server via SSE: {server_url}")
        
        try:
            # STEP 1: Establish SSE Connection
            # sse_client returns an async context manager that handles:
            # - HTTP connection establishment
            # - SSE protocol negotiation  
            # - Stream creation for bidirectional communication
            self._streams_context = sse_client(url=server_url)
            streams = await self._streams_context.__aenter__()
            
            print("SSE connection established")
            
            # STEP 2: Initialize MCP Session
            # ClientSession wraps the SSE streams with MCP protocol handling
            # It manages message serialization, protocol negotiation, and error handling
            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()
            
            # STEP 3: MCP Protocol Handshake
            # This exchanges capabilities and initializes the protocol
            # Server will respond with its supported features and available tools
            await self.session.initialize()
            print("MCP protocol initialized")

            # STEP 4: Tool Discovery
            # Retrieve all available tools from the MCP server
            # This gives us the tool names, descriptions, and parameter schemas
            print("Discovering available tools...")
            response = await self.session.list_tools()
            tools = response.tools
            
            tool_names = [tool.name for tool in tools]
            print(f"Found {len(tools)} tools: {tool_names}")

            # STEP 5: AI Integration Setup
            # Convert MCP tool definitions to Gemini-compatible function declarations
            # This allows Gemini to understand what tools are available and how to call them
            self.function_declarations = convert_mcp_tools_to_gemini(tools)
            print("Tools integrated with Gemini AI")
            
        except Exception as e:
            print(f"Connection failed: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """
        Gracefully shutdown all connections and clean up resources.
        
        This is critical for SSE connections because:
        - SSE maintains persistent HTTP connections
        - Improper shutdown can leave connections hanging
        - Server resources need to be freed
        
        The cleanup order matters:
        1. Close MCP session (sends proper protocol termination)
        2. Close SSE streams (terminates HTTP connection)
        """
        print("Cleaning up connections...")
        
        # Close MCP session first (graceful protocol termination)
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
                print("MCP session closed")
            except Exception as e:
                print(f"Warning during MCP session cleanup: {e}")
        
        # Close SSE connection streams
        if self._streams_context:
            try:
                await self._streams_context.__aexit__(None, None, None)
                print("SSE connection closed")
            except Exception as e:
                print(f"Warning during SSE cleanup: {e}")

    async def process_query(self, query: str) -> str:
        """
        Process user query through AI with intelligent tool calling.
        
        This is where the magic happens:
        1. **AI Analysis**: Gemini analyzes the user query
        2. **Tool Decision**: Gemini decides if tools are needed
        3. **Tool Execution**: If needed, calls MCP tools via SSE
        4. **Response Synthesis**: Gemini creates final response using tool results
        
        The AI Function Calling Flow:
        User Query → Gemini → Function Call Request → MCP Server → Tool Result → Gemini → Final Answer
        
        This creates intelligent tool usage where the AI determines:
        - Which tools to use
        - What parameters to pass
        - How to interpret results
        - How to present final answers
        
        Args:
            query (str): Natural language query from user
        
        Returns:
            str: AI-generated response, potentially enhanced with tool results
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_sse_server() first.")
        
        print(f"\nProcessing query: '{query}'")
        
        # STEP 1: Format User Query for Gemini
        # Create structured content object that Gemini expects
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )

        # STEP 2: Send Query to Gemini with Tool Context
        # Include function declarations so Gemini knows what tools are available
        print("Sending query to Gemini AI...")
        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_prompt_content],
            config=types.GenerateContentConfig(
                tools=self.function_declarations,  # Available MCP tools
            ),
        )

        final_text = []

        # STEP 3: Process Gemini Response
        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    
                    # STEP 4: Handle Function Calls (Tool Usage)
                    if part.function_call:
                        # Gemini has decided to use a tool!
                        tool_name = part.function_call.name
                        tool_args = part.function_call.args
                        
                        print(f"Gemini requested tool: {tool_name}")
                        print(f"Tool arguments: {tool_args}")

                        # Execute the tool call on MCP server via SSE
                        try:
                            print("Calling MCP server...")
                            result = await self.session.call_tool(tool_name, tool_args)
                            function_response = {"result": result.content}
                            print(f"Tool result received: {result.content[:100]}...")  # Truncate for display
                            
                        except Exception as e:
                            print(f"Tool call failed: {e}")
                            function_response = {"error": str(e)}

                        # STEP 5: Send Tool Result Back to Gemini
                        # Format the tool result for Gemini to process
                        function_response_part = types.Part.from_function_response(
                            name=tool_name,
                            response=function_response
                        )

                        function_response_content = types.Content(
                            role='tool',
                            parts=[function_response_part]
                        )

                        # STEP 6: Get Final Response from Gemini
                        # Send the complete conversation: original query + function call + tool result
                        print("Getting final response from Gemini...")
                        response = self.genai_client.models.generate_content(
                            model='gemini-2.0-flash-001',
                            contents=[
                                user_prompt_content,       # Original user query
                                types.Content(role='model', parts=[part]),  # Gemini's function call
                                function_response_content, # Tool execution result
                            ],
                            config=types.GenerateContentConfig(
                                tools=self.function_declarations,
                            ),
                        )

                        # Extract the final synthesized response
                        if response.candidates and response.candidates[0].content.parts:
                            final_text.append(response.candidates[0].content.parts[0].text)
                    else:
                        # No tool call needed - direct text response from Gemini
                        final_text.append(part.text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """
        Interactive chat interface for continuous user interaction.
        
        Provides a terminal-based chat experience where users can:
        - Ask questions in natural language
        - Trigger tool calls automatically through AI
        - See real-time responses via SSE
        - Exit gracefully with 'quit'
        
        This demonstrates the complete end-to-end flow of MCP + AI integration.
        """
        print("\n" + "="*60)
        print("MCP + Gemini AI Client Started!")
        print("Ask questions and I'll use tools intelligently when needed")
        print("Available tools will be called automatically by AI")
        print("Type 'quit' to exit")
        print("="*60)

        while True:
            try:
                # Get user input
                query = input("\nYou: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                # Process query through AI + MCP tools
                print("Processing...")
                response = await self.process_query(query)
                
                # Display response
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")


def clean_schema(schema):
    """
    Sanitize JSON schema for Gemini function calling compatibility.
    
    Why this is needed:
    - MCP tools include JSON schemas with metadata fields
    - Gemini's function calling has specific schema requirements
    - Some fields like 'title' can cause validation issues
    
    This function recursively removes problematic fields while preserving
    the essential schema structure that Gemini needs for parameter validation.
    
    Args:
        schema (dict): Raw JSON schema from MCP tool definition
    
    Returns:
        dict: Cleaned schema compatible with Gemini function calling
    """
    if isinstance(schema, dict):
        # Remove metadata fields that can cause issues
        schema.pop("title", None)
        
        # Recursively clean nested properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])
    
    return schema


def convert_mcp_tools_to_gemini(mcp_tools):
    """
    Convert MCP tool definitions to Gemini-compatible function declarations.
    
    This bridges two different function calling systems:
    
    MCP Tool Format:
    - name: string
    - description: string  
    - inputSchema: JSON Schema object
    
    Gemini Function Declaration Format:
    - name: string
    - description: string
    - parameters: cleaned JSON Schema
    
    The conversion process:
    1. Extract tool metadata (name, description)
    2. Clean and validate the parameter schema
    3. Wrap in Gemini Tool objects for function calling
    
    Args:
        mcp_tools (list): List of MCP tool objects from server
    
    Returns:
        list: List of Gemini Tool objects ready for AI function calling
    """
    gemini_tools = []
    
    print(f"Converting {len(mcp_tools)} MCP tools to Gemini format...")

    for tool in mcp_tools:
        try:
            # Clean the parameter schema for Gemini compatibility
            parameters = clean_schema(tool.inputSchema.copy())

            # Create Gemini function declaration
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters
            )

            # Wrap in Gemini Tool object
            gemini_tool = Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)
            
            print(f"  {tool.name}: {tool.description}")
            
        except Exception as e:
            print(f"  Failed to convert tool {tool.name}: {e}")
            continue

    print(f"Successfully converted {len(gemini_tools)} tools")
    return gemini_tools


async def main():
    """
    Main application entry point.
    
    Orchestrates the complete client lifecycle:
    1. **Validation**: Ensure server URL is provided
    2. **Connection**: Establish SSE connection to MCP server  
    3. **Interaction**: Run interactive chat loop
    4. **Cleanup**: Gracefully shutdown all connections
    
    Error handling ensures cleanup happens even if exceptions occur.
    
    Usage:
        python client_sse.py http://localhost:8081/sse
    """
    # Validate command line arguments
    if len(sys.argv) < 2:
        print("Error: Missing server URL")
        print("\nUsage:")
        print("   python client_sse.py <server_url>")
        print("\nExample:")
        print("   python client_sse.py http://localhost:8081/sse")
        sys.exit(1)

    server_url = sys.argv[1]
    client = MCPClient()
    
    try:
        # Connect to MCP server
        await client.connect_to_sse_server(server_url)
        
        # Start interactive chat
        await client.chat_loop()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        # Always cleanup connections
        await client.cleanup()


if __name__ == "__main__":
    # Run the async main function
    print("Starting MCP SSE Client with Gemini AI Integration")
    asyncio.run(main())