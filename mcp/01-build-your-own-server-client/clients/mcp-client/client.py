# Essential imports for async operations and system interaction
import asyncio  # Enables asynchronous programming patterns
import os       # Provides access to operating system interface
import sys      # System-specific parameters and runtime environment
import json     # JSON serialization and parsing utilities

# Type annotations and async context management
from typing import Optional  # Type hints for optional parameter values
from contextlib import AsyncExitStack  # Async resource management utilities
from mcp import ClientSession, StdioServerParameters  # MCP framework components
from mcp.client.stdio import stdio_client  # Standard I/O based MCP client

# Google Generative AI SDK components
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

from dotenv import load_dotenv  # Environment variable management

# Initialize environment configuration
load_dotenv()

class GeminiMCPInterface:
    def __init__(self):
        """Set up the MCP interface and initialize Gemini AI configuration."""
        self.client_session: Optional[ClientSession] = None  # Active MCP session instance
        self.resource_manager = AsyncExitStack()  # Handles async resource lifecycle

        # Extract Gemini API credentials from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment. Please configure your .env file.")

        # Initialize the Gemini AI interface
        self.ai_client = genai.Client(api_key=api_key)

    async def establish_server_connection(self, script_path: str):
        """Establish connection with MCP server and discover available capabilities."""

        # Auto-detect script runtime environment
        runtime_cmd = "python" if script_path.endswith('.py') else "node"

        # Configure server connection parameters
        connection_params = StdioServerParameters(command=runtime_cmd, args=[script_path])

        # Create stdio-based transport layer
        transport_layer = await self.resource_manager.enter_async_context(stdio_client(connection_params))

        # Extract communication channels
        self.read_stream, self.write_stream = transport_layer

        # Initialize client session for server communication
        self.client_session = await self.resource_manager.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

        # Perform handshake with server
        await self.client_session.initialize()

        # Discover available server tools
        tool_response = await self.client_session.list_tools()
        available_tools = tool_response.tools

        # Display discovered capabilities
        tool_names = [capability.name for capability in available_tools]
        print(f"\nSuccessfully connected to server. Available tools: {tool_names}")

        # Transform MCP tools for Gemini compatibility
        self.gemini_functions = transform_tools_for_gemini(available_tools)

    async def handle_user_query(self, user_input: str) -> str:
        """
        Process user queries through Gemini AI with tool execution capabilities.

        Args:
            user_input (str): User's natural language query.

        Returns:
            str: AI-generated response with tool execution results.
        """

        # Structure user input for Gemini processing
        user_message = types.Content(
            role='user',
            parts=[types.Part.from_text(text=user_input)]
        )

        # Submit query to Gemini with tool access
        ai_response = self.ai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_message],
            config=types.GenerateContentConfig(
                tools=self.gemini_functions,
            ),
        )

        # Collect response components
        response_parts = []
        message_history = []

        # Parse AI response and handle tool invocations
        for candidate in ai_response.candidates:
            if candidate.content.parts:
                for response_part in candidate.content.parts:
                    if isinstance(response_part, types.Part):
                        if response_part.function_call:
                            # Process tool execution request
                            tool_request = response_part
                            requested_tool = tool_request.function_call.name
                            tool_parameters = tool_request.function_call.args

                            print(f"\n[AI invoking tool: {requested_tool} with parameters {tool_parameters}]")

                            # Execute requested tool via MCP
                            try:
                                execution_result = await self.client_session.call_tool(requested_tool, tool_parameters)
                                tool_output = {"result": execution_result.content}
                            except Exception as error:
                                tool_output = {"error": str(error)}

                            # Package tool result for AI consumption
                            result_part = types.Part.from_function_response(
                                name=requested_tool,
                                response=tool_output
                            )

                            # Structure tool response message
                            tool_response_msg = types.Content(
                                role='tool',
                                parts=[result_part]
                            )

                            # Generate final response incorporating tool results
                            final_response = self.ai_client.models.generate_content(
                                model='gemini-2.0-flash-001',
                                contents=[
                                    user_message,
                                    tool_request,
                                    tool_response_msg,
                                ],
                                config=types.GenerateContentConfig(
                                    tools=self.gemini_functions,
                                ),
                            )

                            # Extract final AI response text
                            response_parts.append(final_response.candidates[0].content.parts[0].text)
                        else:
                            # Handle direct text responses
                            response_parts.append(response_part.text)

        # Combine all response components
        return "\n".join(response_parts)

    async def start_interactive_session(self):
        """Launch interactive chat interface for user queries."""
        print("\nMCP-Gemini Client Ready! Enter 'quit' to terminate session.")

        while True:
            user_query = input("\nYour Query: ").strip()
            if user_query.lower() == 'quit':
                break

            # Process and display AI response
            ai_response = await self.handle_user_query(user_query)
            print(f"\nResponse:\n{ai_response}")

    async def shutdown_resources(self):
        """Properly clean up all allocated resources."""
        await self.resource_manager.aclose()

def sanitize_schema_structure(schema_obj):
    """
    Remove unnecessary 'title' attributes from JSON schema recursively.

    Args:
        schema_obj (dict): Input schema object to clean.

    Returns:
        dict: Sanitized schema without title fields.
    """
    if isinstance(schema_obj, dict):
        # Remove title attribute if present
        schema_obj.pop("title", None)

        # Recursively process nested properties
        if "properties" in schema_obj and isinstance(schema_obj["properties"], dict):
            for prop_key in schema_obj["properties"]:
                schema_obj["properties"][prop_key] = sanitize_schema_structure(schema_obj["properties"][prop_key])

    return schema_obj

def transform_tools_for_gemini(mcp_tool_list):
    """
    Convert MCP tool specifications to Gemini-compatible function declarations.

    Args:
        mcp_tool_list (list): Collection of MCP tool definitions.

    Returns:
        list: Gemini Tool objects with properly formatted declarations.
    """
    converted_tools = []

    for mcp_tool in mcp_tool_list:
        # Clean and prepare parameter schema
        cleaned_params = sanitize_schema_structure(mcp_tool.inputSchema)

        # Create Gemini function declaration
        func_declaration = FunctionDeclaration(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=cleaned_params
        )

        # Wrap in Gemini Tool container
        gemini_tool_obj = Tool(function_declarations=[func_declaration])
        converted_tools.append(gemini_tool_obj)

    return converted_tools

async def initialize_application():
    """Application entry point and orchestration."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script_path>")
        sys.exit(1)

    # Initialize MCP-Gemini interface
    mcp_interface = GeminiMCPInterface()
    
    try:
        # Establish server connection and start interaction
        await mcp_interface.establish_server_connection(sys.argv[1])
        await mcp_interface.start_interactive_session()
    finally:
        # Ensure proper cleanup
        await mcp_interface.shutdown_resources()

if __name__ == "__main__":
    # Launch application in async context
    asyncio.run(initialize_application())