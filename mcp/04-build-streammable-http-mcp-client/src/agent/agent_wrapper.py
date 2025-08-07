"""
Agent Wrapper
Manages the Google ADK agent and MCP toolset connections with enhanced error handling and debugging.
"""

import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

# Google ADK imports
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

# Local imports
from src.utils.config_loader import config_loader
from src.utils.formatters import formatter

logger = logging.getLogger(__name__)

class AgentWrapper:
    """
    Enhanced wrapper for Google ADK agent with MCP toolset management.
    
    This class orchestrates the connection between the ADK agent and multiple MCP servers,
    providing automatic server discovery, connection health monitoring, and tool filtering.
    """
    
    def __init__(self, tool_filter: Optional[List[str]] = None):
        """
        Initialize the agent wrapper.
        
        Args:
            tool_filter: Optional list of tool names to allow. If None, all tools are loaded.
        """
        self.tool_filter = tool_filter
        self.agent: Optional[LlmAgent] = None
        self.toolsets: List[MCPToolset] = []
        self.server_status: Dict[str, str] = {}
        
        logger.info("AgentWrapper initialized")
        if tool_filter:
            logger.info(f"Tool filter active: {tool_filter}")

    async def build(self) -> None:
        """
        Build the ADK agent with MCP toolsets.
        
        This method orchestrates the entire agent building process:
        1. Loads server configurations from config file
        2. Establishes connections to each configured server
        3. Discovers and filters available tools from each server
        4. Creates the ADK agent with all loaded toolsets
        """
        logger.info("Building agent with MCP toolsets...")
        
        try:
            # Load toolsets from all configured servers
            toolsets = await self._load_toolsets()
            
            if not toolsets:
                logger.warning("No toolsets loaded - agent will have no tools available")
            
            # Create the ADK agent with Gemini 2.0 Flash Exp
            self.agent = LlmAgent(
                model="gemini-2.0-flash-exp",  # Use latest Gemini model
                name="universal_mcp_assistant",
                instruction=self._get_agent_instruction(),
                tools=toolsets  # Provide all loaded toolsets
            )
            
            self.toolsets = toolsets
            logger.info(f"Agent built successfully with {len(toolsets)} toolsets")
            
        except Exception as e:
            logger.error(f"Failed to build agent: {e}")
            raise

    def _get_agent_instruction(self) -> str:
        """Get the system instruction that defines the agent's behavior and capabilities."""
        return """You are a helpful assistant with access to temperature conversion tools and local file operations.

Your capabilities include:
- Converting temperatures between Celsius, Fahrenheit, and Kelvin
- Executing local commands for file operations
- Providing detailed explanations of conversions and formulas

When handling temperature conversions:
- Always validate input values for physical reasonableness
- Show the conversion formula used
- Round results to appropriate precision
- Handle multiple conversions in sequence when requested

When working with files:
- Use the terminal tools to create, read, and modify files
- Format output clearly and professionally
- Confirm successful file operations

Be precise, helpful, and educational in your responses. Show your work and explain the steps you're taking."""

    async def _load_toolsets(self) -> List[MCPToolset]:
        """
        Load toolsets from all configured MCP servers.
        
        This method iterates through all configured servers, attempts to connect
        to each one, and loads their available tools into MCPToolset instances.
        
        Returns:
            List of successfully connected MCPToolset instances.
        """
        servers = config_loader.get_servers()
        toolsets = []
        
        logger.info(f"Loading toolsets from {len(servers)} configured servers...")
        
        for server_name, server_config in servers.items():
            try:
                # Validate server configuration first
                if not config_loader.validate_server_config(server_name, server_config):
                    self.server_status[server_name] = "invalid_config"
                    continue
                
                # Create connection parameters based on server type
                connection_params = await self._create_connection_params(
                    server_name, server_config
                )
                
                if not connection_params:
                    self.server_status[server_name] = "connection_failed"
                    continue
                
                # Create MCPToolset and connect to server
                toolset = MCPToolset(
                    connection_params=connection_params,
                    tool_filter=self.tool_filter  # Apply tool filtering if specified
                )
                
                # Test connection by attempting to get available tools
                tools = await toolset.get_tools()
                tool_names = [tool.name for tool in tools]
                
                if tools:
                    toolsets.append(toolset)
                    self.server_status[server_name] = "connected"
                    formatter.print_tool_summary(server_name, tool_names)
                    logger.info(f"Connected to {server_name}: {len(tool_names)} tools loaded")
                else:
                    logger.warning(f"No tools found on server '{server_name}'")
                    self.server_status[server_name] = "no_tools"
                    
            except Exception as e:
                logger.error(f"Failed to connect to server '{server_name}': {e}")
                self.server_status[server_name] = f"error: {str(e)}"
                continue
        
        logger.info(f"Successfully loaded {len(toolsets)} toolsets")
        return toolsets

    async def _create_connection_params(
        self, 
        server_name: str, 
        server_config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Create appropriate connection parameters based on server transport type.
        
        Args:
            server_name: Name of the server for logging
            server_config: Server configuration dictionary
            
        Returns:
            Connection parameters object or None if creation failed
        """
        server_type = server_config["type"]
        
        try:
            if server_type == "http":
                # Create HTTP connection parameters for streamable HTTP servers
                return StreamableHTTPServerParams(url=server_config["url"])
                
            elif server_type == "stdio":
                # Create stdio connection parameters for local process servers
                command = server_config["command"]
                args = server_config.get("args", [])
                
                # Resolve relative paths to absolute paths
                if args:
                    project_root = Path(__file__).parent.parent.parent
                    resolved_args = []
                    for arg in args:
                        if not os.path.isabs(arg) and arg.endswith('.py'):
                            resolved_args.append(str(project_root / arg))
                        else:
                            resolved_args.append(arg)
                    args = resolved_args
                
                return StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=command,
                        args=args
                    ),
                    timeout=10  # Connection timeout
                )
            else:
                raise ValueError(f"Unsupported server type: {server_type}")
                
        except Exception as e:
            logger.error(f"Error creating connection params for '{server_name}': {e}")
            return None

    async def close(self) -> None:
        """
        Gracefully close all toolset connections and cleanup resources.
        """
        logger.info("Shutting down agent and closing toolset connections...")
        
        for i, toolset in enumerate(self.toolsets):
            try:
                await toolset.close()
                logger.debug(f"Closed toolset {i+1}")
            except Exception as e:
                logger.error(f"Error closing toolset {i+1}: {e}")
        
        self.toolsets.clear()
        self.agent = None
        
        # Small delay to ensure cleanup completes
        await asyncio.sleep(0.5)
        logger.info("Agent shutdown complete")

    def get_server_status(self) -> Dict[str, str]:
        """Get the current connection status of all configured servers."""
        return self.server_status.copy()

    def is_ready(self) -> bool:
        """Check if the agent is properly initialized and ready for use."""
        return self.agent is not None