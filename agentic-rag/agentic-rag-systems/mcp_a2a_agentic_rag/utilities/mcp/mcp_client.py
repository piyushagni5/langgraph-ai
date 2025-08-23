"""
MCP Client for Agentic RAG Integration
Provides a client interface for connecting to and using MCP servers in the agentic RAG system
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from utilities.mcp.mcp_discovery import MCPDiscovery

logger = logging.getLogger(__name__)

class MCPClient:
    """
    MCP Client for integrating with multiple MCP servers in the agentic RAG system.
    This client manages connections to web search, terminal, and HTTP servers.
    """
    
    def __init__(self, config_file: str = None):
        self.discovery = MCPDiscovery(config_file=config_file)
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.stack = AsyncExitStack()
        self.is_connected = False
    
    async def connect_all_servers(self):
        """Connect to all configured MCP servers"""
        servers = self.discovery.list_servers()
        logger.info(f"Connecting to {len(servers)} MCP servers...")
        
        for server_name, server_config in servers.items():
            try:
                await self._connect_server(server_name, server_config)
                logger.info(f"✅ Connected to {server_name}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {server_name}: {e}")
        
        self.is_connected = True
        logger.info(f"MCP Client connected to {len(self.connections)} servers")
    
    async def _connect_server(self, server_name: str, server_config: Dict[str, Any]):
        """Connect to a single MCP server"""
        # Create server parameters
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", {})
        )
        
        # Establish connection
        read_stream, write_stream = await self.stack.enter_async_context(
            stdio_client(server_params)
        )
        
        # Create session
        session = await self.stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # Initialize session
        await session.initialize()
        
        # Get available tools
        tools_response = await session.list_tools()
        tools = [tool.name for tool in tools_response.tools]
        
        # Store connection info
        self.connections[server_name] = {
            "session": session,
            "tools": tools,
            "config": server_config
        }
        
        logger.info(f"Server {server_name} tools: {', '.join(tools)}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on a specific MCP server"""
        if not self.is_connected:
            raise RuntimeError("Not connected to MCP servers. Call connect_all_servers() first.")
        
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' not found. Available servers: {list(self.connections.keys())}")
        
        connection = self.connections[server_name]
        session = connection["session"]
        
        if tool_name not in connection["tools"]:
            raise ValueError(f"Tool '{tool_name}' not found on server '{server_name}'. Available tools: {connection['tools']}")
        
        try:
            result = await session.call_tool(tool_name, arguments)
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    return result.content[0].text
                elif hasattr(result.content, 'text'):
                    return result.content.text
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            raise
    
    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Convenience method for web search"""
        return await self.call_tool("web_search", "web_search", {
            "query": query,
            "max_results": max_results
        })
    
    async def execute_command(self, command: str, args: List[str] = None, working_directory: str = ".") -> str:
        """Convenience method for terminal command execution"""
        if args is None:
            args = []
        return await self.call_tool("terminal", "execute_command", {
            "command": command,
            "args": args,
            "working_directory": working_directory
        })
    
    async def list_files(self, path: str = ".", show_hidden: bool = False) -> str:
        """Convenience method for listing files"""
        return await self.call_tool("terminal", "list_files", {
            "path": path,
            "show_hidden": show_hidden
        })
    
    async def read_file(self, file_path: str, max_lines: int = 100) -> str:
        """Convenience method for reading files"""
        return await self.call_tool("terminal", "read_file", {
            "file_path": file_path,
            "max_lines": max_lines
        })
    
    async def http_request(self, url: str, method: str = "GET", headers: Dict[str, str] = None, 
                          body: str = "", timeout: float = 30.0) -> str:
        """Convenience method for HTTP requests"""
        if headers is None:
            headers = {}
        return await self.call_tool("http", "http_request", {
            "url": url,
            "method": method,
            "headers": headers,
            "body": body,
            "timeout": timeout
        })
    
    async def fetch_webpage(self, url: str, extract_text: bool = True, follow_redirects: bool = True) -> str:
        """Convenience method for fetching webpages"""
        return await self.call_tool("http", "fetch_webpage", {
            "url": url,
            "extract_text": extract_text,
            "follow_redirects": follow_redirects
        })
    
    async def api_call(self, url: str, method: str = "GET", data: Dict[str, Any] = None, 
                      headers: Dict[str, str] = None, auth_token: str = "") -> str:
        """Convenience method for API calls"""
        if data is None:
            data = {}
        if headers is None:
            headers = {}
        return await self.call_tool("http", "api_call", {
            "url": url,
            "method": method,
            "data": data,
            "headers": headers,
            "auth_token": auth_token
        })
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all connected servers"""
        status = {}
        for server_name, connection in self.connections.items():
            status[server_name] = {
                "connected": True,
                "tools": connection["tools"],
                "tool_count": len(connection["tools"])
            }
        return status
    
    def list_all_tools(self) -> Dict[str, List[str]]:
        """List all available tools across all servers"""
        return {server_name: connection["tools"] for server_name, connection in self.connections.items()}
    
    async def cleanup(self):
        """Clean up all connections"""
        try:
            if self.stack:
                # ADDED: More robust cleanup to handle Python 3.13/anyio cancellation issues
                try:
                    await asyncio.wait_for(self.stack.aclose(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("MCP client cleanup timed out")
                except (RuntimeError, asyncio.CancelledError) as e:
                    # Handle cancellation scope errors in Python 3.13
                    logger.warning(f"Cancellation error during cleanup (expected in Python 3.13): {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        finally:
            self.connections.clear()
            self.is_connected = False
            logger.info("MCP Client cleaned up")

# Global MCP client instance for the application
_mcp_client: Optional[MCPClient] = None

async def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client instance"""
    global _mcp_client
    if _mcp_client is None or not _mcp_client.is_connected:
        _mcp_client = MCPClient()
        await _mcp_client.connect_all_servers()
    return _mcp_client

async def cleanup_mcp_client():
    """Clean up the global MCP client instance"""
    global _mcp_client
    if _mcp_client:
        try:
            # ADDED: More defensive cleanup for Python 3.13 compatibility
            await asyncio.wait_for(_mcp_client.cleanup(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("MCP client global cleanup timed out")
        except (RuntimeError, asyncio.CancelledError) as e:
            logger.warning(f"Cancellation error during global cleanup (expected in Python 3.13): {e}")
        except Exception as e:
            logger.warning(f"Error cleaning up MCP client: {e}")
        finally:
            _mcp_client = None
