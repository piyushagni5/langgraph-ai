import asyncio
import logging
from utilities.mcp.mcp_discovery import MCPDiscovery
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from rich import print

# Configure logging for MCP cleanup issues to reduce noise during shutdown
logging.getLogger("mcp").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class MCPConnector:
    """
    Discovers the MCP servers from the config.
    Config will be loaded by the MCP discovery class
    Then it lists each server's tools
    and then caches them as MCPToolsets that are compatible with 
    Google's Agent Development Kit
    """

    def __init__(self, config_file: str = None):
        self.discovery = MCPDiscovery(config_file=config_file)
        self.tools: list[MCPToolset] = []
        
    async def _load_all_tools(self):
        """
        Loads all tools from the discovered MCP servers 
        and caches them as MCPToolsets.
        """
    
        tools = []

        for name, server in self.discovery.list_servers().items():
            try:
                conn = StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=server["command"],
                        args=server["args"]
                    ),
                    timeout=5
                )
                
                try:
                    # Wrap toolset creation with timeout and error handling
                    # This prevents hanging on unresponsive MCP servers
                    toolset = await asyncio.wait_for(
                        MCPToolset(connection_params=conn).get_tools(),
                        timeout=10.0
                    )
                    
                    if toolset:
                        # Create the actual toolset object for caching
                        mcp_toolset = MCPToolset(connection_params=conn)
                        tool_names = [tool.name for tool in toolset]
                        print(f"[bold green]Loaded tools from server [cyan]'{name}'[/cyan]:[/bold green] {', '.join(tool_names)}")
                        tools.append(mcp_toolset)
                        
                # Specific error handling for different types of connection failures
                except asyncio.TimeoutError:
                    print(f"[bold yellow]Timeout loading tools from server '{name}' (skipping)[/bold yellow]")
                except asyncio.CancelledError:
                    # Explicitly catch cancellations produced by underlying clients so they don't crash the app
                    print(f"[bold yellow]Cancelled while loading tools from server '{name}' (skipping)[/bold yellow]")
                    continue
                except ConnectionError as e:
                    print(f"[bold yellow]Connection error loading tools from server '{name}': {e} (skipping)[/bold yellow]")
                except Exception as e:
                    print(f"[bold yellow]Error loading tools from server '{name}': {e} (skipping)[/bold yellow]")
            except Exception as outer_e:
                print(f"[bold yellow]Failed to initialize server '{name}': {outer_e} (skipping)[/bold yellow]")
    
        self.tools = tools
    
    async def get_tools(self) -> list[MCPToolset]:
        """
        Returns the cached list of MCPToolsets.
        """

        await self._load_all_tools()
        return self.tools.copy()    