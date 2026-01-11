import json
import os
from typing import Any, Dict

class MCPDiscovery:
    """
    Reads a JSON config file defining MCP servers and provides access 
    to the server definitions under the "mcpServers" key

    Attributes:
        config_file (str): Path to the JSON configuration file.
        config (Dict[str, Any]): Parsed JSON content, expected to contain the "mcpServers" key.
    """

    def __init__(self, config_file: str = None):
        """
        Initializes the MCPDiscovery with a configuration file.

        Args:
            config_file (str, optional): Path to the JSON configuration file. 
            If None, defaults to 'mcp_config.json' 
            located in the same directory as this module.
        """
        if config_file is None:
            # Look for config file in project root
            self.config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                'mcp_config.json'
            )
        else:
            self.config_file = config_file
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid configuration format in {self.config_file}")
            
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading configuration file {self.config_file}: {e}")
    
    def list_servers(self) -> Dict[str, Any]:
        """
        Returns the MCP servers defined in the configuration file.

        Returns:
            Dict[str, Any]: The content of the "mcpServers" key from the config.
        
        Raises:
            KeyError: If "mcpServers" key is not found in the configuration.
        """
        if 'mcpServers' not in self.config:
            raise KeyError(f"'mcpServers' key not found in {self.config_file}")
        
        return self.config.get('mcpServers', {})