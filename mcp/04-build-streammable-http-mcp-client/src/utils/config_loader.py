"""Configuration management utilities."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Handles loading and validation of MCP server configurations."""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config path override."""
        self.config_path = self._resolve_config_path(config_path)
        self._config_cache = None
    
    def _resolve_config_path(self, config_path: str = None) -> Path:
        """Resolve configuration file path with fallbacks."""
        if config_path:
            return Path(config_path)
        
        # Try environment variable
        env_path = os.getenv("MCP_CONFIG_PATH")
        if env_path:
            return Path(env_path)
        
        # Default to config/servers.json
        project_root = Path(__file__).parent.parent.parent
        return project_root / "config" / "servers.json"
    
    def load_config(self) -> Dict[str, Any]:
        """Load and cache configuration from JSON file."""
        if self._config_cache is not None:
            return self._config_cache
        
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_cache = json.load(f)
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            return self._config_cache
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP server configurations."""
        config = self.load_config()
        return config.get("mcpServers", {})
    
    def validate_server_config(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """Validate individual server configuration."""
        required_fields = ["type"]
        
        for field in required_fields:
            if field not in server_config:
                logger.error(f"Server '{server_name}' missing required field: {field}")
                return False
        
        server_type = server_config["type"]
        
        if server_type == "http":
            if "url" not in server_config:
                logger.error(f"HTTP server '{server_name}' missing 'url' field")
                return False
                
        elif server_type == "stdio":
            if "command" not in server_config:
                logger.error(f"Stdio server '{server_name}' missing 'command' field")
                return False
        else:
            logger.error(f"Server '{server_name}' has unsupported type: {server_type}")
            return False
        
        return True

# Global config loader instance
config_loader = ConfigLoader()