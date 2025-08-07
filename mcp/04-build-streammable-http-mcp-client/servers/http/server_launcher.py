"""
HTTP Server Launcher
Provides utilities to start and manage HTTP MCP servers with health monitoring.
"""

import subprocess
import sys
import time
import requests
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class ServerLauncher:
    """Manages HTTP MCP server lifecycle with health monitoring."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
    
    def start_temperature_server(self, port: int = 8000, host: str = "localhost") -> bool:
        """Start the temperature conversion server with health monitoring."""
        try:
            server_path = Path(__file__).parent / "temperature_server.py"
            
            cmd = [
                sys.executable,
                str(server_path),
                "--port", str(port),
                "--host", host,
                "--log-level", "INFO"
            ]
            
            logger.info(f"Starting temperature server on {host}:{port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            
            # Wait for server to be ready and verify health
            return self._wait_for_server(host, port)
            
        except Exception as e:
            logger.error(f"Failed to start temperature server: {e}")
            return False
    
    def _wait_for_server(self, host: str, port: int, timeout: int = 10) -> bool:
        """Wait for server to become available with health checking."""
        url = f"http://{host}:{port}/mcp"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the MCP endpoint
                # We expect a 406 "Not Acceptable" response which means the server is running
                # but needs proper MCP headers (this confirms the MCP server is active)
                response = requests.get(url, timeout=1)
                if response.status_code == 406:  # MCP server expects proper headers
                    logger.info(f"Server ready at {host}:{port}")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(0.5)
        
        logger.warning(f"Server at {host}:{port} not ready within {timeout}s")
        return False
    
    def stop_all_servers(self) -> None:
        """Stop all managed server processes gracefully."""
        for process in self.processes:
            try:
                process.terminate()  # Send SIGTERM for graceful shutdown
                process.wait(timeout=5)  # Wait up to 5 seconds
                logger.info(f"Stopped server process {process.pid}")
            except Exception as e:
                logger.error(f"Error stopping process {process.pid}: {e}")
                try:
                    process.kill()  # Force kill if graceful shutdown fails
                except:
                    pass
        
        self.processes.clear()

# Global launcher instance
launcher = ServerLauncher()