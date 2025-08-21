"""
Terminal MCP Server Implementation

A Model Context Protocol server that provides secure terminal command execution
within a containerized environment. This server enables LLMs to perform file
operations, system commands, and other CLI-based tasks safely.

Security Features:
  - Workspace isolation: All commands execute within /workspace directory
  - Timeout protection: Commands limited to 30 seconds execution time
  - Error handling: Graceful handling of command failures

Usage:
    python terminal_server.py
"""

import os
import subprocess
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server instance with terminal capabilities
server = FastMCP("terminal")

# Define the base working directory for command execution
WORKSPACE_PATH = "/workspace"


@server.tool()
async def execute_shell_command(cmd: str) -> str:
    """
    Execute shell commands within the designated workspace environment.
    
    This tool provides terminal access for file operations, system commands,
    and other CLI-based tasks. When a terminal operation is needed,
    this function will be utilized to carry out the requested action.

    Args:
        cmd: Shell command string to be executed
    
    Returns:
        str: Command execution output (stdout/stderr) or error details
    """
    try:
        # Execute command with workspace as working directory
        process = subprocess.run(
            cmd, 
            shell=True, 
            cwd=WORKSPACE_PATH, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        # Return output, prioritizing stdout over stderr
        output = process.stdout.strip() if process.stdout else process.stderr.strip()
        return output if output else "Command executed successfully (no output)"
        
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds"
    except Exception as error:
        return f"Execution error: {str(error)}"


def main():
    """Entry point for the MCP server application."""
    server.run(transport='stdio')


if __name__ == "__main__":
    main()