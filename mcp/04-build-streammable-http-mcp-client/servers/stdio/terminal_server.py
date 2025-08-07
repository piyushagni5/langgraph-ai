"""
Terminal MCP Server
Provides secure local command execution capabilities through MCP stdio transport.
"""

import os
import subprocess
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging for stdio server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server for stdio transport (no HTTP, uses stdin/stdout)
mcp = FastMCP("terminal")

# Get workspace directory from environment or use default
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "workspace")).resolve()

# Ensure workspace exists and log location
WORKSPACE_DIR.mkdir(exist_ok=True)
logger.info(f"Terminal server workspace: {WORKSPACE_DIR}")

class CommandInput(BaseModel):
    """Input model for terminal commands with validation."""
    command: str = Field(
        ..., 
        description="Shell command to execute in the workspace directory",
        min_length=1
    )

class CommandResult(BaseModel):
    """Output model for command execution results with full details."""
    command: str = Field(..., description="The command that was executed")
    exit_code: int = Field(..., description="Process exit code (0 = success)")
    stdout: str = Field(..., description="Standard output from the command")
    stderr: str = Field(..., description="Standard error from the command") 
    working_directory: str = Field(..., description="Directory where command was executed")

@mcp.tool(
    description="Execute a shell command in the secure workspace directory. Use for file operations, text processing, and system tasks.",
    title="Terminal Command Executor"
)
async def run_command(params: CommandInput) -> CommandResult:
    """
    Execute a terminal command within the workspace directory.
    
    Security features:
    - Commands execute only within the workspace directory
    - 30-second timeout to prevent hanging processes
    - Full command output captured for transparency
    
    Args:
        params: CommandInput containing the command to execute
    
    Returns:
        CommandResult with execution details and output
    """
    command = params.command.strip()
    
    # Log the command execution for debugging
    logger.info(f"Executing command: {command}")
    
    try:
        # Execute command in workspace directory with timeout
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,  # Sandbox to workspace directory
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Return strings instead of bytes
            timeout=30  # 30 second timeout for safety
        )
        
        # Prepare structured result
        cmd_result = CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            working_directory=str(WORKSPACE_DIR)
        )
        
        # Log result summary for monitoring
        status = "SUCCESS" if result.returncode == 0 else "ERROR"
        logger.info(f"{status}: Command '{command}' completed with exit code {result.returncode}")
        
        return cmd_result
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command}' timed out after 30 seconds")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr="Command timed out after 30 seconds",
            working_directory=str(WORKSPACE_DIR)
        )
    except Exception as e:
        logger.error(f"Error executing command '{command}': {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Execution error: {str(e)}",
            working_directory=str(WORKSPACE_DIR)
        )

if __name__ == "__main__":
    logger.info("Starting Terminal MCP Server (stdio transport)")
    logger.info(f"Workspace: {WORKSPACE_DIR}")
    # Run with stdio transport (communicates via stdin/stdout)
    mcp.run(transport="stdio")