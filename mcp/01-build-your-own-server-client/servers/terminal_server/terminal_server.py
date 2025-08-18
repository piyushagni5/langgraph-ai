import os
import subprocess
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server instance with terminal capabilities
server = FastMCP("terminal")

# Define the base working directory for command execution
WORKSPACE_PATH = os.path.expanduser("~/Desktop/github/langgraph-ai/mcp/01-build-your-own-server-client/workspace")

@server.tool()
async def execute_shell_command(cmd: str) -> str:
    """
    Execute shell commands within the designated workspace environment.
    
    This tool provides terminal access for file operations, system commands,
    and other CLI-based tasks. When a terminal operation is needed,
    this function will be utilized to carry out the requested action.

    Parameters:
        cmd: Shell command string to be executed
    
    Returns:
        Command execution output (stdout/stderr) or error details
    """
    try:
        # Execute command with workspace as working directory
        process = subprocess.run(
            cmd, 
            shell=True, 
            cwd=WORKSPACE_PATH, 
            capture_output=True, 
            text=True,
            timeout=30  # Add timeout for safety
        )
        
        # Return output, prioritizing stdout over stderr
        output = process.stdout.strip() if process.stdout else process.stderr.strip()
        return output if output else "Command executed successfully (no output)"
        
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds"
    except Exception as error:
        return f"Execution error: {str(error)}"

def main():
    """Entry point for the MCP server application"""
    server.run(transport='stdio')

if __name__ == "__main__":
    main()