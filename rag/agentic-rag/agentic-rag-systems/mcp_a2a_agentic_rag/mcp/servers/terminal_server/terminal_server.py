"""
MCP Terminal Server
A Model Context Protocol server that provides secure terminal command execution
"""

import asyncio
import logging
import os
import subprocess
from typing import List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
app = Server("terminal-server")

# Security: Define allowed commands (can be expanded as needed)
ALLOWED_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "grep", "find", "echo", "date", "whoami",
    "df", "du", "ps", "top", "free", "uname", "which", "wc", "sort", "uniq",
    "mkdir", "touch", "cp", "mv", "rm", "chmod", "chown"
}

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Return available terminal tools"""
    return [
        Tool(
            name="execute_command",
            description="Execute a terminal command securely. Restricted to safe commands only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute (without arguments)"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments for the command",
                        "default": []
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for command execution",
                        "default": "."
                    }
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="list_files",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                        "default": "."
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Show hidden files",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="read_file",
            description="Read the contents of a text file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                        "default": 100
                    }
                },
                "required": ["file_path"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls for terminal operations"""
    try:
        if name == "execute_command":
            result = await execute_command(
                arguments.get("command"),
                arguments.get("args", []),
                arguments.get("working_directory", ".")
            )
        elif name == "list_files":
            result = await list_files(
                arguments.get("path", "."),
                arguments.get("show_hidden", False)
            )
        elif name == "read_file":
            result = await read_file(
                arguments.get("file_path"),
                arguments.get("max_lines", 100)
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

async def execute_command(command: str, args: List[str] = None, working_directory: str = ".") -> str:
    """Execute a terminal command securely"""
    if not command:
        return "❌ No command provided"
    
    # Security check: only allow whitelisted commands
    if command not in ALLOWED_COMMANDS:
        return f"❌ Command '{command}' is not allowed. Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}"
    
    if args is None:
        args = []
    
    try:
        # Build the full command
        full_command = [command] + args
        
        logger.info(f"Executing command: {' '.join(full_command)} in {working_directory}")
        
        # Execute the command
        result = subprocess.run(
            full_command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            check=False
        )
        
        # Format the output
        output_parts = []
        output_parts.append(f"🔧 **Command:** `{' '.join(full_command)}`")
        output_parts.append(f"📁 **Directory:** `{working_directory}`")
        output_parts.append(f"🔢 **Exit Code:** {result.returncode}")
        
        if result.stdout.strip():
            output_parts.append(f"📤 **Output:**\n```\n{result.stdout.strip()}\n```")
        
        if result.stderr.strip():
            output_parts.append(f"❌ **Error:**\n```\n{result.stderr.strip()}\n```")
        
        if not result.stdout.strip() and not result.stderr.strip():
            output_parts.append("✅ **Command completed successfully (no output)**")
        
        return "\n\n".join(output_parts)
        
    except subprocess.TimeoutExpired:
        return f"❌ Command '{command}' timed out after 30 seconds"
    except FileNotFoundError:
        return f"❌ Command '{command}' not found"
    except PermissionError:
        return f"❌ Permission denied executing '{command}'"
    except Exception as e:
        return f"❌ Error executing command: {str(e)}"

async def list_files(path: str = ".", show_hidden: bool = False) -> str:
    """List files in a directory"""
    try:
        # Validate path
        if not os.path.exists(path):
            return f"❌ Path '{path}' does not exist"
        
        if not os.path.isdir(path):
            return f"❌ Path '{path}' is not a directory"
        
        # Get directory contents
        items = os.listdir(path)
        
        if not show_hidden:
            items = [item for item in items if not item.startswith('.')]
        
        if not items:
            return f"📂 Directory '{path}' is empty"
        
        # Sort items
        items.sort()
        
        # Categorize items
        files = []
        directories = []
        
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                directories.append(f"📁 {item}/")
            else:
                # Get file size
                try:
                    size = os.path.getsize(item_path)
                    size_str = format_file_size(size)
                    files.append(f"📄 {item} ({size_str})")
                except:
                    files.append(f"📄 {item}")
        
        # Format output
        output_parts = [f"📂 **Contents of '{path}':**\n"]
        
        if directories:
            output_parts.append("**Directories:**")
            output_parts.extend(directories)
            output_parts.append("")
        
        if files:
            output_parts.append("**Files:**")
            output_parts.extend(files)
        
        return "\n".join(output_parts)
        
    except PermissionError:
        return f"❌ Permission denied accessing '{path}'"
    except Exception as e:
        return f"❌ Error listing directory: {str(e)}"

async def read_file(file_path: str, max_lines: int = 100) -> str:
    """Read the contents of a text file"""
    try:
        # Validate file path
        if not file_path:
            return "❌ No file path provided"
        
        if not os.path.exists(file_path):
            return f"❌ File '{file_path}' does not exist"
        
        if not os.path.isfile(file_path):
            return f"❌ Path '{file_path}' is not a file"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > 1024 * 1024:  # 1MB limit
            return f"❌ File '{file_path}' is too large ({format_file_size(file_size)}). Maximum size: 1MB"
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Limit number of lines
        if len(lines) > max_lines:
            content = ''.join(lines[:max_lines])
            truncated_msg = f"\n\n... (truncated, showing first {max_lines} of {len(lines)} lines)"
        else:
            content = ''.join(lines)
            truncated_msg = ""
        
        # Format output
        output = f"📄 **File:** `{file_path}`\n"
        output += f"📊 **Size:** {format_file_size(file_size)}\n"
        output += f"📝 **Lines:** {len(lines)}\n\n"
        output += f"**Content:**\n```\n{content.rstrip()}\n```{truncated_msg}"
        
        return output
        
    except UnicodeDecodeError:
        return f"❌ Cannot read '{file_path}': File appears to be binary"
    except PermissionError:
        return f"❌ Permission denied reading '{file_path}'"
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

async def main():
    """Run the MCP server using stdio transport"""
    logger.info("Starting MCP Terminal Server...")
    logger.info(f"Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    logger.info("🚀 Starting MCP Terminal Server...")
    asyncio.run(main())
