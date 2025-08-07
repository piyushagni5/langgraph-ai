"""
CLI Main Entry Point
Enhanced command-line interface with detailed MCP interaction debugging.
"""

import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.client.mcp_client import MCPClient
from src.utils.formatters import formatter
from servers.http.server_launcher import launcher

# Configuration
DEFAULT_TOOLS = [
    'celsius_to_fahrenheit',
    'fahrenheit_to_celsius', 
    'celsius_to_kelvin',
    'kelvin_to_celsius',
    'fahrenheit_to_kelvin',
    'kelvin_to_fahrenheit',
    'run_command'
]

# Suppress specific warnings before logging configuration
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk")
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
warnings.filterwarnings("ignore", message=".*auth_config.*")
warnings.filterwarnings("ignore", message=".*BaseAuthenticatedTool.*")

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_client.log")
    ]
)

# Suppress verbose logs and warnings from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google_adk").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)
logging.getLogger("google.adk.tools").setLevel(logging.ERROR)
logging.getLogger("mcp.client").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class MCPClientCLI:
    """Enhanced CLI for MCP Client with detailed debugging and server management."""
    
    def __init__(self):
        self.client: Optional[MCPClient] = None
        self.server_started = False
        self.debug_mode = True  # Enable debug mode by default
    
    async def start_servers(self) -> bool:
        """Start required HTTP servers with health monitoring."""
        try:
            logger.info("Starting HTTP servers...")
            
            # Start temperature server
            if launcher.start_temperature_server(port=8000):
                self.server_started = True
                logger.info("Temperature server started successfully")
                return True
            else:
                logger.error("Failed to start temperature server")
                return False
                
        except Exception as e:
            logger.error(f"Error starting servers: {e}")
            return False
    
    async def initialize_client(self) -> bool:
        """Initialize the MCP client with debugging enabled."""
        try:
            logger.info("Initializing MCP client...")
            
            self.client = MCPClient(
                app_name="universal_mcp_client",
                user_id="cli_user_001", 
                session_id="cli_session_001",
                tool_filter=DEFAULT_TOOLS,
                debug_mode=self.debug_mode  # Enable debugging
            )
            
            await self.client.initialize()
            logger.info("MCP client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            formatter.print_error("Client initialization failed", e)
            return False
    
    async def chat_loop(self) -> None:
        """Main interactive chat loop with enhanced debugging."""
        formatter.print_welcome_banner()
        
        print("\nChat started. Commands:")
        print("  - Type your requests in natural language")
        print("  - 'status' - Show system status")
        print("  - 'debug on/off' - Toggle detailed debugging") 
        print("  - 'help' - Show example requests")
        print("  - 'quit', 'exit', ':q' - Exit the application\n")
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit', ':q']:
                        print("Goodbye!")
                        break
                    elif user_input.lower() == 'status':
                        self._show_status()
                        continue
                    elif user_input.lower().startswith('debug'):
                        self._handle_debug_command(user_input)
                        continue
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    
                    # Send message to agent with debugging
                    await self._handle_user_message(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Goodbye!")
                    break
                except EOFError:
                    print("\n\nEnd of input. Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}")
                    formatter.print_error("An error occurred", e)
                    
        except Exception as e:
            logger.error(f"Critical error in chat loop: {e}")
            formatter.print_error("Critical error occurred", e)
    
    async def _handle_user_message(self, message: str) -> None:
        """Handle a user message with detailed MCP interaction debugging."""
        event_count = 0
        final_response = None
        
        try:
            print(f"\nAssistant: Processing your request...")
            if self.debug_mode:
                print("[DEBUG MODE] Showing detailed MCP interactions:\n")
            
            async for event in self.client.send_message(message):
                event_count += 1
                
                # In debug mode, detailed interactions are shown by the client
                # Here we just count events and look for the final response
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    final_response = event
                    break
            
            # Display final response
            if final_response and hasattr(final_response, 'content'):
                if hasattr(final_response.content, 'parts') and final_response.content.parts:
                    response_text = final_response.content.parts[0].text
                    print(f"\nFinal Response:\n{response_text}\n")
                else:
                    print("Task completed (no text response)\n")
            else:
                print("No final response received\n")
                
            if self.debug_mode:
                print(f"Total events processed: {event_count}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            formatter.print_error("Failed to process message", e)
    
    def _handle_debug_command(self, command: str) -> None:
        """Handle debug mode toggle commands."""
        parts = command.split()
        if len(parts) > 1:
            mode = parts[1].lower()
            if mode == 'on':
                self.debug_mode = True
                if self.client:
                    self.client.debug_mode = True
                print("Debug mode enabled - detailed MCP interactions will be shown")
            elif mode == 'off':
                self.debug_mode = False
                if self.client:
                    self.client.debug_mode = False
                print("Debug mode disabled - only final responses will be shown")
            else:
                print("Usage: debug on/off")
        else:
            current_state = "enabled" if self.debug_mode else "disabled"
            print(f"Debug mode is currently {current_state}")
    
    def _show_status(self) -> None:
        """Show current system status with detailed information."""
        if not self.client:
            print("Client not initialized")
            return
        
        status = self.client.get_status()
        
        print("\nSystem Status:")
        print(f"  - Client initialized: {'[OK]' if status['initialized'] else '[FAIL]'}")
        print(f"  - Agent ready: {'[OK]' if status['agent_ready'] else '[FAIL]'}")
        print(f"  - Debug mode: {'ON' if status['debug_mode'] else 'OFF'}")
        print(f"  - User: {status['user_id']}")
        print(f"  - Session: {status['session_id']}")
        
        print("\nServer Status:")
        for server_name, server_status in status['server_status'].items():
            status_icon = "[OK]" if server_status == "connected" else "[FAIL]"
            print(f"  - {server_name}: {status_icon} {server_status}")
        print()
    
    def _show_help(self) -> None:
        """Show example requests and debugging tips."""
        examples = [
            "Convert 25 degrees Celsius to Fahrenheit",
            "What is 100°F in Celsius and Kelvin?", 
            "Convert 300 Kelvin to Celsius and Fahrenheit",
            "Convert 0°C to all other temperature scales",
            "Convert room temperature (20°C) to Fahrenheit and save to file",
            "Create a temperature conversion table for 0, 25, 50, 75, 100°C"
        ]
        
        debug_tips = [
            "Use 'debug on' to see detailed MCP server interactions",
            "Watch for tool calls, parameters, and server responses",
            "Each event shows the communication between client and servers",
            "Use 'debug off' to see only final responses"
        ]
        
        print("\nExample Requests:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        
        print("\nDebug Mode Tips:")
        for i, tip in enumerate(debug_tips, 1):
            print(f"  {i}. {tip}")
        print()
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop servers."""
        try:
            if self.client:
                await self.client.shutdown()
            
            if self.server_started:
                launcher.stop_all_servers()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point for the CLI application with enhanced debugging."""
    # Additional warning suppression at runtime
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    cli = MCPClientCLI()
    
    try:
        # Start servers
        print("Starting Universal MCP Client with Debug Mode...")
        
        if not await cli.start_servers():
            print("Failed to start required servers")
            return 1
        
        # Initialize client
        if not await cli.initialize_client():
            print("Failed to initialize client")
            return 1
        
        # Start chat loop with debugging
        await cli.chat_loop()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        formatter.print_error("Unexpected error occurred", e)
        return 1
    finally:
        await cli.cleanup()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except asyncio.CancelledError:
        # Suppress cancelled error messages during shutdown
        logger.debug("Main coroutine cancelled during shutdown")
        sys.exit(0)