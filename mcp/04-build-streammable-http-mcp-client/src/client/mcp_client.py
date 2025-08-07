"""
MCP Client
Enhanced client interface with detailed debugging of MCP interactions.
"""

import logging
from typing import Optional, List, AsyncGenerator, Any
from google.genai.types import Content, Part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from src.agent.agent_wrapper import AgentWrapper
from src.utils.formatters import formatter

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Enhanced MCP client with session management, streaming responses, and detailed debugging.
    
    This client provides a high-level interface for interacting with the ADK agent
    while showing detailed information about MCP server interactions.
    """
    
    def __init__(
        self,
        app_name: str = "universal_mcp_client",
        user_id: str = "default_user",
        session_id: str = "default_session",
        tool_filter: Optional[List[str]] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the MCP client with debugging capabilities.
        
        Args:
            app_name: Application identifier for ADK
            user_id: User identifier for session context
            session_id: Session identifier for conversation context
            tool_filter: Optional list of allowed tool names
            debug_mode: Enable detailed debugging of MCP interactions
        """
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.debug_mode = debug_mode
        
        # Initialize core components
        self.session_service = InMemorySessionService()
        self.agent_wrapper = AgentWrapper(tool_filter=tool_filter)
        self.runner: Optional[Runner] = None
        
        # State tracking
        self.is_initialized = False
        
        logger.info(f"MCPClient initialized for user '{user_id}', session '{session_id}'")
        if debug_mode:
            logger.info("Debug mode enabled - detailed MCP interactions will be shown")

    async def initialize(self) -> None:
        """
        Initialize the client session and agent.
        
        This method must be called before using send_message().
        It sets up the session, builds the agent, and prepares the runner.
        """
        if self.is_initialized:
            logger.warning("Client already initialized")
            return
        
        try:
            logger.info("Initializing MCP client...")
            
            # Create ADK session
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id
            )
            logger.debug("ADK session created")
            
            # Build agent with all MCP toolsets
            await self.agent_wrapper.build()
            
            if not self.agent_wrapper.is_ready():
                raise RuntimeError("Agent failed to initialize properly")
            
            # Create runner to handle agent execution
            self.runner = Runner(
                agent=self.agent_wrapper.agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            
            self.is_initialized = True
            logger.info("MCP client initialized successfully")
            
            # Print server status summary
            status = self.agent_wrapper.get_server_status()
            connected = sum(1 for s in status.values() if s == "connected")
            logger.info(f"Server status: {connected}/{len(status)} servers connected")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            await self.shutdown()
            raise

    async def send_message(self, message: str) -> AsyncGenerator[Any, None]:
        """
        Send a message to the agent and stream the response with detailed debugging.
        
        Args:
            message: User message to send to the agent
            
        Yields:
            Streaming response events from the agent with MCP interaction details
            
        Raises:
            RuntimeError: If client is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        if not message.strip():
            raise ValueError("Message cannot be empty")
        
        logger.info(f"Sending message: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        try:
            # Create content object for ADK
            content = Content(
                role="user",
                parts=[Part(text=message)]
            )
            
            event_count = 0
            # Send to agent and yield streaming responses with debugging
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                event_count += 1
                
                # Show detailed debugging information
                if self.debug_mode:
                    formatter.print_json_response(event, f"Event #{event_count}")
                    self._analyze_event(event, event_count)
                
                yield event
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    def _analyze_event(self, event: Any, event_count: int) -> None:
        """
        Analyze and display detailed information about MCP events.
        
        Args:
            event: The event object from the agent
            event_count: Sequential event number for tracking
        """
        try:
            # Check if this is a tool-related event
            if hasattr(event, 'tool_calls') and event.tool_calls:
                for tool_call in event.tool_calls:
                    formatter.print_mcp_interaction(
                        "tool_call",
                        {
                            "tool_name": tool_call.name if hasattr(tool_call, 'name') else "Unknown",
                            "parameters": tool_call.args if hasattr(tool_call, 'args') else {},
                            "server": "MCP Server",
                            "event_number": event_count
                        }
                    )
            
            # Check for tool responses
            if hasattr(event, 'tool_responses') and event.tool_responses:
                for tool_response in event.tool_responses:
                    formatter.print_mcp_interaction(
                        "tool_response", 
                        {
                            "tool_name": getattr(tool_response, 'name', 'Unknown'),
                            "result": str(tool_response.content) if hasattr(tool_response, 'content') else "No result",
                            "status": "success" if not hasattr(tool_response, 'error') else "error",
                            "event_number": event_count
                        }
                    )
            
            # Check for agent thinking/processing
            if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts and not getattr(event, 'is_final_response', lambda: False)():
                    formatter.print_mcp_interaction(
                        "agent_thinking",
                        {
                            "content": event.content.parts[0].text if event.content.parts else "Processing...",
                            "event_number": event_count
                        }
                    )
            
            # Check for final response
            if hasattr(event, 'is_final_response') and event.is_final_response():
                content = ""
                if hasattr(event, 'content') and hasattr(event.content, 'parts') and event.content.parts:
                    content = event.content.parts[0].text
                
                formatter.print_mcp_interaction(
                    "final_response",
                    {
                        "content": content,
                        "event_number": event_count
                    }
                )
                
        except Exception as e:
            logger.debug(f"Error analyzing event {event_count}: {e}")

    def toggle_debug_mode(self) -> bool:
        """Toggle debug mode on/off and return new state."""
        self.debug_mode = not self.debug_mode
        logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        return self.debug_mode

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the client and cleanup all resources.
        """
        logger.info("Shutting down MCP client...")
        
        try:
            if self.agent_wrapper:
                await self.agent_wrapper.close()
            
            # Reset state
            self.runner = None
            self.is_initialized = False
            
            logger.info("MCP client shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> dict:
        """
        Get comprehensive client status information.
        
        Returns:
            Dictionary with detailed client status information
        """
        status = {
            "initialized": self.is_initialized,
            "debug_mode": self.debug_mode,
            "app_name": self.app_name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_ready": self.agent_wrapper.is_ready() if self.agent_wrapper else False,
            "server_status": self.agent_wrapper.get_server_status() if self.agent_wrapper else {}
        }
        
        return status