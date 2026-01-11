import httpx
from a2a.client import A2AClient
from a2a.types import AgentCard

class AgentConnector:
    """
    Simple A2A Agent Connector that facilitates communication with other agents
    """
    
    def __init__(self, agent_card: AgentCard):
        """
        Initialize the connector with an agent card
        
        Args:
            agent_card: The AgentCard object containing agent connection details
        """
        self.agent_card = agent_card
    
    async def send_task(self, message: str, session_id: str) -> str:
        """
        Send a task/message to the connected agent
        
        Args:
            message: The message or task to send to the agent
            session_id: Session ID for conversation tracking
            
        Returns:
            Response from the agent
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as httpx_client:
                client = A2AClient(
                    base_url=self.agent_card.base_url.rstrip('/'),
                    httpx_client=httpx_client
                )
                
                # Send the message to the agent
                response = await client.send_request(
                    message=message,
                    session_id=session_id
                )
                
                return response
                
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"
