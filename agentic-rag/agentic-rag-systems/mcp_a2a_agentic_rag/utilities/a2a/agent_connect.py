from typing import Any, Optional
from uuid import uuid4
from a2a.types import (
    AgentCard, 
    Task,
    SendMessageRequest,
    MessageSendParams
)
import httpx
from a2a.client import A2AClient
import json

class AgentConnector:
    """
    Connects to a remote A2A agent and provides a uniform method to delegate tasks
    with improved function call handling
    """

    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card

    async def send_task(self, message: str, session_id: str) -> str:
        """
        Send a task to the agent and return the Task object
        
        Args:
            message (str): The message to send to the agent
            session_id (str): The session ID for tracking the task

        Returns:
            str: The response text from the agent
        """

        async with httpx.AsyncClient(timeout = 300.0) as httpx_client:
            a2a_client = A2AClient(
                httpx_client=httpx_client,
                agent_card=self.agent_card,
            )

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'messageId': str(uuid4()),
                    'parts': [
                        {
                            'text': message,
                            'kind': 'text'
                        }
                    ]
                }
            }

            request = SendMessageRequest(
                id = str(uuid4()),
                params=MessageSendParams(
                    **send_message_payload
                )
            )

            try:
                response = await a2a_client.send_message(
                    request=request
                )

                response_data = response.model_dump(mode='json', exclude_none=True)
                
                # Try to extract the agent response with better error handling
                try:
                    # Look for the response in the result structure
                    if 'result' in response_data:
                        result = response_data['result']
                        
                        # Check if there's a status message
                        if 'status' in result and 'message' in result['status']:
                            message_parts = result['status']['message']['parts']
                            if message_parts and len(message_parts) > 0:
                                # Get the text content
                                text_content = message_parts[0].get('text', '')
                                if text_content:
                                    return text_content
                        
                        # Fallback: look for direct text content
                        if 'text' in result:
                            return result['text']
                        
                        # Another fallback: look for content in different locations
                        if 'content' in result:
                            return result['content']
                        
                        # If we can't find text, return the full response for debugging
                        return f"Response received but no text found. Full response: {json.dumps(response_data, indent=2)}"
                    
                    # If no result, try to find any text content
                    if 'text' in response_data:
                        return response_data['text']
                    
                    if 'content' in response_data:
                        return response_data['content']
                    
                    # Last resort: return the full response for debugging
                    return f"No response structure found. Full response: {json.dumps(response_data, indent=2)}"
                    
                except (KeyError, IndexError, TypeError) as e:
                    # If we can't parse the response, return what we can
                    return f"Error parsing response: {str(e)}. Raw response: {json.dumps(response_data, indent=2)}"
                    
            except Exception as e:
                return f"Error communicating with agent: {str(e)}"

    async def send_task_with_retry(self, message: str, session_id: str, max_retries: int = 3) -> str:
        """
        Send a task to the agent with retry logic for better reliability
        
        Args:
            message (str): The message to send to the agent
            session_id (str): The session ID for tracking the task
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: The response text from the agent
        """
        
        for attempt in range(max_retries):
            try:
                response = await self.send_task(message, session_id)
                
                # Check if response indicates an error
                if "Error" in response or "error" in response.lower():
                    if attempt < max_retries - 1:
                        # Wait a bit before retrying
                        import asyncio
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait a bit before retrying
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                else:
                    return f"Failed after {max_retries} attempts. Last error: {str(e)}"
        
        return f"Failed to get response after {max_retries} attempts"

    async def check_agent_health(self) -> bool:
        """
        Check if the agent is responding and healthy
        
        Returns:
            bool: True if agent is healthy, False otherwise
        """
        try:
            response = await self.send_task("Hello, are you working?", str(uuid4()))
            return "error" not in response.lower() and len(response.strip()) > 0
        except Exception:
            return False