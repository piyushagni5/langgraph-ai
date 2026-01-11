"""
Executor implementation for the Agentic RAG Agent.
Handles task execution, status updates, and error management for RAG operations.
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import (
    new_task,
    new_agent_text_message
)
from a2a.utils.errors import ServerError
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError
)
import asyncio
# Import the AgenticRAGAgent class
from agents.agentic_rag_agent.agent import AgenticRAGAgent


class AgenticRAGExecutor(AgentExecutor):
    """
    Executor for the Agentic RAG Agent that handles document queries.
    Manages task lifecycle, status updates, and streaming responses.
    """
    
    def __init__(self):
        # Initialize the underlying RAG agent
        self.agent = AgenticRAGAgent()

    async def create(self):
        """Factory method to create and initialize the executor asynchronously."""
        await self.agent.create()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the RAG agent with user query and stream results.
        Handles task creation, status updates, and error management.
        """
        query = context.get_user_input()
        task = context.current_task
        
        print(f"[AgenticRAGExecutor] Executing query: {query}")
        
        # Create new task if none exists
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Stream agent execution results
            async for item in self.agent.invoke(query, task.context_id):
                is_task_complete = item.get("is_task_complete", False)
                if not is_task_complete:
                    # Send progress updates to client
                    message = item.get('updates', 'The Agentic RAG Agent is processing your document query.')
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(message, task.context_id, task.id)
                    )
                else:
                    # Send final result and complete task
                    final_result = item.get('content', 'No result received')
                    print(f"[AgenticRAGExecutor] Final result: {final_result}")
                    
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(final_result, task.context_id, task.id)
                    )
                    await asyncio.sleep(0.1)  # Brief pause for event processing
                    break
                    
        except Exception as e:
            # Handle execution errors and update task status
            error_message = f"Agentic RAG Agent error: {str(e)}"
            print(f"[AgenticRAGExecutor] Error: {error_message}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_message, task.context_id, task.id)
            )
            raise

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        """Cancel operation is not supported for this agent."""
        raise ServerError(error=UnsupportedOperationError())