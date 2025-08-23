from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from agents.agentic_rag_agent.agent import AgenticRAGAgent
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

class AgenticRAGExecutor(AgentExecutor):
    '''Implements the AgentExecutor interface for the agentic RAG agent.'''
    
    def __init__(self):
        self.agent = AgenticRAGAgent()

    async def create(self):
        '''Factory method to create and asynchronously initialize the AgenticRAGExecutor.'''
        await self.agent.create()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        '''Executes the agentic RAG agent with the provided context and event queue.'''
        query = context.get_user_input()
        task = context.current_task
        
        print(f"[AgenticRAGExecutor] Executing query: {query}")
        
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Execute the agent
            async for item in self.agent.invoke(query, task.context_id):
                is_task_complete = item.get("is_task_complete", False)
                if not is_task_complete:
                    message = item.get('updates', 'The Agentic RAG Agent is processing your document query.')
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(message, task.context_id, task.id)
                    )
                else:
                    final_result = item.get('content', 'No result received')
                    print(f"[AgenticRAGExecutor] Final result: {final_result}")
                    
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(final_result, task.context_id, task.id)
                    )
                    await asyncio.sleep(0.1)
                    break
                    
        except Exception as e:
            error_message = f"Agentic RAG Agent error: {str(e)}"
            print(f"[AgenticRAGExecutor] Error: {error_message}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_message, task.context_id, task.id)
            )
            raise

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())