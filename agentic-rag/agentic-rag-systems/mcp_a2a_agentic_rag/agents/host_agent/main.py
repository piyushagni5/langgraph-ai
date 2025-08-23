import asyncio
import uvicorn
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
import asyncclick as click
from a2a.server.request_handlers import DefaultRequestHandler
from agents.host_agent.agent_executor import HostAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

async def serve_host_agent(host: str, port: int):
    '''Run the host orchestrator agent server.'''
    skill = AgentSkill(
        id="host_orchestrator_skill",
        name="host_orchestrator_skill", 
        description="An orchestrator that coordinates A2A agents and MCP tools for intelligent document processing and web search",
        tags=["host", "orchestrator", "rag", "search"],
        examples=[
            "Search my documents for information about machine learning and supplement with web search if needed.",
            "Find the latest information about AI developments using both my documents and web search.",
            "Process this PDF document and answer questions about it using RAG capabilities."
        ]
    )

    agent_card = AgentCard(
        name="host_orchestrator",
        description="An intelligent orchestrator that coordinates document processing, web search, and question answering",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"], 
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True),
    )

    # Create agent executor
    print(f"[HostAgent] Creating agent executor...")
    agent_executor = HostAgentExecutor()
    await agent_executor.create()
    print(f"[HostAgent] Agent executor created successfully")

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    # Add health check endpoint
    app = server.build()
    print(f"[HostAgent] Server app built successfully")
    
    # Add routes using Starlette routing instead of FastAPI decorators
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health_check(request):
        return JSONResponse({"status": "healthy", "agent": "host_orchestrator", "port": port})
    
    async def debug_info(request):
        return JSONResponse({
            "agent": "host_orchestrator",
            "status": "running",
            "port": port,
            "description": "Host orchestrator agent for coordinating A2A agents and MCP tools"
        })
    
    # Add routes to the existing app
    app.routes.extend([
        Route("/health", health_check, methods=["GET"]),
        Route("/debug", debug_info, methods=["GET"])
    ])

    # Use uvicorn.Config and Server to avoid event loop issues
    print(f"[HostAgent] About to start uvicorn server on {host}:{port}...")
    print(f"[HostAgent] Starting uvicorn server on {host}:{port}...")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    print(f"[HostAgent] Server instance created, starting to serve...")
    await server_instance.serve()
    print(f"[HostAgent] Server finished serving (this shouldn't print during normal operation)")  # This should never print

@click.command()
@click.option('--host', default='localhost', help='Host for the agent server')
@click.option('--port', default=10001, help='Port for the agent server')
async def main(host: str, port: int):
    '''Main function to create and run the host orchestrator agent.'''
    await serve_host_agent(host, port)

if __name__ == "__main__":
    asyncio.run(main())