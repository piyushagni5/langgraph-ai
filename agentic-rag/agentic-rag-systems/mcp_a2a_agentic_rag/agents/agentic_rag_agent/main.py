import asyncio
import uvicorn
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
import asyncclick as click
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

from agents.agentic_rag_agent.agent_executor import AgenticRAGExecutor


async def serve_agentic_rag_agent(host: str, port: int):
    '''Run the agentic RAG agent server.'''
    
    skill = AgentSkill(
        id="agentic_rag_skill",
        name="agentic_rag_skill",
        description="Advanced document processing and question answering using agentic RAG with multi-candidate ranking and web search integration",
        tags=["rag", "documents", "pdf", "search", "qa", "ranking"],
        examples=[
            "Process this PDF document and make it searchable for questions.",
            "Answer questions about uploaded documents using intelligent retrieval.",
            "Find information in documents and supplement with web search if needed.",
            "Use multi-candidate ranking to provide the best possible answers from document context."
        ]
    )

    agent_card = AgentCard(
        name="agentic_rag_agent",
        description="Intelligent document processing agent with advanced RAG capabilities, multi-candidate ranking, and web search integration",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text", "file"],
        defaultOutputModes=["text"],
        skills=[skill],
        capabilities=AgentCapabilities(streaming=True),
    )

    # Create agent executor
    print(f"[AgenticRAGAgent] Creating agent executor...")
    agent_executor = AgenticRAGExecutor()
    await agent_executor.create()
    print(f"[AgenticRAGAgent] Agent executor created successfully")

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
    print(f"[AgenticRAGAgent] Server app built successfully")
    
    # Add routes using Starlette routing instead of FastAPI decorators
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health_check(request):
        return JSONResponse({"status": "healthy", "agent": "agentic_rag_agent", "port": port})
    
    async def debug_info(request):
        return JSONResponse({
            "agent": "agentic_rag_agent",
            "status": "running",
            "port": port,
            "functions": [
                "ingest_pdf_document",
                "search_documents", 
                "get_rag_system_status",
                "get_model_configuration"
            ]
        })
    
    # Add routes to the existing app
    app.routes.extend([
        Route("/health", health_check, methods=["GET"]),
        Route("/debug", debug_info, methods=["GET"])
    ])

    print(f"[AgenticRAGAgent] About to start uvicorn server on {host}:{port}...")
    print(f"[AgenticRAGAgent] Starting uvicorn server on {host}:{port}...")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    print(f"[AgenticRAGAgent] Server instance created, starting to serve...")
    await server_instance.serve()
    print(f"[AgenticRAGAgent] Server finished serving (this shouldn't print during normal operation)")  # This should never print

@click.command()
@click.option('--host', default='localhost', help='Host for the agent server')
@click.option('--port', default=10002, help='Port for the agent server')
async def main(host: str, port: int):
    '''Main function to create and run the agentic RAG agent.'''
    await serve_agentic_rag_agent(host, port)

if __name__ == "__main__":
    asyncio.run(main())