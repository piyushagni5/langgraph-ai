from collections.abc import AsyncIterable
import json
import os
from typing import Any
from uuid import uuid4
from utilities.common.file_loader import load_instructions_file
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from rich import print as rprint
from rich.syntax import Syntax
from utilities.mcp.mcp_connect import MCPConnector
from dotenv import load_dotenv
from model import MODEL_INFO, get_llm_model

load_dotenv()

class AgenticRAGAgent:
    '''Advanced RAG agent with agentic capabilities using centralized models'''
    
    def __init__(self):
        self.system_instruction = load_instructions_file("agents/agentic_rag_agent/instructions.txt")
        self.description = load_instructions_file("agents/agentic_rag_agent/description.txt")
        self.MCPConnector = MCPConnector()
        self._agent = None
        self._user_id = "agentic_rag_user"
        self._runner = None
        # Defer RAG orchestrator initialization to prevent import issues
        self.rag_orchestrator = None

    async def create(self):
        '''Create and asynchronously initialize the AgenticRAGAgent.'''
        try:
            print("[AgenticRAGAgent] Creating agent...")
            self._agent = await self._build_agent()
            self._runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
            print("[AgenticRAGAgent] Agent created successfully")
        except Exception as e:
            print(f"[AgenticRAGAgent] Error creating agent: {e}")
            raise

    def _get_rag_orchestrator(self):
        '''Get or create the RAG orchestrator instance.'''
        if self.rag_orchestrator is None:
            try:
                print("[AgenticRAGAgent] Initializing RAG orchestrator...")
                # Import here to avoid segmentation fault during module import
                from agents.agentic_rag_agent.rag_orchestrator import get_shared_rag_orchestrator
                self.rag_orchestrator = get_shared_rag_orchestrator()
                print("[AgenticRAGAgent] RAG orchestrator initialized")
            except Exception as e:
                print(f"[AgenticRAGAgent] Error initializing RAG orchestrator: {e}")
                raise
        return self.rag_orchestrator

    async def ingest_pdf_document(self, pdf_path: str) -> str:
        '''Ingest a PDF document into the RAG system
        
        Args:
            pdf_path: Path to the PDF file to ingest
            
        Returns:
            Success or error message about the ingestion process
        '''
        try:
            print(f"[AgenticRAGAgent] Starting PDF ingestion: {pdf_path}")
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                error_msg = f"PDF file not found: {pdf_path}"
                print(f"[AgenticRAGAgent] {error_msg}")
                return error_msg
            
            # Get RAG orchestrator with error handling
            try:
                rag_orchestrator = self._get_rag_orchestrator()
            except Exception as e:
                error_msg = f"Failed to initialize RAG orchestrator: {str(e)}"
                print(f"[AgenticRAGAgent] {error_msg}")
                return error_msg
            
            # Perform ingestion with error handling
            try:
                rag_orchestrator.ingest(pdf_path)
                result = f"Successfully ingested PDF: {pdf_path}. Document is now searchable using {MODEL_INFO['embed_model']} embeddings."
                print(f"[AgenticRAGAgent] Ingestion complete: {result}")
                return result
            except Exception as e:
                error_msg = f"Error during PDF ingestion: {str(e)}"
                print(f"[AgenticRAGAgent] {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error ingesting PDF: {str(e)}"
            print(f"[AgenticRAGAgent] {error_msg}")
            return error_msg

    async def search_documents(self, question: str) -> str:
        '''Query the ingested documents using agentic RAG with Gemini
        
        Args:
            question: The question to search for in the documents
            
        Returns:
            Answer based on the document content and web search if needed
        '''
        try:
            print(f"[AgenticRAGAgent] Searching documents for: {question}")
            
            rag_orchestrator = self._get_rag_orchestrator()
            
            if not rag_orchestrator.retriever:
                error_msg = "No documents have been ingested yet. Please ingest a PDF first using the ingest_pdf_document function."
                print(f"[AgenticRAGAgent] {error_msg}")
                return error_msg
            
            # Use the RAG orchestrator's query method which includes intelligent web search
            answer = rag_orchestrator.query(question)
            print(f"[AgenticRAGAgent] Generated answer of length: {len(answer)}")
            return answer
            
        except Exception as e:
            error_msg = f"Error querying documents: {str(e)}"
            print(f"[AgenticRAGAgent] {error_msg}")
            return error_msg

    async def get_rag_system_status(self) -> str:
        '''Get the current status of the RAG system
        
        Returns:
            Status information about the RAG system including loaded documents
        '''
        try:
            rag_orchestrator = self._get_rag_orchestrator()
            
            if rag_orchestrator.retriever:
                total_chunks = len(rag_orchestrator.text_chunks)
                total_vectors = rag_orchestrator.embedder.index.ntotal if rag_orchestrator.embedder.index else 0
                return f"""RAG system is active and ready:
- Text chunks loaded: {total_chunks}
- Vectors indexed: {total_vectors}
- LLM Model: {MODEL_INFO['llm_model']} ({MODEL_INFO['llm_provider']})
- Embedding Model: {MODEL_INFO['embed_model']} ({MODEL_INFO['embed_provider']})
- Multi-candidate ranking: Enabled
- Web search integration: Enabled
- Status: Ready to answer questions"""
            else:
                return f"""RAG system is initialized but empty:
- No documents have been ingested yet
- LLM Model: {MODEL_INFO['llm_model']} ({MODEL_INFO['llm_provider']})
- Embedding Model: {MODEL_INFO['embed_model']} ({MODEL_INFO['embed_provider']})
- Status: Ready to ingest documents
- Next step: Use ingest_pdf_document function to load documents"""
                
        except Exception as e:
            return f"Error getting RAG system status: {str(e)}"

    async def get_model_configuration(self) -> str:
        '''Get the current model configuration details
        
        Returns:
            Detailed information about the current model setup
        '''
        return f"""Current Model Configuration:
- LLM Model: {MODEL_INFO['llm_model']}
- LLM Provider: {MODEL_INFO['llm_provider']}
- Embedding Model: {MODEL_INFO['embed_model']}
- Embedding Provider: {MODEL_INFO['embed_provider']}
- Embedding Dimension: 768 (dynamic based on model)
- Multi-candidate Retrieval: Enabled (3 candidates, top-5 chunks each)
- LLM-based Ranking: Enabled (Gemini self-evaluation)
- Web Search Integration: Enabled via MCP servers
- RAG Architecture: Agentic with parallel candidate generation"""

    async def _build_agent(self) -> LlmAgent:
        mcp_tools = await self.MCPConnector.get_tools()
        llm_model = get_llm_model()
        
        # Create function tools - FunctionTool automatically extracts metadata
        function_tools = [
            FunctionTool(self.ingest_pdf_document),
            FunctionTool(self.search_documents),
            FunctionTool(self.get_rag_system_status),
            FunctionTool(self.get_model_configuration)
        ]
        
        return LlmAgent(
            name="agentic_rag_agent",
            model=llm_model,
            instruction=self.system_instruction,
            description=self.description,
            tools=function_tools + mcp_tools
        )

    async def invoke(self, query: str, session_id: str) -> AsyncIterable[dict]:
        '''Invoke the agentic RAG agent with improved function calling'''
        print(f"[AgenticRAGAgent] Processing query: {query}")
        
        # Ensure agent is properly initialized
        if not self._agent or not self._runner:
            print(f"[AgenticRAGAgent] Agent not initialized, attempting to create...")
            try:
                await self.create()
            except Exception as e:
                print(f"[AgenticRAGAgent] Failed to create agent: {e}")
                yield {
                    'is_task_complete': True,
                    'content': f"Error: Agent initialization failed: {str(e)}"
                }
                return
        
        # Check if this is a direct function call request
        if self._should_call_function_directly(query):
            try:
                result = await self._handle_direct_function_call(query)
                yield {
                    'is_task_complete': True,
                    'content': result
                }
                return
            except Exception as e:
                print(f"[AgenticRAGAgent] Direct function call failed: {e}")
                # Fall through to normal agent processing
        
        # Use the standard agent flow with improved error handling
        try:
            session = await self._runner.session_service.get_session(
                app_name=self._agent.name,
                session_id=session_id,
                user_id=self._user_id,
            )
            
            if not session:
                session = await self._runner.session_service.create_session(
                    app_name=self._agent.name,
                    session_id=session_id,
                    user_id=self._user_id,
                )

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )

            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=user_content
            ):
                print(f"[AgenticRAGAgent] Event received: {type(event).__name__}")
                
                if event.is_final_response():
                    final_response = ""
                    if event.content and event.content.parts and event.content.parts[-1].text:
                        final_response = event.content.parts[-1].text
                    
                    print(f"[AgenticRAGAgent] Final response: {final_response}")
                    
                    # Check if response indicates function calling issue and try fallback
                    if self._is_problematic_response(final_response):
                        print(f"[AgenticRAGAgent] Detected problematic response, trying fallback")
                        try:
                            fallback_result = await self._try_fallback_response(query, final_response)
                            yield {
                                'is_task_complete': True,
                                'content': fallback_result
                            }
                        except Exception as e:
                            print(f"[AgenticRAGAgent] Fallback failed: {e}")
                            yield {
                                'is_task_complete': True,
                                'content': final_response
                            }
                    else:
                        yield {
                            'is_task_complete': True,
                            'content': final_response
                        }
                else:
                    # Check if this is a function call event
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'function_call'):
                                    print(f"[AgenticRAGAgent] Function call detected: {part.function_call}")
                                elif hasattr(part, 'text'):
                                    print(f"[AgenticRAGAgent] Text part: {part.text}")
                    
                    yield {
                        'is_task_complete': False,
                        'updates': "Agentic RAG agent is processing your request..."
                    }
                    
        except Exception as e:
            print(f"[AgenticRAGAgent] Error in agent execution: {e}")
            # Final fallback - try to handle the request directly
            try:
                result = await self._emergency_fallback(query)
                yield {
                    'is_task_complete': True,
                    'content': result
                }
            except Exception as fallback_error:
                print(f"[AgenticRAGAgent] Emergency fallback failed: {fallback_error}")
                yield {
                    'is_task_complete': True,
                    'content': f"Error processing your request: {str(e)}. Please try again or check the system status."
                }

    def _should_call_function_directly(self, query: str) -> bool:
        '''Check if query should trigger direct function calling'''
        query_lower = query.lower()
        direct_triggers = [
            'ingest', 'upload', 'process pdf', 'load document',
            'system status', 'rag status', 'model config',
            'what models', 'configuration'
        ]
        return any(trigger in query_lower for trigger in direct_triggers)

    async def _handle_direct_function_call(self, query: str) -> str:
        '''Handle direct function calls based on query content'''
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ingest', 'upload', 'process', 'load']):
            # Extract file path if present
            if '.pdf' in query_lower:
                # Simple extraction - could be improved with regex
                words = query.split()
                for word in words:
                    if '.pdf' in word:
                        return await self.ingest_pdf_document(word.strip('"\''))
            return "Please provide the PDF file path to ingest."
        
        elif any(word in query_lower for word in ['status', 'rag status', 'system']):
            return await self.get_rag_system_status()
        
        elif any(word in query_lower for word in ['config', 'model', 'configuration']):
            return await self.get_model_configuration()
        
        else:
            # Default to document search
            return await self.search_documents(query)

    def _is_problematic_response(self, response: str) -> bool:
        '''Check if response indicates function calling problems'''
        problematic_phrases = [
            "need the document", "please provide", "i need the resume", 
            "could you please provide", "which project", "which document", 
            "file path", "specify which", "no documents available",
            "haven't provided", "please upload"
        ]
        return any(phrase in response.lower() for phrase in problematic_phrases)

    async def _try_fallback_response(self, query: str, problematic_response: str) -> str:
        '''Try to provide a better response when function calling fails'''
        # First check if we have documents loaded
        try:
            status = await self.get_rag_system_status()
            if "No documents have been ingested" in status or "RAG system is initialized but empty" in status:
                return f"I don't have any documents loaded yet. {status}"
            
            # If we have documents, try to search them
            return await self.search_documents(query)
            
        except Exception as e:
            return f"I encountered an issue: {problematic_response}\n\nDiagnostic: {str(e)}"

    async def _emergency_fallback(self, query: str) -> str:
        '''Emergency fallback when everything else fails'''
        try:
            # Try the most basic operations
            if any(word in query.lower() for word in ['status', 'config']):
                return await self.get_rag_system_status()
            else:
                return await self.search_documents(query)
        except Exception as e:
            return f"I'm experiencing technical difficulties. Error: {str(e)}. Please check the system status or try restarting the agent."