from collections.abc import AsyncIterable
import json
from typing import Any
from uuid import uuid4
from utilities.a2a.agent_connect import AgentConnector
from utilities.a2a.agent_discovery import AgentDiscovery
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
from a2a.types import AgentCard
from dotenv import load_dotenv
from model import get_llm_model, MODEL_INFO

load_dotenv()

class HostAgent:
    '''Enhanced orchestrator agent that coordinates A2A agents and MCP tools'''
    
    def __init__(self):
        self.system_instruction = load_instructions_file("agents/host_agent/instructions.txt")
        self.description = load_instructions_file("agents/host_agent/description.txt") 
        self.MCPConnector = MCPConnector()
        self.AgentDiscovery = AgentDiscovery()
        self._agent = None
        self._user_id = "host_agent_user"
        self._runner = None

    async def create(self):
        self._agent = await self._build_agent()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _list_agents(self) -> list[dict]:
        '''A2A tool: returns the list of dictionaries with agent card objects of registered A2A child agents'''
        cards = await self.AgentDiscovery.list_agent_cards()
        return [card.model_dump(exclude_none=True) for card in cards]

    async def _delegate_task(self, agent_name: str, message: str) -> str:
        '''Delegate task to a specific A2A agent
        
        Args:
            agent_name: Name of the agent to delegate to (e.g. "agentic_rag_agent")
            message: The task/question to send to the agent
            
        Returns:
            Response from the delegated agent
        '''
        try:
            print(f"[HostAgent] Delegating to agent: {agent_name}")
            print(f"[HostAgent] Message: {message}")
            
            cards = await self.AgentDiscovery.list_agent_cards()
            matched_card = None
            
            for card in cards:
                if card.name.lower() == agent_name.lower():
                    matched_card = card
                elif getattr(card, "id", "").lower() == agent_name.lower():
                    matched_card = card
                    
            if matched_card is None:
                available_agents = [card.name for card in cards]
                print(f"[HostAgent] Agent '{agent_name}' not found. Available: {available_agents}")
                return f"Agent '{agent_name}' not found. Available agents: {available_agents}"
                
            connector = AgentConnector(agent_card=matched_card)
            result = await connector.send_task(message=message, session_id=str(uuid4()))
            print(f"[HostAgent] Received response from {agent_name}: {len(str(result))} characters")
            return result
            
        except Exception as e:
            error_msg = f"Error delegating to {agent_name}: {str(e)}"
            print(f"[HostAgent] {error_msg}")
            return error_msg

    async def _get_model_info(self) -> str:
        '''Get information about the current model configuration'''
        info = f"""Current Model Configuration:
- LLM Model: {MODEL_INFO['llm_model']} ({MODEL_INFO['llm_provider']})
- Embedding Model: {MODEL_INFO['embed_model']} ({MODEL_INFO['embed_provider']})
- System: MCPxA2AxAgentic-RAG"""
        return info

    async def _smart_delegate_to_rag(self, question: str) -> str:
        '''Smart delegation to RAG agent for document-related questions
        
        Args:
            question: The question to ask about documents
            
        Returns:
            Response from the RAG agent
        '''
        try:
            print(f"[HostAgent] Smart delegating document question to RAG agent: {question}")
            
            # Always delegate document questions to the RAG agent
            result = await self._delegate_task("agentic_rag_agent", question)
            
            # Check if RAG agent says no documents are loaded
            if any(phrase in result.lower() for phrase in [
                "no documents", "not ingested", "please ingest", "no pdf"
            ]):
                return f"No documents have been loaded into the RAG system yet. Please upload a PDF document first, then I can answer questions about it.\n\nRAG Agent Response: {result}"
            
            return result
            
        except Exception as e:
            return f"Error querying documents: {str(e)}"

    def _is_document_question(self, query: str) -> bool:
        '''Check if query is about document content'''
        query_lower = query.lower()
        document_indicators = [
            'resume', 'document', 'pdf', 'file', 'skills', 'experience', 
            'education', 'work history', 'projects', 'qualifications',
            'what is in', 'tell me about', 'what does the document say',
            'based on the document', 'from my resume', 'in the pdf',
            'listed in', 'mentioned in', 'according to'
        ]
        return any(indicator in query_lower for indicator in document_indicators)

    async def _build_agent(self) -> LlmAgent:
        mcp_tools = await self.MCPConnector.get_tools()
        # Use the centralized model configuration
        llm_model = get_llm_model()
        
        return LlmAgent(
            name="host_orchestrator",
            model=llm_model,  # Use the dynamically configured model
            instruction=self.system_instruction,
            description=self.description,
            tools=[
                FunctionTool(self._delegate_task),
                FunctionTool(self._list_agents),
                FunctionTool(self._get_model_info),
                FunctionTool(self._smart_delegate_to_rag),
                *mcp_tools
            ]
        )

    async def invoke(self, query: str, session_id: str) -> AsyncIterable[dict]:
        '''Invoke the agent and return a stream of updates'''
        print(f"[HostAgent] Processing query: {query}")
        
        # Check if this is clearly a document question and delegate immediately
        if self._is_document_question(query):
            print(f"[HostAgent] Detected document question, delegating to RAG agent")
            try:
                result = await self._smart_delegate_to_rag(query)
                yield {
                    'is_task_complete': True,
                    'content': result
                }
                return
            except Exception as e:
                print(f"[HostAgent] Direct delegation failed: {e}")
                # Fall through to normal agent processing
        
        # Normal agent processing
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
                print_json_response(event, "================ NEW EVENT ================")
                
                if event.is_final_response():
                    final_response = ""
                    if event.content and event.content.parts and event.content.parts[-1].text:
                        final_response = event.content.parts[-1].text
                    
                    # Check if the response is problematic and try delegation
                    if self._is_problematic_response(final_response) and self._is_document_question(query):
                        print(f"[HostAgent] Detected problematic response for document question, trying delegation")
                        try:
                            delegated_result = await self._smart_delegate_to_rag(query)
                            yield {
                                'is_task_complete': True,
                                'content': delegated_result
                            }
                        except Exception as e:
                            print(f"[HostAgent] Fallback delegation failed: {e}")
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
                    yield {
                        'is_task_complete': False,
                        'updates': "Host agent is coordinating your request..."
                    }
                    
        except Exception as e:
            print(f"[HostAgent] Error in agent execution: {e}")
            # Try delegation as final fallback for document questions
            if self._is_document_question(query):
                try:
                    result = await self._smart_delegate_to_rag(query)
                    yield {
                        'is_task_complete': True,
                        'content': result
                    }
                except Exception as fallback_error:
                    yield {
                        'is_task_complete': True,
                        'content': f"Error processing request: {str(e)}. Fallback also failed: {str(fallback_error)}"
                    }
            else:
                yield {
                    'is_task_complete': True,
                    'content': f"Error processing request: {str(e)}"
                }

    def _is_problematic_response(self, response: str) -> bool:
        '''Check if response indicates the agent couldn't access documents'''
        problematic_phrases = [
            "need access to the file", "provide the file path", 
            "i need access", "could you please provide",
            "file path to your resume", "access to your resume"
        ]
        return any(phrase in response.lower() for phrase in problematic_phrases)

def print_json_response(response: Any, title: str) -> None:
    print(f"\n=== {title} ===")
    try:
        if hasattr(response, "root"):
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        rprint(syntax)
    except Exception as e:
        rprint(f"[red bold]Error printing JSON:[/red bold] {e}")
        rprint(repr(response))