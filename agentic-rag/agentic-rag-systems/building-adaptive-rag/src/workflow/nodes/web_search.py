from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from src.workflow.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    
    # Initialize documents - this was the missing part!
    documents = state.get("documents", [])  # Get existing documents or empty list
    
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    
    # Add web results to existing documents (or create new list if documents was empty)
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}