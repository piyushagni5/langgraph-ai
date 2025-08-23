"""
MCP Web Search Server
A Model Context Protocol server that provides web search capabilities using SerpAPI
"""

import asyncio
import logging
import os
from typing import Any, List
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
# Initialize the MCP server
app = Server("web-search-server")

# Configuration
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# Debug logging
logger.info(f"🚀 Starting MCP Web Search Server...")
logger.info(f"Starting MCP Web Search Server...")
if SERPAPI_KEY:
    logger.info(f"✅ SERPAPI_KEY found: {SERPAPI_KEY[:10]}...")
else:
    logger.warning(f"⚠️  SERPAPI_KEY not found in environment variables")
    logger.warning(f"   Set SERPAPI_KEY to enable web search functionality")
    logger.info(f"   Available env vars: {list(os.environ.keys())}")
    logger.info(f"   Current working directory: {os.getcwd()}")
    # Try loading from .env file manually
    from dotenv import load_dotenv
    load_dotenv()
    SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
    if SERPAPI_KEY:
        logger.info(f"✅ SERPAPI_KEY loaded from .env: {SERPAPI_KEY[:10]}...")
    else:
        logger.warning(f"❌ SERPAPI_KEY still not found after loading .env")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Return available web search tools"""
    return [
        Tool(
            name="web_search",
            description="Search the web for current information and facts using SerpAPI",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "max_results": {
                        "type": "integer", 
                        "description": "Maximum number of search results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls for web search"""
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")
    
    query = arguments.get("query")
    max_results = arguments.get("max_results", 5)
    
    if not query:
        raise ValueError("Query parameter is required")
    
    try:
        result = await search_web(query, max_results)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return [TextContent(type="text", text=f"Search failed: {str(e)}")]

async def search_web(query: str, max_results: int = 5) -> str:
    """Perform web search using SerpAPI"""
    if not SERPAPI_KEY or SERPAPI_KEY == "YOUR_SERPAPI_KEY":
        return "❌ No SerpAPI key found. Please set the SERPAPI_KEY environment variable."
    
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "q": query,
                "num": max_results
            }
            
            logger.info(f"Searching for: {query}")
            response = await client.get(SERPAPI_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            organic_results = data.get("organic_results", [])
            if not organic_results:
                return f"🔍 No search results found for: '{query}'. Try different keywords."
            
            # Format search results
            formatted_results = []
            formatted_results.append(f"🔍 **Web Search Results for: {query}**\n")
            
            for i, item in enumerate(organic_results[:max_results], 1):
                title = item.get('title', 'No title')
                link = item.get('link', 'No link')
                snippet = item.get('snippet', 'No description available')
                
                formatted_results.append(f"**{i}. {title}**")
                formatted_results.append(f"🔗 {link}")
                formatted_results.append(f"📝 {snippet}")
                formatted_results.append("")  # Empty line for separation
            
            result = "\n".join(formatted_results)
            logger.info(f"Successfully returned {len(organic_results)} search results")
            return result
            
    except httpx.HTTPStatusError as e:
        logger.error(f"SerpAPI HTTP error: {e}")
        return f"❌ Search API error: {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"SerpAPI request error: {e}")
        return f"❌ Search request failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        return f"❌ Search failed: {str(e)}"

async def main():
    """Run the MCP server using stdio transport"""
    logger.info("Starting MCP Web Search Server...")
    
    # Check if SerpAPI key is configured
    if not SERPAPI_KEY:
        logger.warning("⚠️  SERPAPI_KEY not found in environment variables")
        logger.warning("   Set SERPAPI_KEY to enable web search functionality")
    else:
        logger.info("✅ SerpAPI key configured")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    logger.info("🚀 Starting MCP Web Search Server...")
    asyncio.run(main())