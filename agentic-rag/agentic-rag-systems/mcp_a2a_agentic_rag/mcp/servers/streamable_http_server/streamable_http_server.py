"""
MCP Streamable HTTP Server
A Model Context Protocol server that provides HTTP request capabilities with streaming support
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
app = Server("streamable-http-server")

# HTTP client configuration
HTTP_TIMEOUT = 30.0
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB limit

@app.list_tools()
async def list_tools() -> List[Tool]:
    """Return available HTTP tools"""
    return [
        Tool(
            name="http_request",
            description="Make HTTP requests with full control over method, headers, and body",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to make the request to"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, DELETE, etc.)",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers as key-value pairs",
                        "additionalProperties": {"type": "string"},
                        "default": {}
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body (for POST, PUT, PATCH methods)",
                        "default": ""
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds",
                        "default": 30.0,
                        "minimum": 1,
                        "maximum": 120
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="fetch_webpage",
            description="Fetch and extract text content from a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to extract text content from HTML",
                        "default": True
                    },
                    "follow_redirects": {
                        "type": "boolean",
                        "description": "Whether to follow HTTP redirects",
                        "default": True
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="api_call",
            description="Make a JSON API call with automatic content-type handling",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The API endpoint URL"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "default": "GET"
                    },
                    "data": {
                        "type": "object",
                        "description": "JSON data to send in the request body",
                        "default": {}
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional HTTP headers",
                        "additionalProperties": {"type": "string"},
                        "default": {}
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Bearer token for authentication",
                        "default": ""
                    }
                },
                "required": ["url"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls for HTTP operations"""
    try:
        if name == "http_request":
            result = await make_http_request(
                url=arguments.get("url"),
                method=arguments.get("method", "GET"),
                headers=arguments.get("headers", {}),
                body=arguments.get("body", ""),
                timeout=arguments.get("timeout", HTTP_TIMEOUT)
            )
        elif name == "fetch_webpage":
            result = await fetch_webpage(
                url=arguments.get("url"),
                extract_text=arguments.get("extract_text", True),
                follow_redirects=arguments.get("follow_redirects", True)
            )
        elif name == "api_call":
            result = await make_api_call(
                url=arguments.get("url"),
                method=arguments.get("method", "GET"),
                data=arguments.get("data", {}),
                headers=arguments.get("headers", {}),
                auth_token=arguments.get("auth_token", "")
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"HTTP tool error: {e}")
        return [TextContent(type="text", text=f"❌ HTTP Error: {str(e)}")]

async def make_http_request(url: str, method: str = "GET", headers: Dict[str, str] = None, 
                          body: str = "", timeout: float = HTTP_TIMEOUT) -> str:
    """Make a raw HTTP request"""
    if not url:
        return "❌ URL is required"
    
    if headers is None:
        headers = {}
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            logger.info(f"Making {method} request to {url}")
            
            # Prepare request
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": headers
            }
            
            if body and method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["content"] = body
            
            # Make the request
            response = await client.request(**request_kwargs)
            
            # Format response
            output_parts = []
            output_parts.append(f"🌐 **HTTP {method} Request to:** `{url}`")
            output_parts.append(f"✅ **Status:** {response.status_code} {response.reason_phrase}")
            
            # Response headers
            if response.headers:
                headers_str = "\n".join([f"  {k}: {v}" for k, v in response.headers.items()])
                output_parts.append(f"📋 **Response Headers:**\n```\n{headers_str}\n```")
            
            # Response body
            try:
                response_text = response.text
                if len(response_text) > MAX_RESPONSE_SIZE:
                    response_text = response_text[:MAX_RESPONSE_SIZE] + "\n... (truncated)"
                
                if response_text.strip():
                    # Try to format as JSON if possible
                    try:
                        json_data = response.json()
                        formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                        output_parts.append(f"📄 **Response Body (JSON):**\n```json\n{formatted_json}\n```")
                    except:
                        output_parts.append(f"📄 **Response Body:**\n```\n{response_text}\n```")
                else:
                    output_parts.append("📄 **Response Body:** (empty)")
            except Exception as e:
                output_parts.append(f"❌ **Error reading response body:** {str(e)}")
            
            return "\n\n".join(output_parts)
            
    except httpx.TimeoutException:
        return f"❌ Request to {url} timed out after {timeout} seconds"
    except httpx.ConnectError:
        return f"❌ Failed to connect to {url}"
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP error {e.response.status_code}: {e.response.reason_phrase}"
    except Exception as e:
        return f"❌ Request failed: {str(e)}"

async def fetch_webpage(url: str, extract_text: bool = True, follow_redirects: bool = True) -> str:
    """Fetch and optionally extract text from a webpage"""
    if not url:
        return "❌ URL is required"
    
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=follow_redirects) as client:
            logger.info(f"Fetching webpage: {url}")
            
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            
            output_parts = []
            output_parts.append(f"🌐 **Webpage:** `{url}`")
            output_parts.append(f"✅ **Status:** {response.status_code}")
            output_parts.append(f"📋 **Content-Type:** {content_type}")
            
            if "text/html" in content_type and extract_text:
                # Try to extract text content
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text_content = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Limit content size
                    if len(text_content) > MAX_RESPONSE_SIZE:
                        text_content = text_content[:MAX_RESPONSE_SIZE] + "\n... (truncated)"
                    
                    output_parts.append(f"📄 **Extracted Text:**\n```\n{text_content}\n```")
                    
                except ImportError:
                    output_parts.append("❌ BeautifulSoup not available for text extraction")
                    output_parts.append(f"📄 **Raw HTML:**\n```html\n{response.text[:2000]}...\n```")
                except Exception as e:
                    output_parts.append(f"❌ Text extraction failed: {str(e)}")
                    output_parts.append(f"📄 **Raw HTML:**\n```html\n{response.text[:2000]}...\n```")
            else:
                # Return raw content for non-HTML or when text extraction is disabled
                content = response.text
                if len(content) > MAX_RESPONSE_SIZE:
                    content = content[:MAX_RESPONSE_SIZE] + "\n... (truncated)"
                output_parts.append(f"📄 **Content:**\n```\n{content}\n```")
            
            return "\n\n".join(output_parts)
            
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP error {e.response.status_code}: {e.response.reason_phrase}"
    except Exception as e:
        return f"❌ Failed to fetch webpage: {str(e)}"

async def make_api_call(url: str, method: str = "GET", data: Dict[str, Any] = None, 
                       headers: Dict[str, str] = None, auth_token: str = "") -> str:
    """Make a JSON API call"""
    if not url:
        return "❌ URL is required"
    
    if data is None:
        data = {}
    
    if headers is None:
        headers = {}
    
    # Set up headers for JSON API
    api_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **headers
    }
    
    # Add authentication if provided
    if auth_token:
        api_headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            logger.info(f"Making {method} API call to {url}")
            
            # Prepare request
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": api_headers
            }
            
            if data and method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data
            
            # Make the request
            response = await client.request(**request_kwargs)
            
            # Format response
            output_parts = []
            output_parts.append(f"🔗 **API {method} Call:** `{url}`")
            output_parts.append(f"✅ **Status:** {response.status_code} {response.reason_phrase}")
            
            # Response body
            try:
                if response.text.strip():
                    # Try to parse as JSON
                    try:
                        json_data = response.json()
                        formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                        if len(formatted_json) > MAX_RESPONSE_SIZE:
                            formatted_json = formatted_json[:MAX_RESPONSE_SIZE] + "\n... (truncated)"
                        output_parts.append(f"📄 **Response:**\n```json\n{formatted_json}\n```")
                    except:
                        response_text = response.text
                        if len(response_text) > MAX_RESPONSE_SIZE:
                            response_text = response_text[:MAX_RESPONSE_SIZE] + "\n... (truncated)"
                        output_parts.append(f"📄 **Response:**\n```\n{response_text}\n```")
                else:
                    output_parts.append("📄 **Response:** (empty)")
            except Exception as e:
                output_parts.append(f"❌ **Error reading response:** {str(e)}")
            
            return "\n\n".join(output_parts)
            
    except httpx.HTTPStatusError as e:
        return f"❌ API error {e.response.status_code}: {e.response.reason_phrase}"
    except Exception as e:
        return f"❌ API call failed: {str(e)}"

async def main():
    """Run the MCP server using stdio transport"""
    logger.info("Starting MCP Streamable HTTP Server...")
    logger.info(f"Timeout: {HTTP_TIMEOUT}s, Max response size: {MAX_RESPONSE_SIZE / 1024 / 1024:.1f}MB")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    logger.info("🚀 Starting MCP Streamable HTTP Server...")
    asyncio.run(main())
