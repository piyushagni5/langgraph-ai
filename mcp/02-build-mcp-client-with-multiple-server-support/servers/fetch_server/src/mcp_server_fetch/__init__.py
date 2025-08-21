"""
MCP Fetch Server Package

A Model Context Protocol server providing tools to fetch and convert web content
for usage by LLMs. This server enables intelligent agents to retrieve current
information from the web with proper content formatting.

Features:
  - Web content fetching with HTTP/HTTPS support
  - HTML to Markdown conversion for LLM consumption
  - Robots.txt compliance (configurable)
  - Proxy support for enterprise environments
  - Custom User-Agent configuration

Usage:
    python -m mcp_server_fetch [options]
"""

from .server import serve


def main():
    """
    MCP Fetch Server entry point with command line argument support.
    
    Configures and starts the fetch server with optional parameters for
    user agent customization, robots.txt handling, and proxy configuration.
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Provide web fetching capabilities for Model Context Protocol"
    )
    parser.add_argument("--user-agent", type=str, help="Custom User-Agent string")
    parser.add_argument(
        "--ignore-robots-txt",
        action="store_true",
        help="Ignore robots.txt restrictions",
    )
    parser.add_argument("--proxy-url", type=str, help="Proxy URL to use for requests")

    args = parser.parse_args()
    asyncio.run(serve(args.user_agent, args.ignore_robots_txt, args.proxy_url))


if __name__ == "__main__":
    main()
