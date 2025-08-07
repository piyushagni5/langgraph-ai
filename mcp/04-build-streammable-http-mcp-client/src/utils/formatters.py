"""Response formatting utilities using rich for beautiful output."""

import json
import logging
from typing import Any, Dict
from rich import print as rprint
from rich.syntax import Syntax
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()

class ResponseFormatter:
    """Handles formatting of different response types for display."""
    
    @staticmethod
    def print_json_response(response: Any, title: str = "Response") -> None:
        """Pretty print JSON responses with syntax highlighting - shows client-server interactions."""
        try:
            # Extract data from different response types
            if hasattr(response, "root"):
                data = response.root.model_dump(mode="json", exclude_none=True)
            elif hasattr(response, "model_dump"):
                data = response.model_dump(mode="json", exclude_none=True)
            elif isinstance(response, dict):
                data = response
            else:
                data = {"content": str(response)}
            
            # Create syntax-highlighted JSON
            syntax = Syntax(
                json.dumps(data, indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                line_numbers=False,
                background_color="default"
            )
            
            # Display in a panel with debugging context
            panel = Panel(
                syntax,
                title=f"DEBUG {title}",
                border_style="blue",
                expand=False
            )
            
            rprint(panel)
            
        except Exception as e:
            logger.error(f"Error formatting JSON response: {e}")
            # Fallback to simple print
            rprint(f"[red bold]ERROR formatting response:[/red bold] {e}")
            rprint(f"[yellow]Raw response:[/yellow] {repr(response)}")
    
    @staticmethod
    def print_mcp_interaction(event_type: str, details: Dict[str, Any]) -> None:
        """Display MCP client-server interactions with clear formatting."""
        interaction_text = ""
        
        # Format based on event type
        if event_type == "tool_call":
            interaction_text = f"""
[cyan]TOOL CALL:[/cyan] {details.get('tool_name', 'Unknown')}
[dim]Parameters:[/dim] {json.dumps(details.get('parameters', {}), indent=2)}
[dim]Server:[/dim] {details.get('server', 'Unknown')}
            """
        elif event_type == "tool_response":
            interaction_text = f"""
[green]TOOL RESPONSE:[/green] {details.get('tool_name', 'Unknown')}
[dim]Result:[/dim] {details.get('result', 'No result')}
[dim]Status:[/dim] {details.get('status', 'Unknown')}
            """
        elif event_type == "agent_thinking":
            interaction_text = f"""
[yellow]AGENT THINKING:[/yellow]
[dim]{details.get('content', 'Processing...')}[/dim]
            """
        elif event_type == "final_response":
            interaction_text = f"""
[magenta]FINAL RESPONSE:[/magenta]
{details.get('content', 'No content')}
            """
        
        panel = Panel(
            interaction_text.strip(),
            title=f"MCP INTERACTION - {event_type.replace('_', ' ').title()}",
            border_style="green",
            expand=False
        )
        
        rprint(panel)
    
    @staticmethod
    def print_temperature_table(conversions: Dict[str, float]) -> None:
        """Display temperature conversions in a formatted table."""
        table = Table(title="Temperature Conversions")
        
        table.add_column("Scale", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Symbol", style="green")
        
        scale_info = {
            "celsius": ("°C", "cyan"),
            "fahrenheit": ("°F", "red"),
            "kelvin": ("K", "blue")
        }
        
        for scale, value in conversions.items():
            symbol, color = scale_info.get(scale.lower(), ("", "white"))
            table.add_row(
                scale.capitalize(),
                f"{value:.2f}",
                f"[{color}]{symbol}[/{color}]"
            )
        
        console.print(table)
    
    @staticmethod
    def print_tool_summary(server_name: str, tool_names: list) -> None:
        """Display loaded tools summary."""
        tools_text = ", ".join(f"[cyan]{tool}[/cyan]" for tool in tool_names)
        
        panel = Panel(
            tools_text,
            title=f"Tools from '{server_name}'",
            border_style="green",
            expand=False
        )
        
        rprint(panel)
    
    @staticmethod
    def print_error(message: str, error: Exception = None) -> None:
        """Display error messages with consistent formatting."""
        error_text = f"[red bold]ERROR {message}[/red bold]"
        if error:
            error_text += f"\n[red]Details: {str(error)}[/red]"
        
        panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            expand=False
        )
        
        rprint(panel)
    
    @staticmethod
    def print_welcome_banner() -> None:
        """Display welcome banner for the application."""
        banner = """
[bold blue]Universal MCP Client[/bold blue]
[dim]Powered by Google ADK & Gemini[/dim]

[green]Features:[/green]
- Temperature conversions (°C, °F, K)
- Local terminal operations
- Streamable HTTP & Stdio support
- Real-time client-server interaction debugging

[yellow]Type your requests in natural language![/yellow]
[dim]Example: "Convert 25°C to Fahrenheit and save to file"[/dim]

[blue]Debug Commands:[/blue]
- 'status' - Show system status
- 'debug on/off' - Toggle detailed debugging
        """
        
        panel = Panel(
            banner,
            title="Welcome",
            border_style="blue",
            expand=False
        )
        
        rprint(panel)

# Global formatter instance
formatter = ResponseFormatter()