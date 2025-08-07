"""
Temperature Conversion MCP Server
Provides comprehensive temperature conversion tools supporting Celsius, Fahrenheit, and Kelvin.
"""

import click
import logging
from typing import Union
from pydantic import BaseModel, Field, validator
from mcp.server.fastmcp import FastMCP

@click.command()
@click.option("--port", default=8000, help="Port to run the server on")
@click.option("--host", default="localhost", help="Host to bind the server to")
@click.option("--log-level", default="INFO", help="Logging level")

def main(port: int, host: str, log_level: str) -> None:
    """Launch the Temperature Conversion MCP Server."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Temperature Conversion MCP Server...")

    # Create FastMCP server with streamable HTTP transport
    mcp = FastMCP(
        "Temperature Converter",
        host=host,
        port=port,
        stateless_http=True  # Enable streamable HTTP protocol
    )

    # Input/Output Models for type safety and validation
    class TemperatureInput(BaseModel):
        """Input model for temperature conversion with validation."""
        temperature: float = Field(
            ..., 
            description="Temperature value to convert"
        )
        
        @validator('temperature')
        def validate_temperature_range(cls, v):
            """Validate temperature is within physically reasonable bounds."""
            if v < -273.15:  # Below absolute zero in Celsius
                raise ValueError("Temperature cannot be below absolute zero (-273.15°C)")
            return v

    class TemperatureResult(BaseModel):
        """Output model for temperature conversion results."""
        original_value: float = Field(..., description="Original temperature value")
        original_scale: str = Field(..., description="Original temperature scale")
        converted_value: float = Field(..., description="Converted temperature value")
        converted_scale: str = Field(..., description="Target temperature scale")
        formula: str = Field(..., description="Conversion formula used")

    # Core conversion functions (business logic)
    def celsius_to_fahrenheit_calc(celsius: float) -> float:
        """Convert Celsius to Fahrenheit using standard formula."""
        return (celsius * 9/5) + 32

    def fahrenheit_to_celsius_calc(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius using standard formula."""
        return (fahrenheit - 32) * 5/9

    def celsius_to_kelvin_calc(celsius: float) -> float:
        """Convert Celsius to Kelvin by adding absolute zero offset."""
        return celsius + 273.15

    def kelvin_to_celsius_calc(kelvin: float) -> float:
        """Convert Kelvin to Celsius by subtracting absolute zero offset."""
        return kelvin - 273.15

    def fahrenheit_to_kelvin_calc(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin via Celsius intermediate."""
        celsius = fahrenheit_to_celsius_calc(fahrenheit)
        return celsius_to_kelvin_calc(celsius)

    def kelvin_to_fahrenheit_calc(kelvin: float) -> float:
        """Convert Kelvin to Fahrenheit via Celsius intermediate."""
        celsius = kelvin_to_celsius_calc(kelvin)
        return celsius_to_fahrenheit_calc(celsius)

    # MCP Tool Registrations - These become available to clients
    
    @mcp.tool(
        description="Convert temperature from Celsius to Fahrenheit",
        title="Celsius to Fahrenheit Converter"
    )
    def celsius_to_fahrenheit(params: TemperatureInput) -> TemperatureResult:
        """Convert Celsius to Fahrenheit with validation and formula info."""
        converted = celsius_to_fahrenheit_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Celsius",
            converted_value=round(converted, 2),
            converted_scale="Fahrenheit",
            formula="°F = (°C × 9/5) + 32"
        )

    @mcp.tool(
        description="Convert temperature from Fahrenheit to Celsius", 
        title="Fahrenheit to Celsius Converter"
    )
    def fahrenheit_to_celsius(params: TemperatureInput) -> TemperatureResult:
        """Convert Fahrenheit to Celsius with additional validation."""
        # Additional validation for Fahrenheit absolute zero
        if params.temperature < -459.67:  # Below absolute zero in Fahrenheit
            raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
        
        converted = fahrenheit_to_celsius_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Fahrenheit",
            converted_value=round(converted, 2),
            converted_scale="Celsius",
            formula="°C = (°F - 32) × 5/9"
        )

    @mcp.tool(
        description="Convert temperature from Celsius to Kelvin",
        title="Celsius to Kelvin Converter"
    )
    def celsius_to_kelvin(params: TemperatureInput) -> TemperatureResult:
        """Convert Celsius to Kelvin - simple offset addition."""
        converted = celsius_to_kelvin_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Celsius",
            converted_value=round(converted, 2),
            converted_scale="Kelvin",
            formula="K = °C + 273.15"
        )

    @mcp.tool(
        description="Convert temperature from Kelvin to Celsius",
        title="Kelvin to Celsius Converter"
    )
    def kelvin_to_celsius(params: TemperatureInput) -> TemperatureResult:
        """Convert Kelvin to Celsius with non-negative validation."""
        # Kelvin cannot be negative by definition
        if params.temperature < 0:
            raise ValueError("Kelvin temperature cannot be negative")
        
        converted = kelvin_to_celsius_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Kelvin",
            converted_value=round(converted, 2),
            converted_scale="Celsius",
            formula="°C = K - 273.15"
        )

    @mcp.tool(
        description="Convert temperature from Fahrenheit to Kelvin",
        title="Fahrenheit to Kelvin Converter"
    )
    def fahrenheit_to_kelvin(params: TemperatureInput) -> TemperatureResult:
        """Convert Fahrenheit to Kelvin via two-step conversion."""
        if params.temperature < -459.67:
            raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
        
        converted = fahrenheit_to_kelvin_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Fahrenheit", 
            converted_value=round(converted, 2),
            converted_scale="Kelvin",
            formula="K = (°F - 32) × 5/9 + 273.15"
        )

    @mcp.tool(
        description="Convert temperature from Kelvin to Fahrenheit",
        title="Kelvin to Fahrenheit Converter"
    )
    def kelvin_to_fahrenheit(params: TemperatureInput) -> TemperatureResult:
        """Convert Kelvin to Fahrenheit via two-step conversion."""
        if params.temperature < 0:
            raise ValueError("Kelvin temperature cannot be negative")
        
        converted = kelvin_to_fahrenheit_calc(params.temperature)
        return TemperatureResult(
            original_value=params.temperature,
            original_scale="Kelvin",
            converted_value=round(converted, 2),
            converted_scale="Fahrenheit", 
            formula="°F = (K - 273.15) × 9/5 + 32"
        )

    # Server startup with error handling
    try:
        logger.info(f"Temperature server running on {host}:{port}")
        logger.info("Available conversions: °C↔°F, °C↔K, °F↔K")
        mcp.run(transport="streamable-http")  # Use new streamable HTTP transport
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Temperature conversion server stopped")

if __name__ == "__main__":
    main()