# Terminal Server Docker Configuration
# Containerized MCP terminal server for secure command execution

FROM python:3.11-slim

# Set application working directory
WORKDIR /app

# Create workspace directory for command execution isolation
RUN mkdir -p /workspace

# Copy server implementation
COPY . /app

# Install Python dependencies without cache to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Expose default MCP server port
EXPOSE 5000

# Start terminal server
CMD ["python", "terminal_server.py"]