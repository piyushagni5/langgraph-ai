#!/bin/bash

# Script to kill all running services for MCPxA2AxAgentic-RAG system

echo "Killing all MCPxA2AxAgentic-RAG services..."

# Kill processes on specific ports
echo "Killing processes on ports 10001, 10002..."
for port in 10001 10002; do
    if lsof -ti :$port >/dev/null 2>&1; then
        echo "Killing process on port $port"
        lsof -ti :$port | xargs kill -9 2>/dev/null
    fi
done

# Kill main.py processes
echo "Killing main.py processes..."
pkill -f "python.*main.py" 2>/dev/null

# Kill streamlit processes (if running)
echo "Killing streamlit processes..."
pkill -f "streamlit" 2>/dev/null

# Kill web search server processes
echo "Killing web search server processes..."
pkill -f "web_search_server.py" 2>/dev/null

echo "All services killed!"

# Verify ports are free
echo "Checking if ports are now free..."
if ! lsof -i :10001 -i :10002 >/dev/null 2>&1; then
    echo "✅ Ports 10001 and 10002 are now free!"
else
    echo "⚠️  Some processes may still be running:"
    lsof -i :10001 -i :10002
fi
