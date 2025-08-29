"""
Main entry point for the MCPxA2AxAgentic-RAG system.
Starts both the host agent and the agentic RAG agent.
"""
import asyncio
import signal
import sys
import logging
import os
from dotenv import load_dotenv
from agents.host_agent.main import serve_host_agent
from agents.agentic_rag_agent.main import serve_agentic_rag_agent

# Set environment variables to prevent segmentation faults from threading/parallel processing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Load environment variables from .env file
load_dotenv()

# Verify critical environment variables are loaded
required_env_vars = ["GOOGLE_API_KEY", "SERPAPI_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ Missing required environment variables: {missing_vars}")
    print("   Please check your .env file and ensure all required API keys are set.")
    sys.exit(1)

print("✅ Environment variables loaded successfully")

# Configure logging to reduce noise from MCP cleanup issues (Python 3.13 compatibility)
logging.getLogger("mcp.client").setLevel(logging.ERROR)
logging.getLogger("mcp.client.stdio").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Suppress specific asyncio warnings for Python 3.13 MCP compatibility
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")
warnings.filterwarnings("ignore", message=".*cancel scope.*")

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

def exception_handler(loop, context):
    """Custom exception handler to suppress MCP/anyio cancellation errors in Python 3.13"""
    exception = context.get('exception')
    
    # Suppress known Python 3.13/anyio/MCP compatibility issues
    if isinstance(exception, RuntimeError):
        error_msg = str(exception)
        if any(phrase in error_msg for phrase in [
            "cancel scope", 
            "different task", 
            "Attempted to exit cancel scope"
        ]):
            return
    
    if isinstance(exception, asyncio.CancelledError):
        # Suppress cancellation errors during shutdown
        return
        
    # Suppress BaseExceptionGroup from anyio TaskGroup
    if hasattr(exception, '__class__') and 'BaseExceptionGroup' in str(type(exception)):
        return
        
    # For other exceptions, use default handling but suppress if it's related to MCP cleanup
    if 'mcp' in str(exception).lower() or 'anyio' in str(exception).lower():
        return
        
    # For genuine exceptions, use default handling
    loop.default_exception_handler(context)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

async def start_system():
    """Start both agents concurrently with proper error handling"""
    print("🚀 Starting MCPxA2AxAgentic-RAG System...")
    print("   Web UI will be available at:")
    print("   • Host Agent: http://localhost:10001")
    print("   • RAG Agent: http://localhost:10002")
    print("   • Streamlit UI: streamlit run app/streamlit_ui.py")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host_task = None
    rag_task = None
    
    try:
        print("Starting Host Agent on port 10001...")
        print("Starting Agentic RAG Agent on port 10002...")
        
        # Create tasks for both agents
        host_task = asyncio.create_task(serve_host_agent(host="localhost", port=10001))
        rag_task = asyncio.create_task(serve_agentic_rag_agent(host="localhost", port=10002))
        
        # Wait for both tasks to start (give them a moment)
        print("⏳ Waiting for both servers to initialize...")
        await asyncio.sleep(8)  # Give time for servers to start
        
        # Quick check if servers are responding
        import subprocess
        try:
            result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, timeout=5)
            listening_ports = [line for line in result.stdout.split('\n') if '10001' in line or '10002' in line]
            if listening_ports:
                print(f"✅ Found listening ports:")
                for port_line in listening_ports:
                    print(f"   {port_line.strip()}")
            else:
                print("⚠️ Warning: No servers found listening on ports 10001 or 10002")
        except Exception as e:
            print(f"⚠️ Could not check port status: {e}")
        
        print("✅ Both agents started successfully!")
        print("   System is now running...")
        
        # Wait for shutdown signal
        try:
            # Wait for shutdown event
            await shutdown_event.wait()
            print("\n🛑 Shutdown signal received...")
            
        except asyncio.CancelledError:
            print("\n🛑 Tasks cancelled during shutdown...")
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if host_task and not host_task.done():
            host_task.cancel()
        if rag_task and not rag_task.done():
            rag_task.cancel()
        
        # Wait for tasks to complete cleanup
        cleanup_tasks = [task for task in [host_task, rag_task] if task and not task.done()]
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except:
                pass
        
        print("🧹 System shutdown complete")

if __name__ == "__main__":
    # Set up custom exception handler for cleaner output
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(exception_handler)
    
    try:
        loop.run_until_complete(start_system())
    except KeyboardInterrupt:
        print("\n🧹 System shutdown complete")
    except Exception as e:
        print(f"❌ System failed: {e}")
        sys.exit(1)
    finally:
        # Clean shutdown with suppressed errors
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                # Wait for tasks to complete with timeout
                try:
                    loop.run_until_complete(asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True), 
                        timeout=5.0
                    ))
                except asyncio.TimeoutError:
                    print("⚠️  Some tasks took longer than expected to shutdown")
                except:
                    pass
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        try:
            loop.close()
        except Exception as e:
            print(f"⚠️  Error closing event loop: {e}")