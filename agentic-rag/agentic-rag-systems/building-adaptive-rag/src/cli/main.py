from dotenv import load_dotenv
from src.workflow.graph import app

load_dotenv()


def format_response(result):
    """Extract response from workflow result."""
    if isinstance(result, dict) and "generation" in result:
        return result["generation"]
    elif isinstance(result, dict) and "answer" in result:
        return result["answer"]
    else:
        return str(result)


def main():
    """CLI for adaptive RAG system."""
    print("Adaptive RAG System")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                break
                
            print("Processing...")
            result = None
            for output in app.stream({"question": question}):
                for key, value in output.items():
                    result = value
                    
            if result:
                print(f"\nAnswer: {format_response(result)}")
            else:
                print("No response generated.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()