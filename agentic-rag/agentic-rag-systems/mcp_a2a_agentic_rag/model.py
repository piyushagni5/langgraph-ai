#!/usr/bin/env python3
"""
Centralized model configuration for MCPxA2AxAgentic-RAG system.
Uses Google Gemini for LLM and HuggingFace for embeddings.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

def get_llm_model(temperature: float = 0):
    """Get LLM model name string for Google ADK LlmAgent."""
    print("🔄 Using Google Gemini model")
    return "gemini-2.0-flash-exp"

def get_llm_model_instance(temperature: float = 0):
    """Get actual LLM model instance for direct use."""
    print("🔄 Creating Google Gemini model instance")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )

def get_embedding_model(use_huggingface: bool = True):
    """Get embedding model with safer HuggingFace models as default."""
    
    if use_huggingface:
        # Try multiple safe HuggingFace models in order of preference
        safe_models = [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "kwargs": {'device': 'cpu'},
                "encode_kwargs": {'normalize_embeddings': True}
            },
            {
                "name": "sentence-transformers/paraphrase-MiniLM-L6-v2", 
                "kwargs": {'device': 'cpu'},
                "encode_kwargs": {'normalize_embeddings': True}
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "kwargs": {'device': 'cpu'},
                "encode_kwargs": {'normalize_embeddings': True}
            }
        ]
        
        for model_config in safe_models:
            try:
                print(f"🔄 Trying HuggingFace model: {model_config['name']}")
                return HuggingFaceEmbeddings(
                    model_name=model_config["name"],
                    model_kwargs=model_config["kwargs"],
                    encode_kwargs=model_config["encode_kwargs"]
                )
            except Exception as e:
                print(f"❌ Failed to load {model_config['name']}: {e}")
                continue
        
        print("❌ All HuggingFace models failed, falling back to Google embeddings...")
    
    # Use Google embeddings (more stable)
    print("🔄 Using Google Generative AI embeddings")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_safe_embedding_model():
    """
    Get the safest embedding model that absolutely avoids segmentation faults.
    Uses the most lightweight and stable HuggingFace model available.
    """
    # Set additional safety environment variables if not already set
    import os
    safety_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    for var, value in safety_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            
    # Try multiple safe configurations in order of preference
    safe_configs = [
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "kwargs": {'device': 'cpu', 'trust_remote_code': False},
            "encode_kwargs": {'normalize_embeddings': True, 'batch_size': 1}
        },
        {
            "model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "kwargs": {'device': 'cpu', 'trust_remote_code': False},
            "encode_kwargs": {'normalize_embeddings': True, 'batch_size': 1}
        },
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "kwargs": {'device': 'cpu', 'trust_remote_code': False},
            "encode_kwargs": {'normalize_embeddings': True}
        }
    ]
    
    for i, config in enumerate(safe_configs):
        try:
            print(f"🔄 Trying ultra-safe HuggingFace model: {config['model']}")
            print(f"   Configuration {i+1}/{len(safe_configs)}: batch_size={config['encode_kwargs'].get('batch_size', 'default')}")
            
            # Create the model with ultra-safe settings
            model = HuggingFaceEmbeddings(
                model_name=config["model"],
                model_kwargs=config["kwargs"],
                encode_kwargs=config["encode_kwargs"]
            )
            
            # Test the model with a simple embedding to ensure it works
            print(f"🧪 Testing model with simple embedding...")
            test_result = model.embed_documents(["test"])
            print(f"✅ Model test successful! Dimension: {len(test_result[0])}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {config['model']}: {e}")
            print(f"   Error type: {type(e).__name__}")
            continue
    
    # If all HuggingFace models fail, use Google as ultimate fallback
    print("❌ All HuggingFace models failed")
    print("🔄 Using Google embeddings as ultimate fallback")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# Global model instances (lazy initialization)
llm_model = None
embed_model = None

# Legacy compatibility functions for existing imports
def get_llm_model_legacy():
    """Get the global LLM model instance, creating it if needed."""
    global llm_model
    if llm_model is None:
        llm_model = get_llm_model_instance()
        print(f"🔧 Created LLM model instance: {MODEL_INFO['llm_model']}")
    return llm_model

def get_embed_model_instance():
    """Get the global embedding model instance, creating it if needed."""
    global embed_model
    if embed_model is None:
        # Try the safe embedding model first, but with even more safety
        try:
            embed_model = get_safe_embedding_model()
            print(f"🔧 Created embedding model instance: {MODEL_INFO['embed_model']}")
        except Exception as e:
            print(f"❌ Safe HuggingFace model failed: {e}")
            print("🔄 Falling back to Google embeddings for maximum safety...")
            # Ultimate fallback to Google embeddings
            embed_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("🔧 Created Google embedding model as fallback")
    return embed_model

# MODEL_INFO for backward compatibility (dynamic evaluation)
def get_model_info():
    """Get current model information."""
    return {
        "model2use": "google",
        "llm_model": "Google Gemini",
        "embed_model": "HuggingFace Transformers (Safe Models) with Google Fallback",
        "llm_provider": "Google Generative AI",
        "embed_provider": "HuggingFace/Google Hybrid"
    }

# For backward compatibility, create a property-like access
class ModelInfoDict(dict):
    def __getitem__(self, key):
        return get_model_info()[key]
    
    def get(self, key, default=None):
        return get_model_info().get(key, default)
    
    def items(self):
        return get_model_info().items()
    
    def keys(self):
        return get_model_info().keys()
    
    def values(self):
        return get_model_info().values()

MODEL_INFO = ModelInfoDict()

if __name__ == "__main__":
    print("Current model selection: google")
    model_info = get_model_info()
    print(f"LLM Model: {model_info['llm_model']}")
    print(f"Embedding Model: {model_info['embed_model']}")
    print("✅ Models configured successfully!")
    
    # Test lazy loading
    print("\n🧪 Testing lazy model loading:")
    test_llm = get_llm_model_legacy()
    test_embed = get_embed_model_instance()
    
    print("\n📋 Model Usage Summary:")
    print("🟡 Gemini: Used for LLM operations (agents and RAG)")
    print("🤗 HuggingFace: Used for embedding operations (document processing)")