# RAG Orchestrator - Multi-Agent Retrieval-Augmented Generation System
# 
# This module implements a sophisticated RAG system using multiple specialized agents:
# - PDFLoaderAgent: Handles PDF ingestion and text chunking
# - EmbeddingAgent: Manages text embeddings and FAISS vector indexing
# - RetrievalAgent: Performs semantic search with diversity sampling
# - QAAgent: Generates answers using retrieved context
# - RankingAgent: Evaluates and ranks multiple answer candidates
# - WebSearchAgent: Integrates with MCP servers for web search
# - RAGOrchestrator: Coordinates all agents for end-to-end RAG workflows
#
# The system supports both document-based retrieval and dynamic web search integration
# through MCP (Model Context Protocol) servers.

import os
import json
import faiss
import tiktoken
import requests
from dotenv import load_dotenv
from typing import List, Tuple
from PyPDF2 import PdfReader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from model import get_llm_model, get_embedding_model, get_llm_model_instance, get_embed_model_instance

load_dotenv()

# Use the centralized model configuration
ENC = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """Calculate the number of tokens in a text string."""
    return len(ENC.encode(text))

# Global shared RAG orchestrator instance
_shared_rag_orchestrator = None

def get_shared_rag_orchestrator():
    """Get the shared RAG orchestrator instance (singleton pattern)."""
    global _shared_rag_orchestrator
    if _shared_rag_orchestrator is None:
        _shared_rag_orchestrator = RAGOrchestrator()
        print("[RAGOrchestrator] Initialized shared instance")
    return _shared_rag_orchestrator

class PDFLoaderAgent:
    """Handles PDF document loading and text chunking for optimal retrieval."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self, path: str) -> List[str]:
        """Load PDF and split into tokenized chunks with overlap."""
        print(f"[PDFLoaderAgent] Loading and splitting PDF: {path}")
        try:
            print(f"[PDFLoaderAgent] Attempting to read PDF file: {path}")
            reader = PdfReader(path)
            print(f"[PDFLoaderAgent] PDF loaded successfully, extracting text from {len(reader.pages)} pages")
            
            full_text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                    print(f"[PDFLoaderAgent] Extracted text from page {i+1}/{len(reader.pages)}")
                except Exception as page_error:
                    print(f"[PDFLoaderAgent] Warning: Could not extract text from page {i+1}: {page_error}")
                    continue
            
            if not full_text.strip():
                raise Exception("No text could be extracted from the PDF")
                
            print(f"[PDFLoaderAgent] Total text length: {len(full_text)} characters")
            tokens = ENC.encode(full_text)
            print(f"[PDFLoaderAgent] Encoded text into {len(tokens)} tokens")
            
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk = ENC.decode(tokens[start:end])
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append(chunk)
                start += self.chunk_size - self.chunk_overlap
                
            print(f"[PDFLoaderAgent] Total chunks created: {len(chunks)}")
            return chunks
            
        except Exception as e:
            print(f"[PDFLoaderAgent] Error processing PDF: {e}")
            print(f"[PDFLoaderAgent] Exception type: {type(e).__name__}")
            raise Exception(f"Failed to process PDF file: {e}")

class EmbeddingAgent:
    """Manages text embeddings and FAISS vector indexing for semantic search."""
    
    def __init__(self):
        # Use the centralized embedding model (lazy loading)
        self.embedding_model = None  # Initialize lazily
        self.dim = None  # Will be set dynamically based on the model
        self.index = None  # Will be created after we know the dimension
    
    def _get_embedding_model(self):
        """Get the embedding model, creating it if needed."""
        if self.embedding_model is None:
            try:
                print("[EmbeddingAgent] Initializing embedding model...")
                self.embedding_model = get_embed_model_instance()
                print("[EmbeddingAgent] Embedding model loaded successfully")
                
                # Determine the embedding dimension by testing with a sample text
                if self.dim is None:
                    print("[EmbeddingAgent] Determining embedding dimensions...")
                    try:
                        # Use a simple test text to avoid any complex tokenization issues
                        test_embedding = self.embedding_model.embed_documents(["test"])
                        self.dim = len(test_embedding[0])
                        print(f"[EmbeddingAgent] Detected embedding dimension: {self.dim}")
                        
                        # Create FAISS index with the correct dimension
                        print(f"[EmbeddingAgent] Creating FAISS index with dimension {self.dim}")
                        self.index = faiss.IndexFlatL2(self.dim)
                        print(f"[EmbeddingAgent] Successfully created FAISS index with dimension {self.dim}")
                        
                    except Exception as embed_error:
                        print(f"[EmbeddingAgent] Error during dimension detection: {embed_error}")
                        print("[EmbeddingAgent] Falling back to safe assumption of 384 dimensions")
                        self.dim = 384
                        self.index = faiss.IndexFlatL2(self.dim)
                        
            except Exception as e:
                print(f"[EmbeddingAgent] Critical error loading embedding model: {e}")
                print(f"[EmbeddingAgent] Exception type: {type(e).__name__}")
                print(f"[EmbeddingAgent] Traceback: {e}")
                raise Exception(f"Failed to initialize embedding model: {e}")
        return self.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks."""
        print(f"[EmbeddingAgent] Creating embeddings for {len(texts)} texts using HuggingFace")
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 1  # Ultra-safe batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"[EmbeddingAgent] Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                batch_embeddings = self._get_embedding_model().embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
            print(f"[EmbeddingAgent] Created embeddings with dimension {len(all_embeddings[0])}")
            return all_embeddings
            
        except Exception as e:
            print(f"[EmbeddingAgent] Error during embedding generation: {e}")
            print(f"[EmbeddingAgent] Exception type: {type(e).__name__}")
            raise Exception(f"Failed to generate embeddings: {e}")

    def add_to_index(self, texts: List[str]):
        """Add text embeddings to the FAISS index for retrieval."""
        print(f"[EmbeddingAgent] Adding {len(texts)} embeddings to index")
        try:
            # Ensure the embedding model and index are initialized
            self._get_embedding_model()
            
            embs = self.embed(texts)
            vecs = np.array(embs, dtype="float32")
            
            # Double-check index is initialized
            if self.index is None:
                raise Exception("FAISS index not initialized. This should not happen.")
                
            print(f"[EmbeddingAgent] Adding {len(vecs)} vectors to FAISS index")
            self.index.add(vecs)
            print(f"[EmbeddingAgent] Added embeddings to index. Total vectors now: {self.index.ntotal}")
            
        except Exception as e:
            print(f"[EmbeddingAgent] Error adding embeddings to index: {e}")
            print(f"[EmbeddingAgent] Exception type: {type(e).__name__}")
            raise Exception(f"Failed to add embeddings to index: {e}")

class RetrievalAgent:
    """Performs semantic search with diversity sampling for candidate generation."""
    
    def __init__(self, index: faiss.IndexFlatL2, embedding_agent: EmbeddingAgent):
        self.index = index
        self.embedding_agent = embedding_agent

    def retrieve_candidates(self, query: str, texts: List[str], n_candidates: int = 3, k: int = 5) -> List[List[str]]:
        """Retrieve multiple diverse candidate sets for improved answer quality."""
        print(f"[RetrievalAgent] Retrieving {n_candidates} sets of top {k} chunks for query: {query}")
        
        # Use HuggingFace embeddings instead of OpenAI
        base_emb = self.embedding_agent.embed([query])[0]
        candidates = []
        
        for i in range(n_candidates):
            # Add small perturbation for diversity
            perturbed_emb = np.array(base_emb, dtype="float32") + np.random.normal(0, 0.01, len(base_emb))
            D, I = self.index.search(np.array([perturbed_emb], dtype="float32"), k)
            retrieved = [texts[j] for j in I[0] if j < len(texts)]
            candidates.append(retrieved)
            
        print(f"[RetrievalAgent] Created {len(candidates)} candidate sets")
        return candidates

class QAAgent:
    """Generates answers using retrieved context and LLM models."""
    
    def __init__(self, temperature: float = 0.2):
        # Use the centralized LLM model (lazy loading)
        self.temperature = temperature
        self.llm = None  # Initialize lazily
    
    def _get_llm(self):
        """Get the LLM model, creating it if needed."""
        if self.llm is None:
            self.llm = get_llm_model_instance(temperature=self.temperature)
        return self.llm

    def answer(self, question: str, context: List[str]) -> str:
        """Generate a single answer using provided context."""
        print(f"[QAAgent] Answering question with Gemini")
        context_str = '\n---\n'.join(context)
        prompt = (
            "You are an expert assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )
        
        try:
            response = self._get_llm().invoke(prompt)
            answer = response.content.strip()
            print(f"[QAAgent] Received answer of length {len(answer)}")
            return answer
        except Exception as e:
            print(f"[QAAgent] Error with Gemini: {e}")
            return f"Error generating answer: {str(e)}"

    def answer_parallel(self, question: str, candidate_contexts: List[List[str]]) -> List[str]:
        """Generate answers in parallel for multiple context sets."""
        print(f"[QAAgent] Generating answers in parallel for {len(candidate_contexts)} candidates.")
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.answer, question, ctx) for ctx in candidate_contexts]
            for fut in futures:
                results.append(fut.result())
        return results

class RankingAgent:
    """Evaluates and ranks multiple answer candidates using LLM-based scoring."""
    
    def __init__(self, temperature: float = 0.2):
        # Use the centralized LLM model (lazy loading)
        self.temperature = temperature
        self.llm = None  # Initialize lazily
    
    def _get_llm(self):
        """Get the LLM model, creating it if needed."""
        if self.llm is None:
            self.llm = get_llm_model_instance(temperature=self.temperature)
        return self.llm

    def rank(self, question: str, candidate_answers: List[str], candidate_contexts: List[List[str]]) -> Tuple[str, int]:
        """Rank candidates and return the best answer with its index."""
        print("[RankingAgent] Ranking candidates with Gemini self-eval.")
        
        ranking_prompt = f'''You are an expert assistant judging a RAG system. Given several candidate answers (each with their retrieval context) to the same question, first select the single most accurate/supportable candidate, then explain briefly why you chose it.

Output exactly this format:
Candidate #N
Reason: <reason>

Best Answer:
<full text>

Question: {question}
'''
        
        summary = ""
        for idx, (ctx, ans) in enumerate(zip(candidate_contexts, candidate_answers), 1):
            ctx_part = "\n".join(ctx)
            summary += f"\nCandidate #{idx}:\nContext:\n{ctx_part}\nAnswer:\n{ans}\n"
        
        full_prompt = ranking_prompt + summary
        
        try:
            response = self._get_llm().invoke(full_prompt)
            response_text = response.content.strip()
            
            print("\n[RankingAgent] Gemini Decision and Reason:\n" + response_text)
            
            import re
            m = re.search(r"Candidate #(\d+)\s*\nReason:([^\n]*)\n+Best Answer:\n(.+)", response_text, re.DOTALL)
            if m:
                cand_idx = int(m.group(1)) - 1
                reason = m.group(2).strip()
                answer = m.group(3).strip()
                print(f"[RankingAgent] Selected candidate #{cand_idx+1}.")
                print(f"[RankingAgent] Reasoning: {reason}")
            else:
                cand_idx = 0
                answer = candidate_answers[0]
                print("[RankingAgent] Could not parse ranking output, returning first candidate.")
            
            return answer, cand_idx
            
        except Exception as e:
            print(f"[RankingAgent] Error with Gemini ranking: {e}")
            return candidate_answers[0], 0

class WebSearchAgent:
    """Integrates with MCP servers to provide web search capabilities."""
    
    def __init__(self, endpoint="http://localhost:8000/", apikey=None):
        self.endpoint = endpoint
        self.apikey = apikey

    def search_web(self, query: str) -> str:
        """Query MCP server for web search results."""
        print(f"[WebSearchAgent] Querying MCP server for web results: {query}")
        try:
            search_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "search",
                "params": {"query": query}
            }
            resp = requests.post(self.endpoint, json=search_request)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('result') or data.get('results') or str(data)
            print("[WebSearchAgent] Web result from MCP server received")
            return str(results)
        except Exception as e:
            print(f"[WebSearchAgent] MCP server error: {e}")
            return "[WebSearchAgent] Failed to obtain web data."

class RAGOrchestrator:
    """Main orchestrator that coordinates all RAG agents for end-to-end workflows."""
    
    def __init__(self, n_candidates: int = 3, k: int = 5, mcp_endpoint: str = None, mcp_apikey: str = None):
        print("[RAGOrchestrator] Initializing agents with centralized models")
        self.loader = PDFLoaderAgent()
        self.embedder = EmbeddingAgent()
        self.text_chunks: List[str] = []
        self.retriever: RetrievalAgent = None
        self.qa = QAAgent()
        self.ranker = RankingAgent()
        self.n_candidates = n_candidates
        self.k = k
        # Init Web agent
        self.web_agent = WebSearchAgent(
            endpoint=mcp_endpoint or os.getenv('MCP_ENDPOINT', 'http://localhost:8000'),
            apikey=mcp_apikey or os.getenv('MCP_API_KEY')
        )

    def ingest(self, pdf_path: str):
        """Process and index a PDF document for retrieval."""
        print(f"[RAGOrchestrator] Ingesting PDF: {pdf_path}")
        self.text_chunks = self.loader.load_and_split(pdf_path)
        self.embedder.add_to_index(self.text_chunks)
        self.retriever = RetrievalAgent(self.embedder.index, self.embedder)
        print(f"[RAGOrchestrator] Ingestion complete with {len(self.text_chunks)} chunks")

    def query(self, question: str) -> str:
        """Process a query using document retrieval and optional web search."""
        print(f"[RAGOrchestrator] Querying for question: {question}")
        
        # Get document context first if available
        doc_context = ""
        if self.retriever:
            print(f"[RAGOrchestrator] Retrieving document context")
            doc_context = self._search_documents_with_ranking(question)
        
        # Use Gemini to decide if web search is needed
        decision_prompt = f"""
        Given the following document context and question, determine if web search is needed.
        
        Document Context:
        {doc_context if doc_context else "No documents available"}
        
        Question: {question}
        
        If the document context fully answers the question, respond with "SUFFICIENT".
        If additional current information or web search is needed, respond with "SEARCH_NEEDED: <search_query>".
        """
        
        try:
            decision_llm = get_llm_model_instance(temperature=0.1)
            decision_response = decision_llm.invoke(decision_prompt)
            decision = decision_response.content.strip()
            
            if decision.startswith("SEARCH_NEEDED:"):
                search_query = decision.split("SEARCH_NEEDED:", 1)[1].strip()
                print(f"[RAGOrchestrator] Gemini decided to search web: {search_query}")
                web_result = self.web_agent.search_web(search_query)
                web_context = self._process_mcp_response(web_result)
                
                # Generate final answer with both contexts
                final_prompt = f'''Answer the question using both document and web contexts.

Document Context:
{doc_context}

Web Context:
{web_context}

Question: {question}
Answer:'''
                
                final_llm = get_llm_model_instance(temperature=0.2)
                final_response = final_llm.invoke(final_prompt)
                return final_response.content.strip()
            else:
                print(f"[RAGOrchestrator] Document context is sufficient")
                return doc_context
        
        except Exception as e:
            print(f"[RAGOrchestrator] Error: {e}")
            return doc_context if doc_context else "Error processing query"

    def _search_documents_with_ranking(self, query: str) -> str:
        """Perform document search with multi-candidate ranking and selection."""
        print(f"[RAGOrchestrator] Starting ranked document search for: {query}")
        
        # Step 1: Retrieve multiple candidate contexts
        candidate_contexts = self.retriever.retrieve_candidates(
            query, self.text_chunks, 
            n_candidates=self.n_candidates, 
            k=self.k
        )
        
        # Step 2: Generate answers for each candidate in parallel
        candidate_answers = self.qa.answer_parallel(query, candidate_contexts)
        
        # Step 3: Rank candidates and select best answer
        final_answer, chosen_idx = self.ranker.rank(query, candidate_answers, candidate_contexts)
        
        print(f"[RAGOrchestrator] Selected answer from candidate #{chosen_idx+1}")
        return final_answer

    def _process_mcp_response(self, mcp_response: str) -> str:
        """Extract relevant content from MCP server responses."""
        try:
            if isinstance(mcp_response, str):
                try:
                    data = json.loads(mcp_response)
                except json.JSONDecodeError:
                    return mcp_response
            else:
                data = mcp_response
            
            if isinstance(data, dict):
                content = (data.get('result') or 
                          data.get('results') or 
                          data.get('answer') or 
                          data.get('content') or
                          data.get('text'))
                if content:
                    return str(content)
            
            return str(data)
        except Exception as e:
            print(f"[_process_mcp_response] Error: {e}")
            return str(mcp_response)