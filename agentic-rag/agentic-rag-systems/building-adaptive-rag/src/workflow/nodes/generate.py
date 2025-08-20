from typing import Any, Dict
from src.workflow.chains.generation import generation_chain
from src.workflow.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using documents and question."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
