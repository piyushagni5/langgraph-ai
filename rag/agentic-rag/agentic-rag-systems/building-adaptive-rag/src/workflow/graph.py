from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.router import RouteQuery, question_router
from src.workflow.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_documents import grade_documents
from src.workflow.nodes.retrieve import retrieve
from src.workflow.nodes.web_search import web_search
from src.workflow.state import GraphState

load_dotenv()


def decide_to_generate(state):
    """Route to web search or generation."""
    print("---ASSESS DOCUMENTS---")
    return WEBSEARCH if state["web_search"] else GENERATE


def grade_generation_grounded_in_documents_and_question(state):
    """Grade answer quality and groundedness."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if grounded
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    
    if score.binary_score:
        # Check if useful
        score = answer_grader.invoke({"question": question, "generation": generation})
        return "useful" if score.binary_score else "not useful"
    else:
        return "not supported"


def route_question(state: GraphState) -> str:
    """Route question to vectorstore or web search."""
    print("---ROUTE QUESTION---")
    source: RouteQuery = question_router.invoke({"question": state["question"]})
    return WEBSEARCH if source.datasource == WEBSEARCH else RETRIEVE


# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
)
workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()

# Export graph visualization
app.get_graph().draw_mermaid_png(output_file_path="graph.png")