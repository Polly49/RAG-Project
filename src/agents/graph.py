"""
LangGraph Multi-Agent Orchestrator
Connects 3 agents in a stateful pipeline:
START -> Router Agent -> Retrieval Agent -> Synthesizer Agent -> END
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, START, END
from src.agents.state import GraphState
from src.agents.router_agent import router_node
from src.agents.retrieval_agent import retrieval_node
from src.agents.synthesizer_agent import synthesizer_node

def create_agent_graph():
    """Builds and compiles the multi-agent graph."""
    workflow = StateGraph(GraphState)

    # 1. Add Agent Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Connect the Flow
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retrieval")
    workflow.add_edge("retrieval", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

# Compile the pipeline once for reuse
pipeline = create_agent_graph()

def run_query(user_query: str, topic_filters=None, company_filter=None) -> dict:
    """Entrypoint to run a user query through the full agent pipeline."""
    initial_state: GraphState = {
        "user_query": user_query,
        "topic_filters": topic_filters,
        "company_filter": company_filter,
        "expanded_queries": [],
        "needs_synthesis": False,
        "retrieved_chunks": [],
        "final_answer": "",
        "has_conflict": False,
        "sources_used": []
    }
    return pipeline.invoke(initial_state)

if __name__ == "__main__":
    result = run_query("What is the test pattern for Deloitte?")
    print(result.get("final_answer"))
