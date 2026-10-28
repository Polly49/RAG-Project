"""
Placement Prep RAG - Agents Package
Ordered 5-Step Multi-Agent Orchestration:
1. step1_state.py             -> LangGraph shared GraphState TypedDict
2. step2_router_agent.py      -> Query intent classification & filter extraction
3. step3_retrieval_agent.py   -> Retrieval node connecting retrieval pipeline
4. step4_synthesizer_agent.py -> Grounded synthesis & conflict detection
5. step5_graph.py             -> Compiled LangGraph StateGraph pipeline
"""

from src.agents.step1_state import GraphState
from src.agents.step2_router_agent import router_node
from src.agents.step3_retrieval_agent import retrieval_node
from src.agents.step4_synthesizer_agent import synthesizer_node
from src.agents.step5_graph import create_agent_graph, run_query, pipeline
