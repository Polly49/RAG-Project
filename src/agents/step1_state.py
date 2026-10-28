"""
Step 1: LangGraph Shared State Schema
Defines the central data contract for the autonomous, self-reflective multi-agent system.
"""

from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    # 1. Inputs (Immutable from user or session)
    user_query: str                          # Original question from student
    topic_filters: Optional[List[str]]       # Optional topic filters (e.g. ['dsa', 'sql'])
    company_filter: Optional[str]            # Optional company filter (e.g. 'Deloitte')
    
    # 2. Router Decisions
    router_decision: str                     # 'IN_SCOPE' vs 'OUT_OF_SCOPE'
    needs_synthesis: bool                    # True if multi-source strategy or comparison question
    is_out_of_scope: bool                    # True for conversational/greeting or non-placement queries
    
    # 3. Retrieval & Self-Reflection State (Cyclic Loop Channels)
    expanded_queries: List[str]              # Structured query variations generated for hybrid retrieval
    retrieved_chunks: List[Dict[str, Any]]   # Top reranked context chunks from retrieval pipeline
    is_relevant: bool                        # Set by Retrieval Grader: True if chunks satisfy query
    retry_count: int                         # Tracks self-correction loop iterations (capped at 1)
    reformulated_query: Optional[str]        # Autonomous query rewrite generated upon low-relevance
    
    # 4. Synthesizer & Reflection Output
    final_answer: str                        # Grounded answer synthesized from retrieved sources
    has_conflict: bool                       # True if senior opinions/guidance diverge
    sources_used: List[str]                  # Cited senior notes and guides
