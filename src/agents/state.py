"""
LangGraph Shared State Schema
Defines the data contract shared across all 3 agent nodes in the pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    user_query: str                          # Original question from student
    topic_filters: Optional[List[str]]       # Optional topic filters (e.g. ['dsa', 'sql'])
    company_filter: Optional[str]            # Optional company filter (e.g. 'Deloitte')
    
    needs_synthesis: bool                    # Set by Router: True if multi-source strategy question
    retrieved_chunks: List[Dict[str, Any]]   # Set by Retriever: Top reranked context chunks
    final_answer: str                        # Set by Synthesizer: Final grounded answer
    has_conflict: bool                       # Set by Synthesizer: True if senior opinions diverge
    sources_used: List[str]                  # Set by Synthesizer: List of cited senior notes
