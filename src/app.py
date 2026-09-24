"""
Placement Prep Knowledge Assistant — Streamlit UI
Interactive web application powered by LangGraph, Groq, FAISS, BM25, and Cross-Encoder Reranker.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import src.config as config
from src.agents.graph import run_query

# Configure Page
st.set_page_config(
    page_title="Placement Prep Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #6c757d; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .conflict-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .source-pill {
        display: inline-block;
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_corpus_stats():
    """Computes corpus metrics for the sidebar."""
    chunks_path = Path(config.CHUNKS_FILE)
    if not chunks_path.exists():
        return {"total_chunks": 0, "total_seniors": 0, "total_companies": 0, "companies_list": []}

    seniors, companies = set(), set()
    total = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                total += 1
                if c.get("senior_id"): seniors.add(c["senior_id"])
                if c.get("company"): companies.add(c["company"])

    return {
        "total_chunks": total,
        "total_seniors": len(seniors),
        "total_companies": len(companies),
        "companies_list": sorted(list(companies))
    }

stats = load_corpus_stats()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🎓 Placement AI")
    st.caption("Multi-Agent Placement Assistant")
    st.markdown("---")
    
    st.subheader("📊 Knowledge Base")
    col1, col2 = st.columns(2)
    col1.metric("Chunks", stats["total_chunks"])
    col2.metric("Seniors", stats["total_seniors"])
    st.metric("Company Logs", stats["total_companies"])

    st.markdown("---")
    st.subheader("🎯 Search Filters")
    selected_topics = st.multiselect("Filter by Topics:", options=config.TOPICS, default=[])
    selected_company = st.selectbox("Filter by Company:", options=["All Companies"] + stats["companies_list"])
    company_filter = None if selected_company == "All Companies" else selected_company

    st.markdown("---")
    with st.expander("ℹ️ AI Architecture"):
        st.markdown("""
        - **Router:** Classifies query intent
        - **Hybrid Retrieval:** BM25 (keywords) + FAISS (vectors) via RRF
        - **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
        - **Synthesizer:** Groq LLM with Conflict Detection
        """)

# ==================== MAIN PAGE ====================
st.markdown('<div class="main-header">🎓 Placement Prep Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query senior placement notes, interview experiences, CTC logs, and roadmaps.</div>', unsafe_allow_html=True)

# Example Buttons
st.markdown("**💡 Suggested Questions:**")
cols = st.columns(3)
if "query_input" not in st.session_state:
    st.session_state["query_input"] = "What was the interview format and test pattern for Deloitte?"

if cols[0].button("🏢 Deloitte Interview Format", use_container_width=True):
    st.session_state["query_input"] = "What was the interview format and test pattern for Deloitte?"
if cols[1].button("🤖 InfoEdge ML Test", use_container_width=True):
    st.session_state["query_input"] = "What was the test pattern and interview experience for InfoEdge Data Scientist?"
if cols[2].button("⚖️ DSA vs SQL Priority", use_container_width=True):
    st.session_state["query_input"] = "How should I prioritize DSA, SQL, and Aptitude for campus placement prep?"

user_query = st.text_input("Ask a placement question:", value=st.session_state["query_input"])

if st.button("🚀 Ask Assistant", type="primary", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Retrieving and synthesizing senior advice..."):
            result = run_query(
                user_query=user_query,
                topic_filters=selected_topics if selected_topics else None,
                company_filter=company_filter
            )

            # Show Conflict Alert if seniors give differing advice
            if result.get("has_conflict"):
                st.markdown("""
                <div class="conflict-box">
                    <strong>⚡ Divergent Senior Perspectives Detected:</strong><br>
                    Retrieved notes offer differing guidance on this topic. Both perspectives are detailed below.
                </div>
                """, unsafe_allow_html=True)

            # Final Answer
            st.markdown("### 💡 Guidance & Analysis")
            st.markdown(result.get("final_answer", ""))

            # Sources
            st.markdown("---")
            st.markdown("#### 📚 Verified Sources Cited")
            for src in result.get("sources_used", []):
                st.markdown(f'<span class="source-pill">📌 {src}</span>', unsafe_allow_html=True)

            # Retrieved Evidence Chunks
            with st.expander(f"🔍 View {len(result.get('retrieved_chunks', []))} Evidence Chunks"):
                for idx, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                    comp = f" | Company: **{chunk.get('company')}**" if chunk.get("company") else ""
                    st.markdown(f"**Chunk #{idx}** — Contributor: **{chunk.get('senior_id', 'Senior')}** ({chunk.get('source_name')}){comp}")
                    st.code(chunk.get("text", "")[:400] + "...", language="markdown")
