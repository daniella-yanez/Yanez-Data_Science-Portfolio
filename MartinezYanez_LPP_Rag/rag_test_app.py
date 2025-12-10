"""
Simple Streamlit app to test CivicMatch RAG Agent connection.
This is a developer/testing interface.
"""

import os
import streamlit as st
from backend.database import RAGDatabase
from backend.agent import RAGAgent
import config

# -----------------------------------------------------------------------------
# UI Header
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CivicMatch RAG Test", layout="centered")
st.title("🧪 CivicMatch RAG Agent Test")

st.write(
    "This tool verifies that the RAG agent can retrieve and explain "
    "verified civic and political information from the database."
)

# -----------------------------------------------------------------------------
# API Key Input
# -----------------------------------------------------------------------------
api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    help="Your API key is used only for this session."
)

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# Database Check
# -----------------------------------------------------------------------------
db = RAGDatabase(config.DEFAULT_DB_PATH)

if not db.test_connection():
    st.error("❌ Database connection failed. Check your DuckDB file.")
    st.stop()

st.success("✅ Database connected")

# -----------------------------------------------------------------------------
# Initialize Agent
# -----------------------------------------------------------------------------
agent = RAGAgent(
    db=db,
    model_name=config.DEFAULT_MODEL,
    max_iter=config.DEFAULT_MAX_ITER
)

# -----------------------------------------------------------------------------
# Civic Context Inputs (NEW)
# -----------------------------------------------------------------------------
st.subheader("📍 Civic Context")

city = st.text_input("City (optional):", placeholder="e.g., South Bend")
state = st.text_input("State:", placeholder="e.g., Indiana")

# -----------------------------------------------------------------------------
# Question Input
# -----------------------------------------------------------------------------
st.subheader("💬 Test a Civic Question")

question = st.text_input(
    "Ask a question about civic data in the database:",
    placeholder="Example: How has Rudy Yakym voted on immigration-related bills?"
)

if not question:
    st.info("Enter a question to test the agent.")
    st.stop()

# -----------------------------------------------------------------------------
# Run Agent
# -----------------------------------------------------------------------------
with st.spinner("Querying civic database..."):
    try:
        # Add civic and geographic context to the prompt
        contextualized_question = f"""
        The user is a voter in {city or 'an Indiana city'}, {state}.
        Answer the following civic question in a neutral and informative way:

        {question}
        """

        result = agent.ask(contextualized_question)

        st.success("✅ Agent response received")

        # Display answer
        st.markdown("### ✅ Answer")
        st.write(result["answer"])

        # Display sources (if any)
        if result["sources"]:
            st.markdown("### 📚 Sources Retrieved")
            for i, source in enumerate(result["sources"], start=1):
                st.write(f"**Source {i}:**")
                st.caption(source["text"][:400] + "...")

    except Exception as e:
        st.error(f"❌ Agent failed: {e}")
