# =============================================================================
# Civic Explorer / What Matters Here (SB)
# Unified Streamlit Application (User + Developer Testing Interface)
# =============================================================================

import os
import streamlit as st
from backend.final_database import RAGDatabase
from backend.final_agent import RAGAgent
import final_config

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="What Matters Here",
    page_icon="📖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_path" not in st.session_state:
    st.session_state.db_path = final_config.DEFAULT_DB_PATH

if "top_k" not in st.session_state:
    st.session_state.top_k = final_config.DEFAULT_TOP_K

if "database" not in st.session_state:
    st.session_state.database = None

if "civic_interests" not in st.session_state:
    st.session_state.civic_interests = []

# -----------------------------------------------------------------------------
# Sidebar – Configuration & Context
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your API key is used only for this session."
    )

    db_path = st.text_input(
        "Database Path",
        value=st.session_state.db_path,
        help="Path to your DuckDB vector database"
    )
    st.session_state.db_path = db_path

    top_k = st.slider(
        "Results per query",
        min_value=3,
        max_value=20,
        value=st.session_state.top_k
    )
    st.session_state.top_k = top_k

    model_choice = st.selectbox(
        "LLM Model",
        final_config.AVAILABLE_MODELS,
        index=0
    )

    max_iter = st.slider(
        "Max Tool Calls",
        min_value=1,
        max_value=5,
        value=final_config.DEFAULT_MAX_ITER,
        help="Maximum number of database queries per question"
    )

    st.divider()

    # Civic Context
    st.subheader("📍 Civic Context")
    city = st.text_input("City (optional):", placeholder="e.g., South Bend")
    state = st.text_input("State:", placeholder="e.g., Indiana")

    # Civic Interests
    st.subheader("🧭 What Matters to You?")
    civic_interests = st.multiselect(
        "Select issues you care about (optional):",
        [
            "Healthcare",
            "Immigration",
            "Education",
            "Housing",
            "Economic Policy",
            "Criminal Justice",
            "Environmental Policy",
            "Science & Technology",
            "Transportation",
            "National Security"
        ],
        help="We’ll use this to tailor follow-up questions and exploration prompts."
    )
    st.session_state.civic_interests = civic_interests

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("""
    ### About
    This RAG assistant helps you explore civic and political information using:
    - Verified legislative records
    - Local and state policy data
    - Neutral, evidence-based summaries
    """)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("📖 CivicMatch RAG Assistant")
st.markdown(
    "Ask questions about Indiana politics, candidates, or legislation. "
    "This space is for learning, reflection, and democratic engagement."
)

# -----------------------------------------------------------------------------
# API Key Check
# -----------------------------------------------------------------------------
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# Database Connection
# -----------------------------------------------------------------------------
if not st.session_state.database or st.session_state.database.db_path != db_path:
    st.session_state.database = RAGDatabase(db_path)

if not st.session_state.database.test_connection():
    st.error(f"❌ Database not found at: `{db_path}`")
    st.stop()
else:
    st.success(f"✅ Database connected: `{db_path}`")

# -----------------------------------------------------------------------------
# Display Chat History
# -----------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                    st.text_area(
                        f"Passage {i}",
                        source["text"],
                        height=150,
                        label_visibility="collapsed",
                        key=f"history_source_{i}_{id(message)}"
                    )
                    st.divider()

# -----------------------------------------------------------------------------
# Chat Input & Agent Execution
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask a civic or political question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating answer..."):
            try:
                agent = RAGAgent(
                    db=st.session_state.database,
                    model_name=model_choice,
                    max_iter=max_iter
                )

                contextualized_prompt = f"""
                The user is a voter in {city or 'an Indiana city'}, {state or 'Indiana'}.
                Answer the following civic question in a neutral, informative way:

                {prompt}
                """

                result = agent.ask(contextualized_prompt)

                response = result["answer"]
                sources = result["sources"]

                st.markdown(response)

                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)})"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(
                                f"**Source {i}** (Similarity: {source['similarity']:.3f})"
                            )
                            st.text_area(
                                f"Passage {i}",
                                source["text"],
                                height=150,
                                label_visibility="collapsed",
                                key=f"new_source_{i}"
                            )
                            if i < len(sources):
                                st.divider()

                # -------------------------------
                # 🔍 SMART FOLLOW-UP PROMPTS
                # -------------------------------
                st.markdown("### 🔍 Want to Explore Further?")
                follow_ups = []

                if st.session_state.civic_interests:
                    for interest in st.session_state.civic_interests[:2]:
                        follow_ups.append(
                            f"How does this representative compare to others on {interest.lower()}?"
                        )

                follow_ups.extend([
                    "What trade-offs does this position involve?",
                    "How might this issue affect my local community?"
                ])

                for q in follow_ups[:3]:
                    if st.button(q):
                        st.session_state.messages.append(
                            {"role": "user", "content": q}
                        )
                        st.rerun()

                # -------------------------------
                # 🧠 CIVIC REFLECTION
                # -------------------------------
                with st.expander("🧠 Take a Moment to Reflect (Optional)"):
                    st.write(
                        "Being civically engaged doesn’t mean having all the answers. "
                        "It means staying curious and thoughtful."
                    )

                    reflection = st.radio(
                        "After reading this, how do you feel?",
                        [
                            "More confident about where I stand",
                            "Curious but unsure",
                            "I want to compare more representatives",
                            "This challenged my assumptions"
                        ],
                        index=None
                    )

                    if reflection:
                        st.success(
                            "That’s completely valid. Civic learning is a process — "
                            "and you’re actively participating in it."
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# -----------------------------------------------------------------------------
# Example Questions
# -----------------------------------------------------------------------------
with st.expander("💡 Example Questions"):
    examples = [
        "How has Rudy Yakym voted on immigration-related bills?",
        "What are the top issues Rudy Yakym supports?",
        "How does Indiana approach Medicaid expansion?",
        "What legislation affects undocumented immigrants in Indiana?"
    ]

    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages.append(
                {"role": "user", "content": example}
            )
            st.rerun()

# -----------------------------------------------------------------------------
# Civic Action Pathways
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🗳️ Ways to Stay Civically Engaged")

st.markdown(
    "Civic engagement looks different for everyone. "
    "Here are a few meaningful ways to stay involved:"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📘 Learn**")
    st.write("- Compare representatives\n- Track voting records\n- Follow local issues")

with col2:
    st.markdown("**💬 Participate**")
    st.write("- Attend town halls\n- Ask representatives questions\n- Join community forums")

with col3:
    st.markdown("**🧾 Act**")
    st.write("- Register & vote\n- Contact representatives\n- Support local initiatives")

# -----------------------------------------------------------------------------
# Encouragement Footer
# -----------------------------------------------------------------------------
st.caption(
    "Taking time to ask questions, weigh evidence, and reflect on policy is "
    "an act of democratic responsibility. You’re doing important civic work."
)
