# =============================================================================
# Agent module for RAG Assistant
# =============================================================================
# This agent decides when retrieval is necessary and synthesizes retrieved
# sources to explain civic, legislative, and policy positions at both
# local and national levels.
# =============================================================================

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from backend.final_database import RAGDatabase

class RAGAgent:
    def __init__(self, db: RAGDatabase, model_name: str, max_iter: int):
        self.db = db
        self.model_name = model_name
        self.max_iter = max_iter
        self.last_sources = []  # Retrieved passages for UI display

    def create_tool(self):
        @tool("Query RAG Database")
        def query_rag_db(query: str) -> str:
            """
            Search the vector database containing verified civic, legislative,
            policy, and political texts at the local, state, and federal levels.

            Use this tool when answering questions about:
            - Politicians (local, state, or national)
            - Policy positions or issue stances
            - Legislative behavior or voting records
            - Public statements, reports, or civic documents
            - Public or economic sentiment reflected in records

            Args:
                query: A search query describing the political figure, policy,
                       issue area, or sentiment of interest.

            Returns:
                Relevant passages from the database.
            """
            try:
                results = self.db.query(query)

                if results:
                    self.last_sources.extend(results)
                    passages = [row["text"] for row in results]
                    return "\n\n---\n\n".join(
                        [f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)]
                    )
                else:
                    return "No relevant passages found."

            except Exception as e:
                return f"Error querying database: {str(e)}"

        return query_rag_db

    def ask(self, question: str) -> dict:
        """
        Ask a question to the agent.

        The agent may retrieve from the database at most once and must
        synthesize any retrieved sources into a final answer.

        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        self.last_sources = []

        llm = LLM(model=self.model_name)
        query_tool = self.create_tool()

        agent = Agent(
            role="Civic & Policy Intelligence Analyst",
            goal=(
                "Analyze verified civic and legislative records to explain where "
                "politicians—especially local and state officials—stand on major "
                "policy issues and public sentiment."
            ),
            backstory=(
                "You are a non-partisan civic assistant designed to help voters "
                "understand how political actors’ actions and statements align "
                "with policy priorities and public concerns.\n\n"

                "You are capable of analyzing BOTH:\n"
                "- Formal legislative actions (bills, votes, sponsorships)\n"
                "- Informal or contextual evidence (statements, reports, summaries)\n\n"

                "ISSUE AREAS YOU CAN ANALYZE INCLUDE:\n"
                "- Social Policy\n"
                "- Healthcare\n"
                "- Immigration\n"
                "- Electoral Policy\n"
                "- Domestic Policy\n"
                "- Criminal Justice\n"
                "- Education\n"
                "- Foreign Policy\n"
                "- Science & Technology\n"
                "- Housing\n"
                "- National Security\n"
                "- Economic Policy\n"
                "- Environmental Policy\n"
                "- Transportation\n\n"

                "SENTIMENT ANALYSIS CAPABILITIES:\n"
                "- National sentiment\n"
                "- Economic sentiment\n"
                "- Consumer sentiment\n"
                "- Personal finance sentiment\n"
                "- Science & technology sentiment\n\n"

                "You infer sentiment ONLY when supported by the tone, framing, "
                "or implications of the retrieved sources. If sentiment is not "
                "clearly supported, you explicitly say so.\n\n"

                "INTERPRETATION RULES:\n"
                "- Bill sponsorship or introduction = active support\n"
                "- Voting records = explicit position\n"
                "- Repeated patterns across documents strengthen conclusions\n"
                "- Absence of evidence must be acknowledged clearly\n\n"

                "IMPORTANT BEHAVIOR RULES:\n"
                "- You may query the RAG database at most ONCE per user question\n"
                "- If sources are retrieved, you MUST produce a final answer\n"
                "- You may not ask for more information if relevant sources exist\n"
                "- Your task is synthesis, not endless research\n"
                "- Your tone must remain neutral, explanatory, and accessible"
            ),
            tools=[query_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=self.max_iter
        )

        task = Task(
            description=question,
            agent=agent,
            expected_output=(
                "Respond in a structured but friendly, conversational tone that feels like "
                "a knowledgeable civic assistant talking with a curious voter.\n\n"

                "Use the following format, but write naturally and warmly:\n\n"

                "Conclusion:\n"
                "Give a clear, direct answer to the user’s question in plain language. "
                "Avoid jargon where possible. Speak confidently but accessibly.\n\n"

                "Evidence:\n"
                "Summarize the most relevant legislative actions, public statements, or "
                "documented behaviors from the sources. Point out meaningful patterns "
                "(such as repeated sponsorships, consistent votes, or issue focus) "
                "and briefly explain what they signal.\n\n"

                "Why This Matters:\n"
                "Explain how this information connects to real-world impacts—how it might "
                "affect everyday life, community priorities, or long-term policy outcomes. "
                "Frame this in a way that helps the user reflect on whether this representative’s "
                "actions align with what *they* care about.\n\n"

                "Encouragement & Perspective:\n"
                "Acknowledge that political preferences are often shaped by personal values, "
                "experiences, and community context. If the user appears to lean strongly "
                "toward one partisan perspective, respond with respect and understanding—"
                "never judgment. Emphasize that exploring evidence strengthens, rather than "
                "undermines, informed civic engagement.\n\n"

                "Explore Further:\n"
                "Offer 1–2 thoughtful, neutral follow-up questions tailored to what the user "
                "seems most interested in (such as a specific issue area or value). "
                "Encourage the user to compare representatives or examine trade-offs, helping "
                "them think critically about which elected officials best meet their needs "
                "and priorities.\n\n"

                "Tone Guidelines:\n"
                "- Be supportive, calm, and curious—not authoritative or preachy\n"
                "- Assume the user is engaging in good faith\n"
                "- Encourage learning and reflection, not persuasion\n"
                "- Make the user feel comfortable continuing the conversation"
            )
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            max_rpm=20
        )

        result = crew.kickoff()

        return {
            "answer": str(result),
            "sources": self.last_sources.copy()
        }
