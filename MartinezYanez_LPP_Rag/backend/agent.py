# =============================================================================
# Agent module for RAG Assistant
# =============================================================================
# This file creates an AI agent that can DECIDE when to search the database.
# Instead of always retrieving passages, the LLM chooses when retrieval helps.
# =============================================================================

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from backend.database import RAGDatabase

class RAGAgent: #dummy variables, adjust to personal prooject later on
    def __init__(self, db: RAGDatabase, model_name: str, max_iter: int):
        self.db = db
        self.model_name = model_name
        self.max_iter = max_iter
        self.last_sources = []  # We'll store retrieved passages here for the UI

    def create_tool(self):
        # ---------------------------------------------------------------------
        # The @tool decorator transforms this function into something the
        # LLM can call. The docstring is CRUCIAL—it's what the LLM reads
        # to decide whether and how to use this tool.
        # ---------------------------------------------------------------------
        @tool("Query RAG Database")
        def query_rag_db(query: str) -> str:
            """Search the vector database containing customized texts.
            
            Args:
                query: Search query about topic.
                
            Returns:
                Relevant passages from the database
            """
            try:
                results = self.db.query(query)
                
                if results:
                    # Store sources for UI display
                    self.last_sources.extend(results)
                    
                    # Format passages for the LLM to read
                    passages = [row["text"] for row in results]
                    return "\n\n---\n\n".join([f"Passage {i+1}:\n{doc}" for i, doc in enumerate(passages)])
                else:
                    return "No relevant passages found."
                    
            except Exception as e:
                return f"Error querying database: {str(e)}"
        
        return query_rag_db

    # TO DO: Update the ask() function
    def ask(self, question: str) -> dict:
        """
        Ask a question to the agent.
        
        Returns:
            Dictionary with 'answer' and 'sources'.
        """
        # Reset sources for this query
        self.last_sources = []
        
        # TO DO: Create the LLM instance
        llm = LLM(model = self.model_name)

        # TO DO: Call the database tool (e.g. the function above)
        query_tool = self.create_tool()
        

        #agent = Agent( #BIG PART OF FINAL, UPDATE ROLE, GOAL, BACKSTORY TO CUSTOMIZE PROJECT. BE SPECIFIC TO WHAT YOURE WORKING ON
         #   role='TOPIC Content Assistant',
          #  goal='Answer questions about TOPIC using the database',
           # backstory='You are an expert who has access to a database with content about the TOPIC.',
            #tools=[query_tool],
            #llm=llm,
            #verbose=True, # Shows what the agent is doing
            #allow_delegation=False, # Does not create sub-agents
            #max_iter=self.max_iter # Limits tool calls
        #)
        agent = Agent(
            role="Civic Legislative Analyst",
            goal=(
                "Analyze verified legislative records and explain what they reveal "
                "about a politician’s support, opposition, or involvement with specific policies."
            ),
            backstory=(
                "You are a non-partisan civic assistant designed to help voters understand"
                "how elected officials’ actions align with public policy values."
                "Your goal is not to persuade, but to inform clearly and transparently."
                "You analyze verified legislative records and explain what they reveal"
                "about a politician’s priorities, support, or opposition to specific policies."
                "You understand that:"
                "- Introducing or co-sponsoring legislation indicates active support"
                "- Voting records indicate explicit positions"
                "- Legislative actions can be interpreted even if documents do not use identical wording"
                "You translate congressional language into plain, accessible explanations"
                "focused on what the actions mean for everyday people."
                "When information is sufficient, you make a clear conclusion."
                "When evidence is mixed or limited, you explain the uncertainty honestly."
                "IMPORTANT BEHAVIOR RULES:"
                "- You may query the RAG database at most ONCE per user question."
                "- If sources are retrieved, you MUST produce a final answer."
                "- You are NOT allowed to respond with “I need more information” if relevant sources exist."
                "- Your task is to synthesize, not to continue researching indefinitely."
            ),
            tools=[query_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=self.max_iter
        )
        
        # TO DO: Create the task
        #task = Task(
         #   description = question,
          #  agent = agent,
           # expected_output="A comprehensive, simplified answer based on the data personalized to the individual's needs to advance their civic engagement based on their interests and values"
        #)
        task = Task(
            description=question, 
            agent=agent,
            expected_output=(
                "Provide your response in the following format:"
                "Conclusion:"
                "A clear, direct answer to the question."
                "Evidence:"
                "A brief summary of the most relevant legislative actions, citing patterns"
                "such as bill sponsorship, introduction, or committee referral."
                "Why This Matters:"
                "A short explanation of how these actions relate to policy priorities"
                "or public impact, written for a general audience."
                "Follow-Up:"
                "Offer 1–2 optional, neutral follow-up questions the user may want to explore next."

                #"A clear, direct civic answer that synthesizes the retrieved legislative records. "
                #"If a politician introduced or co-sponsored a bill, explicitly state that this "
                #"indicates support. Cite evidence from the retrieved sources and explain reasoning "
                #"in plain language."
            )
        )

        
        # TO DO: Create the Crew and run it
        crew = Crew(agents = [agent],
                    tasks = [task],
                    verbose = True,
                    max_rpm = 20)
        
        result = crew.kickoff()
        
        # Returns the answer and sources
        return {
            "answer": str(result),
            "sources": self.last_sources.copy()
        }
