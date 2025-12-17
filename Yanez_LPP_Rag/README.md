# CivicMatch
*A local-first, conversational civic engagement platform*
* Streamlit Link to Application: http://10.24.88.73:8501

---

## 1. Project Overview

CivicMatch is a conversational civic engagement application designed to help individuals explore local politics, public policy, and democratic participation—especially those with little to no prior experience in civic engagement.

The project focuses on lowering informational and psychological barriers to participation by presenting verified civic information in a friendly, nonjudgmental, and structured conversational format. Rather than assuming political knowledge or partisan alignment, CivicMatch encourages curiosity, reflection, and informed decision-making.

The current pilot deployment focuses on **South Bend / St. Joseph County, Indiana**, allowing the project to evaluate how effective a small-scale, locally grounded tool can be at increasing civic interest and sustained engagement.

---

## 2. Project Goals

CivicMatch is built around four core goals:

1. **Accessibility**  
   Make civic information understandable and welcoming for first-time or disengaged users.

2. **Informed Exploration**  
   Provide evidence-based explanations of representatives, policies, and issues without persuasion.

3. **Critical Civic Thinking**  
   Encourage users to reflect on how policies and representatives align with their own values.

4. **Sustained Democratic Engagement**  
   Support ongoing civic participation beyond single elections or political moments.

---

## 3. System Architecture

CivicMatch uses a **Retrieval-Augmented Generation (RAG)** architecture to ensure that responses are both conversational and grounded in verified source material.

### High-Level Flow

1. A user submits a civic or political question through the Streamlit interface.
2. The agent evaluates whether retrieving source material will improve the answer.
3. If retrieval is helpful, the agent queries a DuckDB-based vector database using semantic search.
4. Relevant passages are returned and provided to the language model.
5. The model synthesizes a structured, plain-language response.
6. Retrieved sources are displayed transparently for user review.

This approach balances accessibility with factual reliability while maintaining transparency.

---

## 4. Agent Design & Configuration Rationale

The CivicMatch agent is intentionally designed with specific constraints to support ethical and effective civic engagement.

### Key Design Decisions

- **Limited Retrieval Calls**  
  The agent is restricted in how often it can query the database to prioritize synthesis over endless searching.

- **Structured Yet Conversational Output**  
  Responses follow a consistent structure (Conclusion, Evidence, Why This Matters, Follow-Up) while maintaining a supportive and approachable tone.

- **Non-Partisan and Non-Persuasive Behavior**  
  The agent explains legislative actions and policy positions without endorsing candidates or parties.

- **Encouragement Without Judgment**  
  When users express strong partisan preferences, the agent acknowledges those views respectfully while encouraging critical comparison and reflection.

These choices reflect the project’s goal of increasing civic confidence rather than directing political outcomes.

---

## 5. Core Features

- **Conversational Civic Assistant** for natural-language questions
- **Verified Source Grounding** using semantic retrieval
- **Local Context Awareness** (city, county, and state)
- **Interest-Guided Exploration** based on user-selected issues
- **Transparent Source Display** for trust and verification
- **Supportive, Beginner-Friendly Tone** designed for civic newcomers

---

## 6. Pilot Scope

The initial pilot focuses on residents of **South Bend / St. Joseph County, Indiana**.

The pilot evaluates:
- User engagement over time
- Willingness to explore follow-up questions
- Increased understanding of local representatives and policies
- Comfort level among users with low initial civic interest

The system is designed to scale to other localities and election cycles.

---

## 7. Technology Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **LLM Integration:** OpenAI API
- **Vector Database:** DuckDB (embeddings-based semantic search)
- **Architecture Pattern:** Retrieval-Augmented Generation (RAG)

---

## 8. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key

### Installation

```bash
git clone https://github.com/daniella-yanez/Yanez-Data_Science-Portfolio.git
cd Yanez-Data_Science-Portfolio/Yanez_LPP_Rag
pip install -r requirements.txt
```

###Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

###Running the Application
```bash
streamlit run final_app.py
```


### Database Configuration

CivicMatch uses a DuckDB vector database containing pre-embedded civic and legislative documents.  
The database path can be configured either through the Streamlit sidebar at runtime or by updating the `config.py` file.

---

## 9. Ethical Commitments

CivicMatch is guided by the following principles:

- **Evidence-based responses** grounded in verified legislative and civic sources  
- **Transparency** in sourcing, reasoning, and limitations  
- **Respect for user values and uncertainty**, including differing political perspectives  
- **No political persuasion or targeted messaging**

The application is designed to inform and support civic understanding—not to influence political choices.

---

## 10. Future Development

Planned expansions include:

- Broader geographic coverage beyond the initial pilot region  
- Civic action and participation prompts tailored to user interests  
- Representative and policy comparison tools  
- Guided learning paths for first-time civic participants  
- Integration of local events, meetings, and community engagement opportunities  

---

## 11. License & Use

This project is developed for educational and research purposes.  
Licensing details will be added in future releases.

---

## 12. Author & Motivation

CivicMatch was created as a civic technology research project focused on democratic access, inclusion, and local engagement.

The project is grounded in the belief that democracy functions best when people feel informed, welcomed, and empowered to participate—regardless of where they begin.

