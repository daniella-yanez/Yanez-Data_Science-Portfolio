# CivicMatch
*A friendly, local-first guide to civic life*

## Overview

CivicMatch is a conversational civic engagement application designed to help people explore local politics, public issues, and democratic participation—especially those who may have little to no prior experience with civic engagement.

Rather than assuming political knowledge or strong partisan interest, CivicMatch meets users where they are. It encourages curiosity, critical thinking, and confidence by providing structured, evidence-based answers in a supportive and approachable tone.

The project currently focuses on a **pilot deployment for South Bend / St. Joseph County, Indiana**, allowing for small-scale evaluation of how conversational tools can increase civic awareness and sustained engagement.

---

## Project Goals

CivicMatch is built around three core goals:

1. **Lower the barrier to civic participation**  
   Make civic information accessible to users who feel intimidated, disengaged, or unsure where to start.

2. **Encourage informed and reflective exploration**  
   Help users compare representatives, policies, and issues without telling them what to think.

3. **Support democratic habits, not just election moments**  
   Promote ongoing engagement such as learning about local issues, understanding representatives’ roles, and exploring ways to participate beyond voting.

---

## Key Features

- 🧠 **Conversational Civic Assistant**  
  Users ask natural-language questions about local politics, legislation, and representatives.

- 📚 **Retrieval-Augmented Generation (RAG)**  
  Responses are grounded in verified legislative and civic source material using semantic search.

- 🗺️ **Local Context Awareness**  
  Answers can be tailored by city, county, and state to emphasize relevance.

- 🧭 **Interest-Guided Exploration**  
  Users can select issue areas they care about (e.g., healthcare, immigration, education), which shapes follow-up prompts and exploration paths.

- 🤝 **Supportive & Nonjudgmental Tone**  
  The assistant remains neutral while acknowledging users’ values, including when users express strong partisan leanings.

- 🔍 **Source Transparency**  
  Every answer includes accessible source passages so users can verify and explore further.

---

## Why This Project Exists

Many civic tools assume users already care deeply about politics or understand how government works. CivicMatch challenges that assumption.

This project is especially designed for:
- First-time voters  
- Young adults and students  
- Community members who feel disconnected from politics  
- Users who want information without pressure or partisanship  

CivicMatch treats civic engagement as a **skill that can be learned**, not a prerequisite.

---

## Technology Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM Integration:** OpenAI API  
- **Vector Database:** DuckDB (embeddings-based semantic search)  
- **Architecture:** Retrieval-Augmented Generation (RAG)

---

## How It Works

1. User submits a civic question through a chat interface.
2. The system retrieves relevant legislative or civic documents from a local vector database.
3. The language model generates a response grounded in those sources.
4. Sources are displayed transparently for user review.
5. The assistant encourages further exploration based on user interests and local context.

---

## Pilot Scope

The current pilot focuses on:
- **South Bend / St. Joseph County, Indiana**
- Testing engagement among users with low initial civic interest
- Evaluating whether conversational design increases:
  - Time spent exploring civic topics
  - Willingness to ask follow-up questions
  - Understanding of local representatives and issues

The architecture is intentionally designed to scale to other locations and election cycles.

---

## Ethical Commitments

- Nonpartisan and evidence-based responses  
- Transparency in sources  
- Respect for user values and uncertainty  
- No persuasion or voter manipulation  

CivicMatch does **not** tell users who to support—it helps them decide for themselves.

---

## Future Directions

Planned expansions include:
- Local event and meeting discovery
- “Ways to get involved” prompts tailored to user interests
- Representative comparison tools
- Civic learning paths for beginners
- Deployment beyond Indiana

---

## Getting Started

1. Clone the repository
2. Install dependencies
3. Add your OpenAI API key
4. Connect a DuckDB vector database
5. Run the Streamlit app

(Setup details may vary depending on data source configuration.)

---

## Author & Motivation

CivicMatch was created as an exploratory civic technology project focused on democratic access, inclusion, and education at the local level.

It is motivated by the belief that democracy works best when people feel informed, welcomed, and empowered to participate—regardless of where they start.

---

## License

This project is for educational and research purposes. Licensing details to be added.
