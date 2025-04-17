# PocketWisdom AI – YouTube-Powered Educational Assistant


## Project Overview
PocketWisdom AI is a RAG-powered educational chatbot that helps parents explain school topics to their children in fun, kid-friendly language. It retrieves knowledge from YouTube transcripts, finds relevant diagrams, and presents simplified educational responses — all driven by GPT-4o-mini.

This project was developed as part of the Ironhack final capstone and uses LangChain, LangSmith, and vector search to build a truly helpful educational assistant.

## Features
Agent-Powered Tool Usage:

🧾 Document Search: Retrieves relevant YouTube transcript chunks

🔢 Calculator: Performs step-by-step numeric reasoning

🖼️ Image Search: Auto-fetches diagrams or visuals related to the topic (no fallbacks used)

GPT-4o-mini: Fast, accurate, and cost-effective reasoning and explanation

LangSmith Evaluation: Validates output accuracy, hallucination, and relevance

Streamlit App: Friendly local interface for natural language chat

## Project Structure
PocketWisdomAI/
<pre> ```bash PocketWisdomAI/ ├── faiss_youtube_index/ │ ├── index.faiss │ └── index.pkl │ ├── app/ │ ├── app.py │ └── faiss_youtube_index/ │ ├── youtube_transcripts/ │ ├── Art.txt │ ├── Biology.txt │ ├── Chemistry.txt │ ├── Computer_Science.txt │ ├── Geography.txt │ ├── History.txt │ └── Physics.txt │ ├── Evaluation.ipynb ├── VectotStore+Agent&tools.ipynb ├── YouTube_decipher.ipynb ├── requirements.txt └── .env ``` </pre>


## Technologies Used
LangChain + Agents

OpenAI GPT-4o-mini

FAISS – Vector DB for transcript retrieval

DuckDuckGo Search – Safe and simple image fetch

Streamlit – Local chat app interface

LangSmith – Evaluation with LLM-based graders

## Prerequisites
Python 3.10+

Install all required Python packages with: (pip install -r requirements.txt)

## Deployment
This application runs locally using Streamlit.
To launch the app: (streamlit run app.py)

## License
This project is licensed under the MIT License — see the LICENSE file for full details.

## Evaluation Metrics (via LangSmith)
rag-answer-vs-reference	  - Compares the AI answer to a reference answer for factual correctness
rag-answer-hallucination	- Checks if the AI answer is grounded in the retrieved documents
rag-document-relevance	  - Verifies that the retrieved documents are relevant to the question

 ##Acknowledgments
  Huge thanks to my teachers for their support and guidance, encouragement, and support throughout this journey.
  Big appreciation to my Ironhack classmates for their feedback and collaboration.
  Acknowledgment is extended to Ironhack for the structure, tools, and opportunity.

