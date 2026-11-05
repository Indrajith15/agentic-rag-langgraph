# RAG + Agentic RAG using LangChain and LangGraph

A hands-on project exploring Retrieval-Augmented Generation (RAG) and Agentic RAG using Python, Gemini, LangChain, ChromaDB and LangGraph.

The project was built while learning how LLM applications retrieve external knowledge, evaluate information and generate context-grounded responses.

## Project Overview

This project implements two approaches:

### 1. Basic RAG

A traditional RAG pipeline that retrieves relevant information from a knowledge base and uses it to generate answers.

**Workflow:**

1. Load documents.
2. Split documents into smaller chunks.
3. Generate embeddings.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using similarity search.
6. Construct context.
7. Generate an answer using Gemini.

### 2. Agentic RAG

An agent-based RAG workflow built using LangGraph.

Unlike the basic RAG pipeline, the agent can make decisions about how to handle a question and evaluate retrieved documents.

**Workflow:**

1. Analyze the user's question.
2. Decide whether retrieval is necessary.
3. Retrieve relevant documents.
4. Grade documents for relevance.
5. Retry retrieval if relevant documents are not found (maximum two attempts).
6. Generate an answer using relevant context.
7. Return a fallback response if sufficient information is unavailable.

General questions are routed directly to the LLM without retrieval.

## Technologies Used

- Python
- Google Gemini API
- LangChain
- LangGraph
- ChromaDB
- Vector Embeddings
- Python-dotenv

## Project Structure

```text
rag_agentic_project/
│
├── document/
│   └── cloud_computing.txt
│
├── main.py
├── ingestion.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retrieval.py
├── rag_pipeline.py
├── agentic_rag.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag_agentic_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Get an API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the project

Basic RAG:

```bash
python rag_pipeline.py
```

Agentic RAG:

```bash
python agentic_rag.py
```

## Key Concepts Learned

- Document ingestion and preprocessing
- Recursive text splitting
- Embeddings and vector representations
- Vector databases
- Semantic similarity search
- Context construction
- Retrieval-Augmented Generation
- LangChain integrations
- LangGraph nodes, edges and state
- Conditional routing
- Document relevance grading
- Retry mechanisms and fallback handling
- Grounded LLM responses

## Learning Objective

The objective of this project was to understand how RAG works internally and how agentic workflows can extend traditional RAG using LangGraph.

This project was developed as a practical learning exercise in Generative AI and Agentic AI.

---

**Built with Python while learning Generative AI.**