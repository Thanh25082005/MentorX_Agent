# AI Academic Advisor Agent

An AI Agent for academic/training support built with **Groq**, **LangChain**, **RAG**, **ReAct-style reasoning**, and **FastAPI**.

This project started as a custom Python implementation and is being **refactored toward LangChain-based orchestration** to make the codebase cleaner, easier to extend, and easier to maintain.

---

## Overview

This system is an **Autonomous AI Agent Architecture** for handling user questions related to academic/training information. It combines:

- a central **LLM Brain** powered by **Groq**
- **short-term memory** for recent conversation context
- **long-term memory** through **RAG**
- a **ReAct-style loop** for reasoning and action
- a **tool layer** for structured lookup, calculation, and web search

The main idea is simple:

- simple questions can be answered directly
- internal knowledge questions go through **RAG**
- action-oriented or multi-step questions go through the **ReAct loop** and may call tools

This matches the architecture where the Brain coordinates memory, retrieval, and tools rather than doing everything from raw prompting alone.

---

## Main Goals

- Answer natural language questions accurately
- Keep recent conversation context
- Retrieve knowledge from internal documents
- Use tools when needed
- Support modular migration from custom logic to **LangChain**
- Stay easy to run locally while remaining production-friendly

---

## Architecture

### Core Components

#### 1. LLM Brain
The Brain is the central decision-maker of the system.

Responsibilities:
- understand user intent
- decide whether to answer directly, use RAG, or call tools
- run reasoning steps
- synthesize the final response

**LLM provider:** Groq

Groq is used here as the main inference layer because the architecture relies on multiple reasoning/action steps, and low latency is especially valuable for ReAct-style execution. :contentReference[oaicite:2]{index=2}

---

#### 2. Short-Term Memory
Short-term memory stores recent chat history so the agent can keep context across turns.

Typical use cases:
- resolving pronouns like “that course”
- remembering the previous question
- keeping the conversation coherent

Recommended implementation:
- MVP: in-memory buffer
- production-ready path: Redis-based memory

The architecture document also highlights short-term memory as session-oriented, fast, and suitable for sliding-window storage. :contentReference[oaicite:3]{index=3}

---

#### 3. Long-Term Memory (RAG)
Long-term memory is implemented through a Retrieval-Augmented Generation pipeline.

It typically includes:
- **Document Store** for source files
- **Embedding Model**
- **Vector Database** for semantic search

Typical knowledge sources:
- syllabus
- policies
- wiki pages
- internal documents

RAG is used when the answer should come from trusted internal knowledge rather than the model’s parametric memory. :contentReference[oaicite:4]{index=4}

---

#### 4. ReAct Reasoning Loop
The system uses a ReAct-style loop:

- **Thought**: reason about what to do next
- **Action**: call a tool or retrieval step
- **Observation**: read the result
- repeat if needed
- generate final answer

This loop is especially useful for multi-step tasks such as:
- looking up course information
- combining retrieved knowledge with calculations
- deciding whether another tool call is needed

---

#### 5. Tool Layer
The agent can interact with external capabilities through tools.

Current tool set:
- **Local Search** on `courses.csv`
- **Calculator**
- **Web Search**

This allows the system to go beyond pure text generation and operate on structured data or external information. :contentReference[oaicite:5]{index=5}

---

## High-Level Flow

1. User sends a query
2. The Brain loads recent chat history from short-term memory
3. The Brain analyzes the intent
4. The system chooses one of these paths:
   - direct answer
   - RAG retrieval
   - ReAct loop with tool usage
5. Retrieved context / tool results are returned to the Brain
6. The Brain composes the final answer
7. The conversation is stored back into short-term memory
8. The API returns the response to the user

This flow reflects the architecture where the Brain coordinates memory, retrieval, and tool observations end to end. :contentReference[oaicite:6]{index=6}

---

## Tech Stack

### Current / Target Stack

- **Python**
- **FastAPI**
- **Groq API**
- **LangChain**
- **FAISS** or **ChromaDB**
- **pandas**
- **Pydantic**
- **python-dotenv**

### Suggested future production upgrades

- **Redis** for short-term memory and caching
- **PostgreSQL** for structured course data and workflow/checkpoint state
- **Qdrant** for production vector search
- **Tavily** for web search
- **LangGraph** for explicit graph-based agent orchestration
- **Langfuse** for tracing and observability
- **Llama-Guard / guardrails** for safer inputs and outputs

The architecture notes also recommend guardrails, caching, observability, and more production-grade storage layers as the next step beyond the MVP. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

---

## Project Structure

```text
app/
├── api/
│   └── routes.py
├── agent/
│   ├── orchestrator.py
│   └── react_loop.py
├── core/
│   ├── config.py
│   └── groq_client.py
├── memory/
│   └── short_term.py
├── models/
│   └── schemas.py
├── rag/
│   ├── ingest.py
│   └── retriever.py
├── tools/
│   ├── base.py
│   ├── calculator.py
│   ├── course_search.py
│   └── web_search.py
└── main.py

data/
├── courses.csv
└── docs/

tests/