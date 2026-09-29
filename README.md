# Multi-Hop Detective Board

A Streamlit comparison of standard vector RAG (ChromaDB) and graph RAG
(Neo4j) over a deliberately fragmented mystery. The graph path connects
**Midnight Sapphire -> Victor -> Red Viper -> abandoned cannery -> Miller**,
demonstrating the multi-hop context that similarity retrieval does not
explicitly preserve.

The five required clues are hardcoded in `graph_rag.py`. Groq extraction is
available through the official Groq client, and Neo4j persistence uses the
official Neo4j driver plus Cypher. The `langchain` and `langchain-neo4j`
packages are included for the required integration stack.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` for Neo4j Aura or a
local Neo4j server. Set `GROQ_API_KEY` to enable LLM triplet extraction. The
default Groq model is `llama-3.3-70b-versatile`.

## Offline-safe behavior

The app remains usable without credentials or network services. Curated
deterministic triplets replace Groq, an in-memory lexical baseline replaces
ChromaDB, and a local breadth-first graph replaces Neo4j. The UI labels every
fallback. Relationship types are sanitized before interpolation into Cypher;
entity values remain parameters.

## Expected comparison

Use the default query, **"How is the Midnight Sapphire connected to Miller?"**
The left panel shows similarity-matched clues. The right panel shows the
shortest-path Cypher, traversed relationships, and a factual path narrative.

## Tests

```bash
pytest -q
```
