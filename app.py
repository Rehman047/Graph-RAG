"""Streamlit dashboard comparing vector retrieval with graph traversal."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from graph_rag import (
    SNIPPETS,
    TARGET_QUERY,
    LocalGraph,
    Neo4jStore,
    Triplet,
    answer_from_path,
    chroma_search,
    deterministic_triplets,
    groq_triplets,
)

load_dotenv()
st.set_page_config(page_title="Multi-Hop Detective Board", page_icon="🕵️", layout="wide")
st.title("🕵️ Multi-Hop Detective Board")
st.caption("Compare Standard Vector RAG (ChromaDB) with Graph RAG (Neo4j).")

if "triplets" not in st.session_state:
    st.session_state.triplets = [edge for snippet in SNIPPETS for edge in deterministic_triplets(snippet)]
if "query" not in st.session_state:
    st.session_state.query = TARGET_QUERY

with st.sidebar:
    st.header("Configuration")
    use_groq = st.checkbox("Use Groq for triplet extraction", value=bool(os.getenv("GROQ_API_KEY")))
    st.write("Neo4j is used when NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are configured.")
    if st.button("Extract graph facts"):
        extracted: list[Triplet] = []
        for snippet in SNIPPETS:
            try:
                extracted.extend(groq_triplets(snippet) if use_groq else deterministic_triplets(snippet))
            except Exception as exc:
                st.warning(f"Groq extraction failed for one clue; using deterministic fact ({exc}).")
                extracted.extend(deterministic_triplets(snippet))
        st.session_state.triplets = extracted
        st.success(f"Loaded {len(extracted)} relationships.")

st.subheader("Evidence")
for index, snippet in enumerate(SNIPPETS, start=1):
    st.markdown(f"**Snippet {index}:** {snippet}")

query = st.text_input("Target query", value=st.session_state.query)
left, right = st.columns(2)

with left:
    st.header("Standard Vector RAG")
    if st.button("Run ChromaDB baseline"):
        documents, backend = chroma_search(SNIPPETS, query)
        st.info(f"Backend: {backend}")
        for document in documents:
            st.markdown(f"- {document}")
        st.warning("Vector retrieval returns isolated clues; it does not explicitly traverse the missing links.")

with right:
    st.header("Neo4j Graph RAG")
    if st.button("Build and query graph", type="primary"):
        local_edges = st.session_state.triplets
        neo4j_path: list[dict[str, str]] = []
        cypher = (
            "MATCH p = shortestPath((start:Entity {name: 'Midnight Sapphire'})"
            "-[*]-(end:Entity {name: 'Miller'})) RETURN p"
        )
        try:
            store = Neo4jStore(
                os.getenv("NEO4J_URI", ""),
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", ""),
            )
            store.clear()
            store.ingest(local_edges)
            neo4j_path = store.shortest_path("Midnight Sapphire", "Miller")
            store.close()
            backend = "Neo4j"
        except Exception as exc:
            st.info(f"Neo4j unavailable ({exc}); showing the deterministic local graph.")
            backend = "Local graph fallback"
        path = neo4j_path or LocalGraph(local_edges).shortest_path("Midnight Sapphire", "Miller")
        st.caption(f"Backend: {backend}")
        st.code(cypher, language="cypher")
        if path:
            st.dataframe(
                [
                    edge if isinstance(edge, dict) else edge.as_dict()
                    for edge in path
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.success(answer_from_path(path))
        else:
            st.error("No path found. Check that the graph was ingested successfully.")
