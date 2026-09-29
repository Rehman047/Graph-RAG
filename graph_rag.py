"""Core data and retrieval logic for the Neo4j Graph RAG demonstration."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SNIPPETS: tuple[str, ...] = (
    "The Midnight Sapphire necklace was stolen from the museum display.",
    "Security cameras show a man named Victor fleeing the museum with a heavy bag.",
    "Victor is known to be a low-level associate of the Red Viper smuggling ring.",
    "The Red Viper ring uses the abandoned cannery on the docks as their primary stash house.",
    "A harbor patrolman named Miller reported seeing a speedboat leave the abandoned cannery at 3:00 AM.",
)
TARGET_QUERY = "How is the Midnight Sapphire connected to Miller?"


@dataclass(frozen=True)
class Triplet:
    head: str
    relation: str
    tail: str
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"head": self.head, "relation": self.relation, "tail": self.tail}


def sanitize_relationship(value: str) -> str:
    """Make an LLM relation safe to interpolate as a Cypher type."""
    token = re.sub(r"[^A-Za-z0-9_]", "_", value.strip().upper())
    token = re.sub(r"_+", "_", token).strip("_")
    return token if token and token[0].isalpha() else "RELATED_TO"


def deterministic_triplets(snippet: str) -> list[Triplet]:
    """Return the curated graph facts for the supplied mystery clue."""
    facts = {
        SNIPPETS[0]: ("Midnight Sapphire", "STOLEN_BY", "Victor"),
        SNIPPETS[1]: ("Victor", "FLED_FROM", "Museum"),
        SNIPPETS[2]: ("Victor", "ASSOCIATE_OF", "Red Viper"),
        SNIPPETS[3]: ("Red Viper", "USES_AS_STASH_HOUSE", "Abandoned Cannery"),
        SNIPPETS[4]: ("Miller", "SAW_SPEEDBOAT_LEAVE", "Abandoned Cannery"),
    }
    fact = facts.get(snippet)
    return [Triplet(*fact, source=snippet)] if fact else []


def normalize_triplets(payload: Any, source: str = "") -> list[Triplet]:
    """Validate Groq JSON output and accept head/relation/tail aliases."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        payload = payload.get("triplets", payload.get("triples", payload.get("data", [])))
    if not isinstance(payload, list):
        raise ValueError("Triplet response must be a JSON list")
    result: list[Triplet] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        values = {str(key).lower(): value for key, value in item.items()}
        head = values.get("head", values.get("subject"))
        relation = values.get("relation", values.get("predicate"))
        tail = values.get("tail", values.get("object"))
        if all(isinstance(value, str) and value.strip() for value in (head, relation, tail)):
            result.append(Triplet(head.strip(), sanitize_relationship(relation), tail.strip(), source))
    return result


def groq_triplets(text: str, api_key: str | None = None, model: str | None = None) -> list[Triplet]:
    """Extract one or more graph edges using Groq's JSON response mode."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The groq package is not installed") from exc
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    'Extract factual graph edges. Return only JSON in this shape: '
                    '{"triplets":[{"head":"Entity","relation":"RELATION","tail":"Entity"}]}.'
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return normalize_triplets(content, source=text)


class LexicalBaseline:
    """Small dependency-free fallback for ChromaDB retrieval."""

    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def search(self, query: str, k: int = 3) -> list[str]:
        query_tokens = self._tokens(query)
        scored = [
            (len(query_tokens & self._tokens(document)), index, document)
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [document for score, _, document in scored[:k] if score] or self.documents[:k]


def chroma_search(documents: Sequence[str], query: str, k: int = 3) -> tuple[list[str], str]:
    """Search ChromaDB when available, otherwise use lexical overlap."""
    try:
        import chromadb

        collection = chromadb.Client().get_or_create_collection("mystery_snippets")
        collection.upsert(ids=[str(i) for i in range(len(documents))], documents=list(documents))
        result = collection.query(query_texts=[query], n_results=k)
        return result.get("documents", [[]])[0], "ChromaDB"
    except Exception:
        return LexicalBaseline(documents).search(query, k), "In-memory lexical fallback"


class LocalGraph:
    """Deterministic graph fallback used when Neo4j is not configured."""

    def __init__(self, triplets: Iterable[Triplet]):
        self.triplets = list(triplets)

    def shortest_path(self, start: str, end: str) -> list[Triplet]:
        adjacency: dict[str, list[Triplet]] = {}
        for edge in self.triplets:
            adjacency.setdefault(edge.head.lower(), []).append(edge)
            adjacency.setdefault(edge.tail.lower(), []).append(
                Triplet(edge.tail, edge.relation, edge.head, edge.source)
            )
        queue: deque[tuple[str, list[Triplet]]] = deque([(start.lower(), [])])
        visited = {start.lower()}
        while queue:
            node, path = queue.popleft()
            if node == end.lower():
                return path
            for edge in adjacency.get(node, []):
                next_node = edge.tail.lower() if edge.head.lower() == node else edge.head.lower()
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [edge]))
        return []


class Neo4jStore:
    """Neo4j persistence through the official driver, with safe relation types."""

    def __init__(self, uri: str, username: str, password: str):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("The neo4j package is not installed") from exc
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self._driver.close()

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def ingest(self, triplets: Iterable[Triplet]) -> int:
        count = 0
        with self._driver.session() as session:
            for triplet in triplets:
                relation = sanitize_relationship(triplet.relation)
                query = (
                    "MERGE (h:Entity {name: $head}) "
                    "MERGE (t:Entity {name: $tail}) "
                    f"MERGE (h)-[:{relation}]->(t)"
                )
                session.run(query, head=triplet.head, tail=triplet.tail).consume()
                count += 1
        return count

    def shortest_path(self, start: str, end: str) -> list[dict[str, str]]:
        query = """
        MATCH p = shortestPath((start:Entity {name: $start})-[*]-(end:Entity {name: $end}))
        UNWIND relationships(p) AS rel
        RETURN startNode(rel).name AS head, type(rel) AS relation,
               endNode(rel).name AS tail
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, start=start, end=end)]


def answer_from_path(path: Sequence[Triplet | dict[str, str]]) -> str:
    if not path:
        return "No connected path was found."
    edges = [
        edge if isinstance(edge, Triplet) else Triplet(edge["head"], edge["relation"], edge["tail"])
        for edge in path
    ]
    chain = edges[0].head
    for edge in edges:
        chain += f" -[{edge.relation}]-> {edge.tail}"
    return f"The evidence connects the Midnight Sapphire to Miller through this path: {chain}."
