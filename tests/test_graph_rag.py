from graph_rag import (
    SNIPPETS,
    LocalGraph,
    answer_from_path,
    deterministic_triplets,
    normalize_triplets,
    sanitize_relationship,
)


def test_required_five_clue_chain_is_deterministic():
    edges = [edge for snippet in SNIPPETS for edge in deterministic_triplets(snippet)]
    path = LocalGraph(edges).shortest_path("Midnight Sapphire", "Miller")
    assert len(path) == 4
    assert path[0].head == "Midnight Sapphire"
    assert path[-1].tail == "Miller"


def test_normalize_triplets_and_safe_relationship():
    result = normalize_triplets('[{"head":"A","relation":"knows person","tail":"B"}]')
    assert result[0].relation == "KNOWS_PERSON"
    assert sanitize_relationship("x` MATCH (n) //") == "X_MATCH_N"


def test_path_answer_is_factual_and_ordered():
    edges = [edge for snippet in SNIPPETS for edge in deterministic_triplets(snippet)]
    answer = answer_from_path(LocalGraph(edges).shortest_path("Midnight Sapphire", "Miller"))
    assert "Midnight Sapphire" in answer
    assert "Miller" in answer
    assert "Victor" in answer
