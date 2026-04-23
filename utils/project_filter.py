import os

_STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "en", "et", "est",
    "que", "qui", "pour", "par", "sur", "dans", "avec", "the", "and", "for",
    "quel", "quelle", "quels", "quelles", "quel", "comment", "pourquoi",
}


def get_source_stem(source_path: str) -> str:
    return os.path.splitext(os.path.basename(source_path))[0]


def detect_project_in_query(query: str, known_stems: list[str]) -> str | None:
    query_tokens = [
        w for w in query.split()
        if len(w) >= 4 and w.lower() not in _STOP_WORDS
    ]
    query_tokens.sort(key=len, reverse=True)

    for token in query_tokens:
        token_lower = token.lower()
        for stem in known_stems:
            if token_lower in stem.lower():
                return token

    return None
