"""
Collecte de sources académiques sur les dictateurs.
Sources : Wikipedia (articles détaillés) + OpenAlex (papers open-access).
Liste des dictateurs : assets/dictators.txt
  (téléchargée le 2026-05-29 depuis https://en.everybodywiki.com/List_of_dictators)
"""

import os
import time
from pathlib import Path

import requests
import wikipediaapi

from utils.logger import get_logger

logger = get_logger(__name__)

ASSETS_DIR = Path("assets")

# Liste des dictateurs chargée depuis le fichier local
# Source : https://en.everybodywiki.com/List_of_dictators (téléchargée le 2026-05-29)
DICTATORS_FILE = ASSETS_DIR / "dictators.txt"


def load_dictators() -> list[str]:
    """Charge la liste des dictateurs depuis assets/dictators.txt."""
    names: list[str] = []
    for line in DICTATORS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    logger.info(f"dictators.txt: {len(names)} dictateurs chargés")
    return names

# Sujets connexes chargés depuis le fichier local
# Source : catégories Wikipedia FR (téléchargées le 2026-05-29)
TOPICS_FILE = ASSETS_DIR / "related_topics.txt"


def load_related_topics() -> list[str]:
    """Charge les sujets connexes depuis assets/related_topics.txt."""
    topics: list[str] = []
    for line in TOPICS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            topics.append(line)
    logger.info(f"related_topics.txt: {len(topics)} sujets chargés")
    return topics


def fetch_wikipedia_articles(language: str = "fr") -> list[Path]:
    """Récupère les articles Wikipedia en texte brut."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="UniversityRAGProject/1.0 (academic research)",
        language=language,
    )

    output_dir = ASSETS_DIR / "wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    dictators = load_dictators()
    topics = load_related_topics()
    all_titles = dictators + topics

    for title in all_titles:
        page = wiki.page(title)
        if not page.exists():
            logger.warning(f"Wikipedia: article '{title}' introuvable en '{language}'")
            continue

        filename = title.replace(" ", "_").replace("/", "_") + ".txt"
        filepath = output_dir / filename

        content = _format_wiki_page(page)

        filepath.write_text(content, encoding="utf-8")
        saved_files.append(filepath)
        logger.info(f"Wikipedia: {title} → {filepath} ({len(content)} chars)")

        time.sleep(0.5)  # rate limit

    logger.info(f"Wikipedia: {len(saved_files)} articles sauvegardés")
    return saved_files


def _format_wiki_page(page: wikipediaapi.WikipediaPage) -> str:
    """Formate un article Wikipedia avec métadonnées."""
    lines = [
        f"# {page.title}",
        f"Source: Wikipedia ({page.language})",
        f"URL: {page.fullurl}",
        "",
        page.summary,
        "",
    ]

    for section in page.sections:
        lines.extend(_format_section(section, level=2))

    return "\n".join(lines)


def _format_section(section: wikipediaapi.WikipediaPageSection, level: int) -> list[str]:
    """Formate une section Wikipedia récursivement."""
    lines = []
    if section.text.strip():
        lines.append(f"{'#' * level} {section.title}")
        lines.append(section.text)
        lines.append("")

    for sub in section.sections:
        lines.extend(_format_section(sub, level=min(level + 1, 4)))

    return lines


def fetch_openalex_papers(max_papers: int = 100) -> list[Path]:
    """Récupère des abstracts de papers open-access depuis OpenAlex."""
    output_dir = ASSETS_DIR / "openalex"
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        "dictator authoritarianism political regime",
        "totalitarianism fascism political repression",
        "genocide human rights violations dictator",
        "cult of personality propaganda authoritarian",
        "military coup dictatorship transition",
    ]

    all_papers: list[dict] = []

    for query in queries:
        papers = _search_openalex(query, per_page=max_papers // len(queries))
        all_papers.extend(papers)
        time.sleep(1)

    # Dédupliquer par ID
    seen_ids: set[str] = set()
    unique_papers: list[dict] = []
    for paper in all_papers:
        if paper["id"] not in seen_ids:
            seen_ids.add(paper["id"])
            unique_papers.append(paper)

    # Sauvegarder en un fichier par batch de 20 papers
    saved_files: list[Path] = []
    for i in range(0, len(unique_papers), 20):
        batch = unique_papers[i:i + 20]
        filepath = output_dir / f"papers_batch_{i // 20 + 1}.txt"

        content = _format_papers_batch(batch)
        filepath.write_text(content, encoding="utf-8")
        saved_files.append(filepath)

    logger.info(f"OpenAlex: {len(unique_papers)} papers → {len(saved_files)} fichiers")
    return saved_files


def _search_openalex(query: str, per_page: int = 20) -> list[dict]:
    """Cherche des papers open-access sur OpenAlex."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": "is_oa:true,type:article",
        "sort": "cited_by_count:desc",
        "per_page": per_page,
        "select": "id,title,doi,abstract_inverted_index,authorships,publication_year,cited_by_count",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.RequestException as e:
        logger.error(f"OpenAlex erreur pour '{query}': {e}")
        return []


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruit l'abstract depuis l'inverted index OpenAlex."""
    if not inverted_index:
        return ""

    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _format_papers_batch(papers: list[dict]) -> str:
    """Formate un batch de papers en texte."""
    lines = ["# Academic Papers on Dictatorships and Authoritarianism", ""]

    for paper in papers:
        title = paper.get("title", "Unknown")
        year = paper.get("publication_year", "?")
        doi = paper.get("doi", "")
        citations = paper.get("cited_by_count", 0)

        authors = []
        for auth in paper.get("authorships", [])[:3]:
            name = auth.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)
        authors_str = ", ".join(authors) if authors else "Unknown"

        abstract = _reconstruct_abstract(paper.get("abstract_inverted_index"))

        lines.append(f"## {title}")
        lines.append(f"Authors: {authors_str}")
        lines.append(f"Year: {year} | Citations: {citations}")
        if doi:
            lines.append(f"DOI: {doi}")
        lines.append("")
        if abstract:
            lines.append(abstract)
        else:
            lines.append("(Abstract not available)")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def collect_all_sources() -> None:
    """Collecte toutes les sources académiques."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Collecte des sources académiques ===")

    logger.info("--- Wikipedia ---")
    wiki_files = fetch_wikipedia_articles(language="fr")

    logger.info("--- OpenAlex (papers) ---")
    openalex_files = fetch_openalex_papers(max_papers=100)

    total = len(wiki_files) + len(openalex_files)
    logger.info(f"=== Collecte terminée : {total} fichiers ===")


if __name__ == "__main__":
    collect_all_sources()
