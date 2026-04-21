from itertools import groupby
from pathlib import Path

from langchain_core.documents import Document


def pdf_to_txt(docs: list[Document]) -> list[Path]:
    results: list[Path] = []
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("source", ""))

    for source, group in groupby(sorted_docs, key=lambda d: d.metadata.get("source", "")):
        txt_path = Path(source).with_suffix(".txt")
        lines: list[str] = []
        for doc in group:
            slide_number = doc.metadata.get("page", 0) + 1
            lines.append(f"===== SLIDE {slide_number} =====")
            lines.append(doc.page_content.strip())
            lines.append("")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        results.append(txt_path)

    return results
