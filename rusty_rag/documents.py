import pathlib


def load_documents(directory: str) -> list[dict]:
    """Load all .txt and .md files from a directory."""
    base = pathlib.Path(directory)
    docs = []
    for path in sorted(base.glob("*.txt")) + sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "document_id": path.stem,
            "source_path": str(path),
            "text": text,
        })
    return docs
