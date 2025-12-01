# src/local_ingest.py
import os
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfReader


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Warning: could not read page {i} of {path}: {e}")
            text = ""
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    # Extend here if you want .md, .docx, etc.
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """
    Very simple chunker: hard split every max_chars.
    You can later improve to split on sentences/paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    return [
        text[i : i + max_chars]
        for i in range(0, len(text), max_chars)
    ]


def build_documents_from_folder(root_folder: str, collection_name: str) -> List[Dict[str, Any]]:
    root = Path(root_folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")

    supported_exts = {".pdf", ".txt"}
    documents: List[Dict[str, Any]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported_exts:
            continue

        try:
            full_text = extract_text(path)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
            continue

        chunks = chunk_text(full_text)
        rel_path = str(path.relative_to(root))

        for idx, chunk in enumerate(chunks):
            doc_id = f"{rel_path}::chunk-{idx}"
            documents.append(
                {
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {
                        "source": "local",
                        "collection": collection_name,
                        "file_path": str(path),
                        "relative_path": rel_path,
                        "chunk_index": idx,
                    },
                }
            )

    return documents
