# src/local_ingest.py
import os
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfReader

# Import config at module level for use in build_documents_from_folder
# chunk_text remains parameterless, callers must provide max_chars


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


def chunk_text(text: str, *, max_chars: int) -> List[str]:
    """
    Chunk text into segments.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk (REQUIRED - caller must specify)
    
    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    return [
        text[i : i + max_chars]
        for i in range(0, len(text), max_chars)
    ]


def build_documents_from_folder(root_folder: str, collection_name: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Build document list from folder for ingestion.
    
    Args:
        root_folder: Path to folder containing documents
        collection_name: Name of collection for metadata
        max_chars: Maximum characters per chunk
    """
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

        chunks = chunk_text(full_text, max_chars=max_chars)
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
