# src/local_ingest.py
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
import platform
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract


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


def remove_control_chars(s: str) -> str:
    return "".join(
        ch
        for ch in s
        if (ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127))
    )


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from PDFs/TXTs:
    - Normalize Unicode (NFKC) so fancy characters become simpler equivalents.
    - Remove control characters (except newlines and tabs).
    - Replace weird PDF artifacts like form feed.
    - Collapse excessive whitespace.
    """

    if not text:
        return ""

    # 1) Unicode normalization to simplify weird characters
    text = unicodedata.normalize("NFKC", text)

    # 2) Replace common PDF artifacts
    #    \x0c is form feed, often appears between pages
    text = text.replace("\x0c", "\n")

    # 3) Strip out control characters except newlines and tabs
    #    Control chars ASCII 0-31 & 127
    #    Keep: \n (10), \t (9)
    text = remove_control_chars(text)

    # 4) Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5) Collapse spaces but preserve paragraph breaks
    #    - First, collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    #    - Then, collapse 3+ newlines into just 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6) Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # 7) Optionally: remove lines that are *only* junk chars
    #    (Lots of PDFs produce lines like "§§§§§" or "———")
    cleaned_lines = []
    junk_pattern = re.compile(r"^[^\w\s]{3,}$")  # 3+ non-word non-space chars
    for line in text.splitlines():
        if junk_pattern.match(line.strip()):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    return text.strip()

def looks_like_gibberish(text: str) -> bool:
    """
    Heuristic to detect junk (random glyphs from bad PDFs).

    This is intentionally a bit strict so obviously-bad text like GL-cap.pdf
    will be treated as gibberish and pushed through OCR.
    """

    if not text:
        return True

    sample = text[:5000].replace("\n", " ")
    total = len(sample)
    if total == 0:
        return True

    letters = sum(c.isalpha() for c in sample)
    spaces = sample.count(" ")
    digits = sum(c.isdigit() for c in sample)
    punctuation = sum(
        (not c.isalnum()) and not c.isspace()
        for c in sample
    )

    letter_ratio = letters / total
    space_ratio = spaces / total
    digit_ratio = digits / total
    punct_ratio = punctuation / total

    # Word stats
    words = re.findall(r"[A-Za-z]{3,}", sample)
    vowel_words = [w for w in words if re.search(r"[aeiouyAEIOUY]", w)]
    vowel_ratio = (len(vowel_words) / len(words)) if words else 0.0

    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0

    # Uncomment to debug thresholds if you want:
    # print(f"DEBUG: words={len(words)}, letter={letter_ratio:.2f}, space={space_ratio:.2f}, "
    #       f"digit={digit_ratio:.2f}, punct={punct_ratio:.2f}, vowel_words={vowel_ratio:.2f}, "
    #       f"avg_word_len={avg_word_len:.2f}")

    # ---- Heuristic rules ----
    # Very few proper words -> probably junk
    if len(words) < 30:
        return True

    # Too many punctuation/symbols compared to everything else
    if punct_ratio > 0.25:
        return True

    # Not enough letters (lots of symbols/numbers)
    if letter_ratio < 0.6:
        return True

    # Almost no spaces = long garbage strings
    if space_ratio < 0.05:
        return True

    # Words rarely contain vowels -> not natural language
    if vowel_ratio < 0.35:
        return True

    # Average word length way too long (e.g. random strings)
    if avg_word_len > 12:
        return True

    return False


def ocr_pdf_to_text(path: Path, max_pages: int | None = 5) -> str:
    """
    Convert PDF pages to images and run OCR on each page.
    max_pages limits pages during testing for speed.
    """

    system = platform.system()
    poppler_path = None

    if system == "Windows":
        poppler_path = r"C:\poppler-25.11.0\Library\bin"

    try:
        if poppler_path:
            images = convert_from_path(
                str(path),
                dpi=300,
                poppler_path=poppler_path,
            )
        else:
            # On Linux/Docker, rely on system poppler (poppler-utils)
            images = convert_from_path(
                str(path),
                dpi=300,
            )
    except Exception as e:
        print(f"Error converting PDF to images for OCR: {e}")
        return ""

    if max_pages is not None:
        images = images[:max_pages]

    ocr_text = []

    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang="eng")
            ocr_text.append(text)
        except Exception as e:
            print(f"Error OCR page {i}: {e}")

    return "\n\n".join(ocr_text)


def extract_text_with_ocr_fallback(path: Path) -> str:
    """
    High-level helper:

    1. Use existing extract_text (PyPDF2 / txt).
    2. Clean it.
    3. If it looks good (not gibberish) -> return cleaned text.
    4. If it looks bad AND it's a PDF -> try OCR, clean that.
    5. If OCR still looks bad/empty -> raise ValueError so caller can skip.
    """

    try:
        raw = extract_text(path)
    except Exception as e:
        print(f"Error in base extract_text for {path}: {e}")
        raw = ""

    cleaned = clean_text(raw)

    # If normal extraction gave us something reasonable, use it
    if cleaned and not looks_like_gibberish(cleaned):
        return cleaned

    # Only PDFs can be OCR’d
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Text for {path} appears gibberish and is not a PDF; skipping.")

    # OCR fallback
    print(f"Normal extraction for {path} looks gibberish/empty. Trying OCR...")
    ocr_raw = ocr_pdf_to_text(path, max_pages=None)  # None = all pages
    ocr_cleaned = clean_text(ocr_raw)

    if ocr_cleaned and not looks_like_gibberish(ocr_cleaned):
        return ocr_cleaned

    raise ValueError(f"Text for {path} appears gibberish even after OCR; skipping.")