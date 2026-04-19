"""
ingest/chunker.py

Converts extracted 10-K section text files into semantically meaningful
chunks with rich metadata.

Robust against:
  - Files with no paragraph breaks (single giant text block)
  - Very short / noisy content
  - Windows-style line endings
  - Memory errors from pathologically large paragraphs
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import asdict, dataclass
from typing import Generator

from config import chunk_cfg, path_cfg


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source_file: str
    filing_index: int
    section: str
    chunk_index: int
    approx_year: str

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _infer_year(text: str, fallback: str) -> str:
    years = _YEAR_RE.findall(text[:2000])
    return max(set(years), key=years.count) if years else fallback


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """
    Clean raw section text before splitting.
    - Normalise Windows line endings
    - Collapse runs of 3+ blank lines to 2 (preserve paragraph breaks)
    - Remove lines that are pure whitespace / page numbers / short noise
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop lines that are only digits, dashes, dots (page numbers / rulers)
    lines = text.split("\n")
    cleaned = [l for l in lines if not re.fullmatch(r"[\s\d\-\.\_\|]{0,10}", l)]
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

# Hard cap: never feed more than this many chars into the sliding window.
# A paragraph larger than this is split by single newlines first.
_PARA_HARD_CAP = 50_000   # 50 KB — well within memory safety


def _hard_split(text: str, size: int) -> list[str]:
    """
    Last-resort hard character split with no overlap.
    Used only when text has no usable line breaks.
    """
    return [text[i : i + size].strip() for i in range(0, len(text), size) if text[i : i + size].strip()]


def _split_large_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a paragraph that is larger than _PARA_HARD_CAP.
    Strategy:
      1. Split on single newlines → sub-paragraphs
      2. If sub-paragraphs are still huge, do a hard character split
    """
    sub_paras = [p.strip() for p in paragraph.split("\n") if p.strip()]
    if not sub_paras:
        return []

    # If even individual lines are too big, hard-split them
    safe: list[str] = []
    for sp in sub_paras:
        if len(sp) > _PARA_HARD_CAP:
            safe.extend(_hard_split(sp, chunk_size))
        else:
            safe.append(sp)
    return safe


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping windows, preferring sentence boundaries.
    Only called on text already known to be <= _PARA_HARD_CAP chars.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Prefer splitting at last sentence boundary before `end`
        boundary = text.rfind(". ", start, end)
        if boundary <= start:
            boundary = end
        chunk = text[start : boundary + 1].strip()
        if chunk:
            chunks.append(chunk)
        # Advance, stepping back by overlap
        start = max(start + 1, boundary + 1 - overlap)

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_section_text(
    text: str,
    source_file: str,
    filing_index: int,
    section: str,
    approx_year: str,
) -> list[Chunk]:
    """
    Split a section string into Chunk objects.

    Strategy:
      1. Normalise the text (line endings, noise removal)
      2. Split on double newlines → paragraph list
      3. If a paragraph exceeds _PARA_HARD_CAP, pre-split on single newlines
      4. Accumulate paragraphs into a rolling buffer
      5. Flush buffer → sliding window when buffer exceeds chunk_size
    """
    text = _normalise(text)

    # ── Step 1: split into raw paragraphs ──────────────────────────────────
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # ── Step 2: guard against giant paragraphs ─────────────────────────────
    paragraphs: list[str] = []
    for para in raw_paras:
        if len(para) > _PARA_HARD_CAP:
            paragraphs.extend(_split_large_paragraph(para, chunk_cfg.chunk_size, chunk_cfg.chunk_overlap))
        else:
            paragraphs.append(para)

    if not paragraphs:
        return []

    # ── Step 3: accumulate buffer, flush to chunks ─────────────────────────
    _HEADING_RE = re.compile(r"^([A-Z][A-Z\s\-]{10,}|ITEM\s+\d)", re.MULTILINE)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush(buf: list[str]) -> None:
        merged = "\n\n".join(buf).strip()
        if len(merged) < chunk_cfg.min_chunk_length:
            return
        windows = _sliding_window(merged, chunk_cfg.chunk_size, chunk_cfg.chunk_overlap)
        for w in windows:
            if len(w) >= chunk_cfg.min_chunk_length:
                chunks.append(Chunk(
                    text=w,
                    source_file=source_file,
                    filing_index=filing_index,
                    section=section,
                    chunk_index=len(chunks),
                    approx_year=approx_year,
                ))

    for para in paragraphs:
        is_heading = bool(_HEADING_RE.match(para)) and len(para) < 200
        over_limit = buffer_len + len(para) > chunk_cfg.chunk_size

        if (is_heading or over_limit) and buffer:
            flush(buffer)
            buffer = []
            buffer_len = 0

        buffer.append(para)
        buffer_len += len(para)

    if buffer:
        flush(buffer)

    return chunks


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_all_chunks(sections_dir: str | None = None) -> list[Chunk]:
    """
    Walk the sections/ directory and chunk every *_item_*.txt file.
    Prints a per-file summary and raises clearly if no files are found.
    """
    sections_dir = sections_dir or path_cfg.sections_dir
    pattern = os.path.join(sections_dir, "10k_*_item_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No section files found matching '{pattern}'.\n"
            f"Check that SECTIONS_DIR is correct and the ingestion pipeline ran."
        )

    all_chunks: list[Chunk] = []
    for filepath in files:
        filename = os.path.basename(filepath)
        # Expected format: 10k_{index}_item_{n}.txt
        parts = filename.replace(".txt", "").split("_")
        try:
            filing_index = int(parts[1])
        except (IndexError, ValueError):
            filing_index = 0

        section = "_".join(parts[2:])   # e.g. "item_1" or "item_7a"

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        file_size_kb = round(len(text) / 1024, 1)
        approx_year = _infer_year(text, fallback=f"filing_{filing_index}")
        chunks = chunk_section_text(text, filename, filing_index, section, approx_year)
        all_chunks.extend(chunks)

        status = f"{len(chunks):3d} chunks" if chunks else "  0 chunks ⚠ (file may be empty or too short)"
        print(f"  ✓  {filename:42s}  {file_size_kb:7.1f} KB  →  {status}")

    total = len(all_chunks)
    if total == 0:
        raise ValueError(
            "Zero chunks produced from all section files.\n"
            "This usually means the section files are empty or contain only\n"
            "very short lines. Check your sections/ directory content."
        )

    print(f"\nTotal chunks produced: {total}")
    return all_chunks


# ---------------------------------------------------------------------------
# Diagnostic helper — run directly to inspect a single file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run ingest/chunker.py <path_to_section_file.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, encoding="utf-8", errors="replace") as f:
        text = f.read()

    print(f"File size  : {len(text):,} chars")
    print(f"Line count : {text.count(chr(10)):,}")
    print(f"Para breaks: {len(re.findall(chr(10)+chr(10), text)):,}")
    print()

    chunks = chunk_section_text(
        text=text,
        source_file=os.path.basename(filepath),
        filing_index=0,
        section="debug",
        approx_year="unknown",
    )
    print(f"Chunks produced: {len(chunks)}")
    if chunks:
        print(f"\n--- First chunk ({len(chunks[0].text)} chars) ---")
        print(chunks[0].text[:500])