"""
ingest/embedder.py

Embeds chunks using fastembed (all-MiniLM-L6-v2 via ONNX runtime).

Why fastembed instead of sentence-transformers:
  - No PyTorch dependency (~50 MB vs ~2 GB)
  - Same all-MiniLM-L6-v2 model, same 384-dim vectors
  - Faster cold start on Windows
  - No CUDA/GPU setup required
  - Model downloads once to ~/.cache/fastembed/
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import chromadb
from fastembed import TextEmbedding

from config import vector_cfg, agent_cfg
from ingest.chunker import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Local ONNX embedder via fastembed
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    """Load model once, cache for process lifetime. Downloads ~50 MB on first run."""
    print(f"Loading '{EMBEDDING_MODEL}' via fastembed (downloads once on first run) …")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    print("✓  Model ready")
    return model


class LocalEmbedder:

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings. Returns 384-dim vectors."""
        model = _get_model()
        # fastembed returns a generator of numpy arrays
        vectors = list(model.embed(texts))
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        model = _get_model()
        vectors = list(model.embed([query]))
        return vectors[0].tolist()


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------

class FinancialVectorStore:
    """Persistent ChromaDB collection for Apple 10-K chunks."""

    def __init__(self) -> None:
        self._embedder = LocalEmbedder()
        self._chroma   = chromadb.PersistentClient(path=vector_cfg.persist_directory)
        self._col      = self._chroma.get_or_create_collection(
            name=vector_cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks — safe to re-run (idempotent)."""
        if not chunks:
            print("No chunks to index.")
            return

        print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} …")
        texts     = [c.text for c in chunks]
        vectors   = self._embedder.embed(texts)
        ids       = [f"{c.source_file}__chunk_{c.chunk_index}" for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        self._col.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"✓  Indexed {len(chunks)} chunks into '{vector_cfg.collection_name}'")

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        section_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
    ) -> list[dict]:
        """Embed the query and return the top-k most similar chunks."""
        k = top_k or agent_cfg.top_k_retrieval

        where: dict | None = None
        filters: list[dict] = []
        if section_filter:
            filters.append({"section": {"$eq": section_filter}})
        if year_filter:
            filters.append({"approx_year": {"$eq": year_filter}})
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        query_vec = self._embedder.embed_query(query_text)

        results = self._col.query(
            query_embeddings=[query_vec],
            n_results=min(k, self._col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "source_file": meta.get("source_file", ""),
                "section": meta.get("section", ""),
                "approx_year": meta.get("approx_year", ""),
                "filing_index": meta.get("filing_index", 0),
                "distance": round(dist, 4),
                "relevance_score": round(1 - dist, 4),
            })

        return output

    def count(self) -> int:
        return self._col.count()

    def collection_exists(self) -> bool:
        return self.count() > 0


# ---------------------------------------------------------------------------
# Build the index  (called by `uv run main.py setup`)
# ---------------------------------------------------------------------------

def build_index(sections_dir: str | None = None) -> None:
    """End-to-end: load chunks → embed → store."""
    from ingest.chunker import load_all_chunks

    print("=" * 60)
    print("STEP 1 — Loading & chunking section files")
    print("=" * 60)
    chunks = load_all_chunks(sections_dir)

    print("\n" + "=" * 60)
    print("STEP 2 — Embedding & indexing into ChromaDB")
    print("=" * 60)
    store = FinancialVectorStore()
    store.index_chunks(chunks)

    print(f"\nDone!  Total documents in store: {store.count()}")


if __name__ == "__main__":
    build_index()