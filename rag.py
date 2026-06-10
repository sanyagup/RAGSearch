"""Minimal RAG (Retrieval-Augmented Generation) search over local text files.

Pipeline:
  1. ingest  -> chunk documents, embed them with OpenAI, save a local vector index
  2. ask     -> embed the question, find the most similar chunks, ask the LLM to
                answer using only those chunks (with citations)

The index is plain numpy on disk (index.npz + index.json) — no external DB needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Load OPENAI_API_KEY (and optional model overrides) from the .env file.
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

INDEX_VECTORS = Path("index.npz")
INDEX_META = Path("index.json")


def _client() -> OpenAI:
    """Create the OpenAI client, with a friendly error if the key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI()


# --------------------------------------------------------------------------- #
# Reading files (text, markdown, PDF)
# --------------------------------------------------------------------------- #
SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    """Extract plain text from a .txt, .md, or .pdf file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping word windows so context isn't cut mid-thought."""
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings (so dot product == cosine similarity)."""
    vectors: list[list[float]] = []
    # Batch to stay well under request-size limits.
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


# --------------------------------------------------------------------------- #
# Index persistence
# --------------------------------------------------------------------------- #
@dataclass
class Index:
    vectors: np.ndarray          # shape (n_chunks, dim)
    chunks: list[str]            # the chunk text
    sources: list[str]           # filename each chunk came from

    def save(self) -> None:
        np.savez_compressed(INDEX_VECTORS, vectors=self.vectors)
        INDEX_META.write_text(
            json.dumps({"chunks": self.chunks, "sources": self.sources})
        )

    @classmethod
    def load(cls) -> "Index":
        if not INDEX_VECTORS.exists() or not INDEX_META.exists():
            raise SystemExit("No index found. Run:  python main.py ingest <folder>")
        meta = json.loads(INDEX_META.read_text())
        with np.load(INDEX_VECTORS) as data:
            vectors = data["vectors"]
        return cls(vectors, meta["chunks"], meta["sources"])


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def ingest(folder: str) -> Index:
    """Read every .txt/.md file under `folder`, chunk, embed, and save the index."""
    client = _client()
    root = Path(folder)
    # Accept a single file or a whole folder.
    if root.is_file():
        files = [root] if root.suffix.lower() in SUPPORTED else []
        root = root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No .txt, .md, or .pdf files found under {folder!r}")

    chunks, sources = [], []
    for path in files:
        text = read_file(path)
        if not text.strip():
            print(f"  (skipped, no extractable text: {path.name})")
            continue
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(str(path.relative_to(root)))

    print(f"Embedding {len(chunks)} chunks from {len(files)} file(s)...")
    index = Index(embed(client, chunks), chunks, sources)
    index.save()
    print(f"Saved index ({len(chunks)} chunks) -> {INDEX_VECTORS}, {INDEX_META}")
    return index


# --------------------------------------------------------------------------- #
# Search + answer
# --------------------------------------------------------------------------- #
def search(index: Index, query_vec: np.ndarray, k: int = 4) -> list[tuple[float, int]]:
    """Return the top-k (score, chunk_index) by cosine similarity."""
    scores = index.vectors @ query_vec
    top = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), int(i)) for i in top]


def ask(question: str, k: int = 4) -> str:
    """Retrieve relevant chunks and have the LLM answer using only those chunks."""
    client = _client()
    index = Index.load()
    query_vec = embed(client, [question])[0]
    hits = search(index, query_vec, k)

    context = "\n\n".join(
        f"[{n + 1}] (source: {index.sources[i]})\n{index.chunks[i]}"
        for n, (_, i) in enumerate(hits)
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the numbered context below. "
                    "Cite sources inline like [1], [2]. If the answer isn't in the "
                    "context, say you don't know."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    used = "\n".join(f"  [{n + 1}] {index.sources[i]} (score {s:.3f})"
                     for n, (s, i) in enumerate(hits))
    return f"{answer}\n\nSources:\n{used}"
