"""Reproducible bench corpus builder.

Stages (each idempotent — skipped if its outputs already exist):
  1. Download pinned arXiv PDFs.
  2. Extract per-page text with pypdf, drop short/reference pages.
  3. Sliding-window chunk per bench/data/CHUNKING.md.
  4. Embed each chunk with jinaai/jina-embeddings-v2-base-en (pinned revision).

Outputs (gitignored):
  pdfs/<id>.pdf
  text/<id>.txt
  chunks.jsonl
  embeddings.npy     (shape (N, 768), L2-normalized float32)
  pdf_sha256.txt     (reproducibility receipt)

Usage:
  uv pip install requests pypdf numpy sentence-transformers
  python bench/data/build_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

DATA = Path(__file__).parent
PDFS = DATA / "pdfs"
TEXT = DATA / "text"
CHUNKS = DATA / "chunks.jsonl"
EMB = DATA / "embeddings.npy"
SHA = DATA / "pdf_sha256.txt"
IDS = DATA / "arxiv_ids.txt"
REVISION_FILE = DATA / "MODEL_REVISION.txt"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_PAGE_CHARS = 200
MAX_CHUNKS_PER_PAPER = 200
DEDUP_PREFIX_LEN = 64


def load_ids() -> list[str]:
    out = []
    for ln in IDS.read_text().splitlines():
        stripped = ln.split("#", 1)[0].strip()
        if stripped:
            out.append(stripped)
    return out


def load_revision() -> tuple[str, str]:
    model, rev = None, None
    for ln in REVISION_FILE.read_text().splitlines():
        if ln.startswith("model:"):
            model = ln.split(":", 1)[1].strip()
        elif ln.startswith("revision:"):
            rev = ln.split(":", 1)[1].strip()
    if not model or not rev:
        sys.exit("ERROR: bench/data/MODEL_REVISION.txt malformed.")
    return model, rev


def download_pdfs(ids: list[str]) -> None:
    PDFS.mkdir(exist_ok=True)
    receipts: dict[str, str] = {}
    if SHA.exists():
        for ln in SHA.read_text().splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                receipts[k.strip()] = v.strip()
    for aid in ids:
        path = PDFS / f"{aid}.pdf"
        if path.exists():
            continue
        url = f"https://arxiv.org/pdf/{aid}.pdf"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": "liblistofclusters-bench/0.1"})
                r.raise_for_status()
                path.write_bytes(r.content)
                receipts[aid] = hashlib.sha256(r.content).hexdigest()
                print(f"  downloaded {aid}")
                time.sleep(1.0)
                break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed for {aid}: {e}", file=sys.stderr)
                time.sleep(2.0)
        else:
            print(f"  FAILED {aid}", file=sys.stderr)
    with SHA.open("w") as f:
        for aid, sha in sorted(receipts.items()):
            f.write(f"{aid}: {sha}\n")


def extract_text(ids: list[str]) -> None:
    TEXT.mkdir(exist_ok=True)
    from pypdf import PdfReader

    ref_re = re.compile(r"^\s*references\s*$", re.IGNORECASE | re.MULTILINE)
    for aid in ids:
        out = TEXT / f"{aid}.txt"
        if out.exists():
            continue
        pdf = PDFS / f"{aid}.pdf"
        if not pdf.exists():
            continue
        try:
            reader = PdfReader(str(pdf))
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if len(re.sub(r"\s", "", t)) < MIN_PAGE_CHARS:
                    continue
                pages.append(t)
            full = "\n".join(pages)
            m = ref_re.search(full)
            if m:
                full = full[: m.start()]
            out.write_text(full)
        except Exception as e:
            print(f"  text extract failed {aid}: {e}", file=sys.stderr)


def chunk_all(ids: list[str]) -> None:
    if CHUNKS.exists():
        return
    seen_prefix: set[str] = set()
    rows = []
    chunk_id = 0
    for aid in ids:
        path = TEXT / f"{aid}.txt"
        if not path.exists():
            continue
        text = path.read_text()
        i = 0
        per_paper = 0
        while i < len(text) and per_paper < MAX_CHUNKS_PER_PAPER:
            chunk = text[i : i + CHUNK_SIZE]
            prefix = chunk[:DEDUP_PREFIX_LEN]
            if prefix not in seen_prefix:
                seen_prefix.add(prefix)
                rows.append({"id": chunk_id, "arxiv_id": aid, "text": chunk})
                chunk_id += 1
                per_paper += 1
            i += CHUNK_SIZE - CHUNK_OVERLAP
    with CHUNKS.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows)} chunks")


def embed_all() -> None:
    if EMB.exists():
        return
    if not CHUNKS.exists():
        sys.exit("chunks.jsonl missing - run chunk stage first")
    rows = [json.loads(ln) for ln in CHUNKS.read_text().splitlines()]
    texts = [r["text"] for r in rows]

    from sentence_transformers import SentenceTransformer

    model_name, revision = load_revision()
    print(f"  loading {model_name}@{revision[:7]}...")
    model = SentenceTransformer(model_name, revision=revision, trust_remote_code=True)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    np.save(EMB, vecs.astype(np.float32))
    print(f"  wrote embeddings shape={vecs.shape}")


def main() -> int:
    ids = load_ids()
    print(f"corpus: {len(ids)} arXiv IDs")
    download_pdfs(ids)
    extract_text(ids)
    chunk_all(ids)
    embed_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
