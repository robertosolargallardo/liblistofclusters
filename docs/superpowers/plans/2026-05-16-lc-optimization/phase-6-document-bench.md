# Phase 6 — Real-document benchmark (arXiv PDFs + Jina v2)

**Goal.** Replace synthetic uniform-random as the headline benchmark with a reproducible real-document workload: arXiv PDFs → 512-char chunks → Jina v2 base English embeddings (768-dim) → angular search. The synthetic sweep stays for controlled ablations.

**References.** [GMW+23] (Jina v2 model), [RG19] (Sentence-BERT foundation, justifies cosine/angular as the natural metric for these encoders).

**Files modified / created.**
- Create: `bench/data/arxiv_ids.txt` — pinned arXiv ID list (committed).
- Create: `bench/data/MODEL_REVISION.txt` — pinned Jina HF revision (committed; **engineer must replace placeholder SHA before first run**).
- Create: `bench/data/CHUNKING.md` — chunking rules (committed).
- Create: `bench/data/.gitignore` — covers `pdfs/`, `text/`, `embeddings.npy`, `chunks.jsonl`, `pdf_sha256.txt`.
- Create: `bench/data/build_corpus.py` — corpus builder (download → extract → chunk → embed).
- Create: `bench/compare_documents.py` — bench harness (LC variants + Faiss + hnswlib + sklearn).
- Modify: `bench/README.md` — restructure: real-document headline first, synthetic ablation below.
- Create: `.github/workflows/bench_documents.yml` — manual + weekly cron workflow (does not run per-PR).

---

### Task 6.1: Corpus metadata files (committed)

**Engineer note before running.** The arXiv ID list and the Jina revision SHA below are placeholders. You **must vet** them before the first corpus build:

1. Open each arXiv URL (`https://arxiv.org/abs/<id>`) and confirm it's a real paper relevant to the benchmark (metric indexing, ANN, embeddings, retrieval, contrastive learning, vector databases, or general ML).
2. Replace the Jina revision in `MODEL_REVISION.txt` with the current HEAD of [`jinaai/jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/commits/main) — copy the 40-char Git SHA, not a tag.

These are reproducibility receipts; once set they freeze the bench output.

- [ ] **Step 1: Create `bench/data/arxiv_ids.txt`**

```
# Pinned arXiv IDs for the bench corpus.
# Each line: one arXiv ID (no URL, no version suffix).
# Vet before first run — see phase-6-document-bench.md Task 6.1.
1603.09320   # HNSW
1908.10396   # ScaNN (anisotropic VQ)
2104.04915   # FAISS engineering paper
2007.00808   # DiskANN
1908.10084   # Sentence-BERT
2310.19923   # Jina Embeddings 2
2309.07597   # BGE (C-Pack)
2104.08663   # GPL retrieval pretraining
2007.13880   # DPR (dense passage retrieval)
2007.00808   # (intentional duplicate to test dedup; remove if unwanted)
2104.13533   # SimCSE contrastive embeddings
2210.07316   # ColBERTv2
2103.00020   # CLIP (multimodal embeddings; relevant as a popular comparator)
1310.4546    # word2vec (classic embedding baseline)
1607.04606   # fastText
1606.04467   # Triplet loss for embeddings
2202.05144   # NV-Embed family
2106.04561   # Contriever
2310.07554   # Mistral-Embed (note: vet for retrieval relevance)
2305.11705   # BGE-M3 era
# --- metric indexing / ANN classics
1108.1990    # M-tree revisited (note: may not exist; replace if so)
0807.0810    # AESA / LAESA exposition (note: example of pinning a survey-era paper)
2009.03300   # FAISS HNSW + IVF integration
# (Continue to ~50 entries; vet each before committing.)
```

Comments after `#` are honored by `build_corpus.py`'s parser (whitespace-trimmed; blank lines skipped). Aim for ~50 vetted entries — the corpus should produce ~5–10 k chunks, in the same N range as the synthetic sweep.

- [ ] **Step 2: Create `bench/data/MODEL_REVISION.txt`**

```
# Jina v2 base English embeddings — pinned HuggingFace revision.
# Update revision: with the current main-branch SHA from
# https://huggingface.co/jinaai/jina-embeddings-v2-base-en/commits/main
# and re-run build_corpus.py to refresh the cache.
model:    jinaai/jina-embeddings-v2-base-en
# TODO: replace with real revision SHA before first run.
revision: REPLACE_WITH_REAL_REVISION_SHA
```

- [ ] **Step 3: Create `bench/data/CHUNKING.md`**

```markdown
# Chunking rules (bench corpus)

- **Chunk size:** 512 characters per chunk.
- **Overlap:** 64 characters between consecutive chunks.
- **Page-level pre-filter:** drop pages with < 200 non-whitespace characters
  (typically figure-only pages or pages that are mostly references).
- **Reference strip:** regex-strip the "References" section onwards
  (case-insensitive line match on "References" alone).
- **Dedup:** skip chunks whose first 64 characters match an earlier chunk
  byte-for-byte (kills boilerplate / repeated paper headers).
- **Per-paper cap:** at most 200 chunks per paper, to keep the corpus
  balanced across sources.
```

- [ ] **Step 4: Create `bench/data/.gitignore`**

```
pdfs/
text/
embeddings.npy
chunks.jsonl
pdf_sha256.txt
```

- [ ] **Step 5: Commit**

```sh
git add bench/data/arxiv_ids.txt bench/data/MODEL_REVISION.txt bench/data/CHUNKING.md bench/data/.gitignore
git commit -m "phase 6.1: bench corpus metadata + chunking rules"
```

---

### Task 6.2: Corpus builder

**References.** Jina v2 documentation [GMW+23] for input length handling and the L2-normalized output convention. SentenceTransformers usage follows the conventions established by [RG19].

- [ ] **Step 1: Create `bench/data/build_corpus.py`**

```python
"""Reproducible bench corpus builder.

Stages, each idempotent (skipped if outputs exist):
  1. Download pinned arXiv PDFs (with SHA256 receipt).
  2. Extract per-page text with pypdf, drop short/reference pages.
  3. Sliding-window chunk per bench/data/CHUNKING.md.
  4. Embed each chunk with jinaai/jina-embeddings-v2-base-en (pinned rev).

Outputs (gitignored):
  pdfs/<id>.pdf
  text/<id>.txt
  chunks.jsonl
  embeddings.npy     (shape N x 768, L2-normalized float32)
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
    text = REVISION_FILE.read_text()
    model, rev = None, None
    for ln in text.splitlines():
        if ln.startswith("model:"):
            model = ln.split(":", 1)[1].strip()
        elif ln.startswith("revision:"):
            rev = ln.split(":", 1)[1].strip()
    if not model or not rev or rev == "REPLACE_WITH_REAL_REVISION_SHA":
        sys.exit(
            "ERROR: bench/data/MODEL_REVISION.txt has placeholder SHA. "
            "Replace it with the real Jina v2 revision before running."
        )
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
                r = requests.get(url, timeout=30)
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
        sys.exit("chunks.jsonl missing — run chunk stage first")
    rows = [json.loads(ln) for ln in CHUNKS.read_text().splitlines()]
    texts = [r["text"] for r in rows]

    from sentence_transformers import SentenceTransformer

    model_name, revision = load_revision()
    model = SentenceTransformer(model_name, revision=revision, trust_remote_code=True)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    np.save(EMB, vecs.astype(np.float32))
    print(f"  wrote embeddings shape={vecs.shape}")


def main() -> int:
    ids = load_ids()
    download_pdfs(ids)
    extract_text(ids)
    chunk_all(ids)
    embed_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test on 3 IDs**

Temporarily edit `arxiv_ids.txt` to 3 entries. Run:

```sh
uv venv && source .venv/bin/activate
uv pip install requests pypdf numpy sentence-transformers
python bench/data/build_corpus.py
ls -la bench/data/
```

Expected: `pdfs/`, `text/`, `chunks.jsonl`, `embeddings.npy`, `pdf_sha256.txt` all populated. The `.npy` shape should be `(num_chunks, 768)`.

Restore the full ID list before the real bench.

- [ ] **Step 3: Commit**

```sh
git add bench/data/build_corpus.py
git commit -m "phase 6.2: corpus builder (arXiv PDFs + Jina v2 embeddings)"
```

---

### Task 6.3: Bench harness `compare_documents.py`

**Note.** The Python bindings expose `bulk_build(V, ids)`, `build_aesa(k)`, `freeze()`, `knn(q, k)`, `batch_knn(Q, k, nthreads)` — all reading lists / numpy arrays as documented in `python/src/_listofclusters.cc`. Do not invent other method names.

- [ ] **Step 1: Create `bench/compare_documents.py`**

```python
"""Real-document benchmark: angular search on arXiv chunks (Jina v2 768-dim).

Methods compared:
  liblistofclusters (multiple LC configs)
  faiss.IndexFlatIP        (SIMD brute force on L2-normalized inputs)
  faiss.IndexIVFFlat       (nprobe sweep)
  faiss.IndexHNSWFlat      (efSearch sweep)
  hnswlib                  (ef sweep, cosine space)
  sklearn.BallTree         (euclidean on L2-normalized = monotone in angular)
  numpy brute              (ground truth)

Outputs:
  bench/compare_documents.csv
  bench/compare_documents_pareto.png

Usage:
  python bench/compare_documents.py [--queries 200] [--k 10]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
EMB = DATA / "embeddings.npy"


def load_embeddings() -> np.ndarray:
    if not EMB.exists():
        raise SystemExit("embeddings.npy missing — run bench/data/build_corpus.py first")
    return np.load(EMB)


def split(emb: np.ndarray, n_queries: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(emb.shape[0])
    return emb[idx[n_queries:]], emb[idx[:n_queries]]


def brute_truth(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    dots = q @ db.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    return np.argsort(dists, axis=1)[:, :k]


def recall(got: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    total = truth.shape[0] * truth.shape[1]
    for g, t in zip(got, truth):
        hits += len(set(g.tolist()) & set(t.tolist()))
    return hits / total


def run_listofclusters(db: np.ndarray, queries: np.ndarray, k: int,
                       k_anchors: int, nthreads: int):
    import listofclusters
    idx = listofclusters.Index(metric="euclidean")  # angular needs L2-norm + euclidean equiv;
    # NOTE: bindings currently support euclidean/manhattan/chebyshev/canberra.
    # For angular search on normalized vectors, Euclidean is monotone w.r.t.
    # angular: d_ang(a,b) = arccos(1 - d_eucl(a,b)^2 / 2). Using metric="euclidean"
    # preserves the ranking; the absolute distances differ but the kNN ids match.
    ids = np.arange(db.shape[0], dtype=np.uint32)
    db_c = np.ascontiguousarray(db, dtype=np.float64)
    q_c  = np.ascontiguousarray(queries, dtype=np.float64)
    idx.bulk_build(db_c, ids)
    if k_anchors > 0:
        idx.build_aesa(k_anchors)
    idx.freeze()
    nbrs, _ = idx.batch_knn(q_c, k=k, nthreads=nthreads)
    return np.array(nbrs, dtype=np.uint32)


def run_faiss_flatip(db: np.ndarray, queries: np.ndarray, k: int):
    import faiss
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db.astype(np.float32))
    return index.search(queries.astype(np.float32), k)[1]


def run_faiss_ivf(db: np.ndarray, queries: np.ndarray, k: int, nprobe: int):
    import faiss
    nlist = min(64, max(4, int(np.sqrt(db.shape[0]))))
    quant = faiss.IndexFlatIP(db.shape[1])
    index = faiss.IndexIVFFlat(quant, db.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(db.astype(np.float32))
    index.add(db.astype(np.float32))
    index.nprobe = nprobe
    return index.search(queries.astype(np.float32), k)[1]


def run_hnswlib(db: np.ndarray, queries: np.ndarray, k: int, ef: int):
    import hnswlib
    index = hnswlib.Index(space="cosine", dim=db.shape[1])
    index.init_index(max_elements=db.shape[0], ef_construction=200, M=16)
    index.add_items(db.astype(np.float32))
    index.set_ef(ef)
    return index.knn_query(queries.astype(np.float32), k=k)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = load_embeddings()
    db, queries = split(emb, args.queries)
    print(f"loaded {emb.shape[0]} embeddings, dim={emb.shape[1]}, db={db.shape[0]}, q={queries.shape[0]}")

    truth = brute_truth(db, queries, args.k)

    rows = []
    methods: list[tuple[str, callable]] = [
        ("LC FFT, no AESA, 1T",     lambda: run_listofclusters(db, queries, args.k, 0,  1)),
        ("LC FFT, AESA k=16, 1T",   lambda: run_listofclusters(db, queries, args.k, 16, 1)),
        ("LC FFT, AESA k=16, HWT",  lambda: run_listofclusters(db, queries, args.k, 16, 0)),
        ("faiss.FlatIP",            lambda: run_faiss_flatip(db, queries, args.k)),
        ("faiss.IVFFlat np=10",     lambda: run_faiss_ivf(db, queries, args.k, 10)),
        ("faiss.IVFFlat np=32",     lambda: run_faiss_ivf(db, queries, args.k, 32)),
        ("hnswlib ef=32",           lambda: run_hnswlib(db, queries, args.k, 32)),
        ("hnswlib ef=64",           lambda: run_hnswlib(db, queries, args.k, 64)),
    ]
    for name, fn in methods:
        try:
            t0 = time.perf_counter()
            got = fn()
            t1 = time.perf_counter()
            qps = queries.shape[0] / (t1 - t0)
            r = recall(np.asarray(got), truth)
            rows.append({"method": name, "recall": r, "qps": qps})
            print(f"  {name:30s}  recall={r:.3f}  qps={qps:10.1f}")
        except Exception as e:
            print(f"  {name:30s}  skipped ({e})")

    csv_path = Path(__file__).parent / "compare_documents.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "qps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in rows:
            ax.scatter(r["recall"], r["qps"], s=80, label=r["method"])
        ax.set_xlabel("Recall@k")
        ax.set_ylabel("QPS (log)")
        ax.set_yscale("log")
        ax.set_title(f"arXiv + Jina v2 (N={db.shape[0]}, D={db.shape[1]}, k={args.k})")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        png = Path(__file__).parent / "compare_documents_pareto.png"
        fig.savefig(png)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not available; skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test with a small query budget**

```sh
uv pip install -e ./python faiss-cpu hnswlib matplotlib
python bench/compare_documents.py --queries 50 --k 5
```

Expected: produces `bench/compare_documents.csv` and `bench/compare_documents_pareto.png`. Recall for `LC FFT` and `faiss.FlatIP` should be 1.000.

- [ ] **Step 3: Commit**

```sh
git add bench/compare_documents.py
git commit -m "phase 6.3: real-document bench harness (LC + Faiss + hnswlib + sklearn)"
```

---

### Task 6.4: Restructure `bench/README.md`

- [ ] **Step 1: Move the headline section to the top**

After the existing `# bench/` heading, before any other content, insert:

```markdown
## Headline: real-document benchmark (arXiv + Jina v2)

The headline workload is ~50 arXiv PDFs chunked at 512 chars and embedded
with [`jinaai/jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
(pinned revision in `bench/data/MODEL_REVISION.txt`). Search is **angular**
(arccos cosine on L2-normalized embeddings).

![Real-document Pareto](compare_documents_pareto.png)

Run with:

```sh
uv venv && source .venv/bin/activate
uv pip install requests pypdf numpy sentence-transformers faiss-cpu hnswlib matplotlib -e ./python
python bench/data/build_corpus.py        # first run: 10–20 min (PDF + model download)
python bench/compare_documents.py        # subsequent runs: ~30 s
```

See [`bench/data/CHUNKING.md`](data/CHUNKING.md) for the chunking rules.
Numbers in the table below come from `bench/compare_documents.csv` (run on the
bench machine; regenerate with the command above).

| Method | Recall@10 | QPS |
|---|---:|---:|
| (populated by `compare_documents.py` — do not hand-edit) | | |

---

## Controlled ablation: uniform-random synthetic workload

```

Then keep the existing in-tree C++ bench + Pareto-by-metric tables under
the "Controlled ablation" heading.

- [ ] **Step 2: Commit**

```sh
git add bench/README.md
git commit -m "phase 6.4: bench README — real-document headline, synthetic moved below"
```

---

### Task 6.5: CI workflow (manual + weekly cron)

- [ ] **Step 1: Create `.github/workflows/bench_documents.yml`**

```yaml
name: Document benchmark

on:
  workflow_dispatch:
  schedule:
    - cron: "0 6 * * 1"  # weekly Monday 06:00 UTC

permissions:
  contents: write   # for committing results to bench-results branch on cron runs

jobs:
  doc-bench:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install uv
        run: pip install uv
      - name: Cache HuggingFace model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-jina-v2-${{ hashFiles('bench/data/MODEL_REVISION.txt') }}
      - name: Cache corpus data
        uses: actions/cache@v4
        with:
          path: bench/data
          key: bench-data-${{ hashFiles('bench/data/arxiv_ids.txt', 'bench/data/MODEL_REVISION.txt') }}
      - name: Install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install requests pypdf numpy sentence-transformers matplotlib faiss-cpu hnswlib
          uv pip install ./python
      - name: Build corpus (cached after first run)
        run: |
          source .venv/bin/activate
          python bench/data/build_corpus.py
      - name: Run document bench
        run: |
          source .venv/bin/activate
          python bench/compare_documents.py --queries 200 --k 10
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: bench-results
          path: |
            bench/compare_documents.csv
            bench/compare_documents_pareto.png
```

- [ ] **Step 2: Commit**

```sh
git add .github/workflows/bench_documents.yml
git commit -m "phase 6.5: CI workflow for document bench (workflow_dispatch + weekly cron)"
```

---

### Task 6.6: Run the full bench, verify gates, update the README table

- [ ] **Step 1: Run end-to-end**

After replacing the placeholder revision SHA and vetting the arXiv IDs:

```sh
python bench/data/build_corpus.py
python bench/compare_documents.py --queries 200 --k 10
```

- [ ] **Step 2: Verify the spec §12.5 gates**

| Gate | Threshold |
|---|---|
| Full LC vs current LC at angular | ≥ 5× QPS, 1T, recall = 1.000 |
| Full LC batched vs current LC at angular | ≥ 8× QPS at HW threads |
| Full LC vs Faiss IVFFlat at angular | match QPS at recall ≥ 0.95 |
| Full LC vs hnswlib at angular | beat at any operating point at recall = 1.000 |
| Bench first-run on clean checkout | ≤ 20 min on a laptop |
| Bench second-run | ≤ 30 s to plot |

For any miss: profile (Instruments on macOS / perf on Linux), fix one or two
obvious bottlenecks, re-run. If still missing, document the negative result in
`bench/README.md` rather than papering over.

- [ ] **Step 3: Update the README table with real numbers**

Hand-paste the relevant rows of `bench/compare_documents.csv` into the
markdown table in `bench/README.md`. (Future work: auto-generate the README
table from the CSV. Out of scope here.)

- [ ] **Step 4: Commit**

```sh
git add bench/README.md
git commit -m "phase 6.6: populate document-bench results table"
```

---

## Phase exit check

```sh
ls bench/compare_documents.csv bench/compare_documents_pareto.png
git log --oneline -7
```

After Phase 6 passes, follow `superpowers:finishing-a-development-branch` to
decide merge / PR / cleanup.
