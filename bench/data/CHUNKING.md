# Chunking rules (bench corpus)

- **Chunk size:** 512 characters per chunk.
- **Overlap:** 64 characters between consecutive chunks.
- **Page-level pre-filter:** drop pages with < 200 non-whitespace characters
  (figure-only pages or pages that are mostly references).
- **Reference strip:** regex-strip the "References" section onwards
  (case-insensitive line match on "References" alone).
- **Dedup:** skip chunks whose first 64 characters match an earlier chunk
  byte-for-byte (kills boilerplate / repeated paper headers).
- **Per-paper cap:** at most 200 chunks per paper to keep the corpus
  balanced across sources.

These rules are implemented in `bench/data/build_corpus.py`. The exact
chunk count depends on paper lengths; for the default 15-paper starter
set we get roughly 2000-3000 chunks.
