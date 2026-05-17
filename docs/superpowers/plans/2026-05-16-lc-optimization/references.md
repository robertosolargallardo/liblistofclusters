# Literature references

Citations grounded in the algorithms and benchmarks this plan touches. Each entry notes where in the plan / spec it applies.

## List of Clusters and metric-space indexing

**[CN05] Chávez, E. and Navarro, G. (2005).** *A compact space decomposition for effective metric indexing.* Pattern Recognition Letters 26(9), 1363–1376.
The canonical LC paper. Defines the fixed-bucket-size list, the canonical static and dynamic insert algorithms, and the early-termination invariant. **Applies:** [Phase 0](phase-0-remove-negative-results.md) (canonical insert preserved), [Phase 3](phase-3-fft.md) (FFT center selection follows §3 of the paper).

**[CNBM01] Chávez, E., Navarro, G., Baeza-Yates, R., and Marroquín, J. L. (2001).** *Searching in metric spaces.* ACM Computing Surveys 33(3), 273–321.
Classic survey of metric-space indexing — taxonomy of pivot-based vs partition-based methods, triangle-inequality bounds, dimensionality effects. **Applies:** the framing of why LC sits between AESA-style pivot tables and tree-structured indexes (e.g., M-tree). Background for [Phase 4](phase-4-aesa.md).

**[BNC03] Bustos, B., Navarro, G., and Chávez, E. (2003).** *Pivot selection techniques for proximity searching in metric spaces.* Pattern Recognition Letters 24(14), 2357–2366.
Empirical study of pivot quality criteria — random, farthest-first, "outlier" (high variance of distances to other points). Concludes FFT is competitive and easy to implement; "outlier" wins on harder workloads. **Applies:** [Phase 3](phase-3-fft.md) (center selection) and [Phase 4](phase-4-aesa.md) (anchor selection).

## AESA family (anchor-based pivot tables)

**[V86] Vidal, E. (1986).** *An algorithm for finding nearest neighbours in (approximately) constant average time.* Pattern Recognition Letters 4(3), 145–157.
Original AESA. Precomputes all-pairs distances (O(N²) memory) and uses the triangle inequality to prune candidates with one stored value per anchor. Achieves O(1) expected query time at the cost of quadratic build/storage. **Applies:** [Phase 4](phase-4-aesa.md) — the inspiration for "AESA-lite" with k ≪ N anchors.

**[MOV94] Micó, M. L., Oncina, J., and Vidal, E. (1994).** *A new version of the nearest-neighbour approximating and eliminating search algorithm (AESA) with linear preprocessing-time and memory requirements.* Pattern Recognition Letters 15(1), 9–17.
LAESA: instead of N anchors (full AESA), use k ≪ N anchors. O(Nk) memory, O(Nk) build, O(k + |survivors|·costly_distance) per query. **Applies:** [Phase 4](phase-4-aesa.md) directly — our AESA-lite is a flavor of LAESA with k pinned at construction time and the bound checked per bucket member.

## Farthest-First Traversal (clustering primitive)

**[G85] Gonzalez, T. F. (1985).** *Clustering to minimize the maximum intercluster distance.* Theoretical Computer Science 38, 293–306.
The FFT heuristic — greedy 2-approximation for the k-center problem. Pick a first point arbitrarily; each next point maximizes the minimum distance to already-chosen points. **Applies:** [Phase 3](phase-3-fft.md) (LC center selection) and [Phase 4](phase-4-aesa.md) (anchor selection).

**[HS85] Hochbaum, D. S. and Shmoys, D. B. (1985).** *A best possible heuristic for the k-center problem.* Mathematics of Operations Research 10(2), 180–184.
Proves Gonzalez's algorithm is 2-approximate and that no polynomial-time (2 − ε)-approximation exists unless P = NP. **Applies:** justifies why FFT is "good enough" without trying for an exact k-center solver.

## Permutation-based and alternative metric indexes

**[CFN08] Chávez, E., Figueroa, K., and Navarro, G. (2008).** *Effective proximity retrieval by ordering permutations.* IEEE TPAMI 30(9), 1647–1658.
Permutation index — each point is summarized by the permutation of pivots ordered by distance. Comparing permutations (Spearman footrule or Kendall tau) cheaply prefilters candidates. Trades exactness for substantial speedup. **Applies:** out-of-scope for this work (we stay exact), but cited as a future direction once approximate modes are on the table.

**[CPZ97] Ciaccia, P., Patella, M., and Zezula, P. (1997).** *M-tree: An efficient access method for similarity search in metric spaces.* Proc. VLDB 1997.
M-tree — balanced tree of "covering" balls, supports insertion/deletion. Closest cousin to LC; M-tree organizes hierarchically, LC keeps a flat list. **Applies:** comparator context for the bench README; explains why LC is preferred for streaming workloads with simple deletion.

**[BK73] Burkhard, W. A. and Keller, R. M. (1973).** *Some approaches to best-match file searching.* Communications of the ACM 16(4), 230–236.
BK-trees — the earliest practical use of the triangle inequality for similarity search (discrete metric, edit distance). **Applies:** historical reference for why triangle inequality is the central building block of every metric-space index in this plan.

## Modern ANN systems (comparators)

**[MY18] Malkov, Y. A. and Yashunin, D. A. (2018).** *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI 42(4), 824–836 (arXiv:1603.09320).
HNSW — layered proximity graph with logarithmic search via skip-list-like long links. State-of-the-art approximate NN for dense vectors. **Applies:** comparator in [Phase 5](phase-5-bench-ablation.md) and [Phase 6](phase-6-document-bench.md) (hnswlib + Faiss `IndexHNSWFlat`).

**[JDJ17] Johnson, J., Douze, M., and Jégou, H. (2019).** *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data 7(3), 535–547 (arXiv:1702.08734).
Faiss paper. Covers `IndexFlatL2` (SIMD brute force), `IndexIVFFlat` (inverted file + flat residuals), `IndexPQ` (product quantization), and GPU kernels. The SIMD brute force is the key diagnostic for this plan: at D=8 it beats our pre-P1 LC, telling us per-distance cost dominates pruning quality. **Applies:** comparator throughout; the diagnostic motivates [Phase 1](phase-1-centers-soa.md) and [Phase 2](phase-2-simd.md).

**[JDS11] Jégou, H., Douze, M., and Schmid, C. (2011).** *Product quantization for nearest neighbor search.* IEEE TPAMI 33(1), 117–128.
Product quantization — splits a vector into M subvectors, learns a codebook per subspace via k-means. Reduces memory and inner-loop cost for ANN. **Applies:** background for why Faiss IVFFlat (no PQ) is the right exact-side comparator vs Faiss IVF-PQ (approximate).

**[GSL+20] Guo, R., Sun, P., Lindgren, E., et al. (2020).** *Accelerating large-scale inference with anisotropic vector quantization.* Proc. ICML 2020 (arXiv:1908.10396).
ScaNN — anisotropic VQ that's loss-aware for the cosine/IP target. State-of-the-art for cosine ANN at the time of writing. **Applies:** noted as a comparator option; not bundled by default because the python wheel is heavier than Faiss/hnswlib.

## Embeddings and the document benchmark

**[GMW+23] Günther, M., Mohr, I., Williams, B., et al. (2023).** *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.* arXiv:2310.19923.
Jina v2 — 768-dim sentence embeddings, supports up to 8192 tokens. The `jinaai/jina-embeddings-v2-base-en` model used for the document benchmark. **Applies:** [Phase 6](phase-6-document-bench.md) directly.

**[RG19] Reimers, N. and Gurevych, I. (2019).** *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* Proc. EMNLP-IJCNLP 2019.
Sentence-BERT — the framework Jina v2 fine-tunes through. Justifies using cosine similarity on the encoder outputs (and therefore `metric::angular` on the LC side). **Applies:** [Phase 6](phase-6-document-bench.md) — explains why the natural metric for these embeddings is angular, not Euclidean.

**[X+22] Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. (2023).** *C-Pack: Packed Resources for General Chinese Embeddings.* arXiv:2309.07597.
BGE family (`BAAI/bge-*`). One of the alternate embedders [Phase 6](phase-6-document-bench.md) can use if Jina v2 is unavailable.

## SIMD and inner-loop performance

**[F+18] Faraj, A., Lemire, D., et al. (2018).** *Roaring bitmaps: Implementation of an optimized software library.* SP&E 48(4), 867–895.
Cited as exemplar of carefully designed SIMD inner loops for AVX2 and NEON. **Applies:** [Phase 2](phase-2-simd.md) — the kernel patterns (load, FMA, horizontal reduce) follow the standard template.

**Intel Architecture Software Developer's Manual** (current revision). AVX2 / FMA intrinsics reference.
**ARM Architecture Reference Manual.** NEON / Advanced SIMD intrinsics reference.

## Diversity-promoting alternatives explored and rejected

These are documented here so future contributors don't re-relitigate them.

**[BFK+12] Brisaboa, N. R., Fariña, A., Kossmann, D., et al. (2012).** *Reducing the size of the document inverted index.* DASFAA Workshops 2012.
LC variants with sketches / sub-byte representations. Promising in principle for memory-tight regimes; out-of-scope here because the plan optimizes for QPS, not bytes.

**HC tree (hierarchical LC), Phase 5.8 ancestor.** Tried in the codebase as a tree of LCs. Negative result: branching overhead dominated cluster-list scan savings at our N. The two-tier block index (Phase 5.8) was a simpler attempt at the same idea — also a negative result, removed in [Phase 0](phase-0-remove-negative-results.md) of this plan.

**Per-cluster pivots (Phase 5.6 ancestor).** One pivot per cluster, used alongside the centroid for triangle-inequality bounds. Negative result documented at the time it landed. Subsumed by AESA-lite (a single global anchor table is strictly more powerful per anchor than per-cluster pivot bookkeeping). Removed in [Phase 0](phase-0-remove-negative-results.md).
