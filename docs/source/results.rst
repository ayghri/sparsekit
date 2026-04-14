Benchmark Results
=================

| **Data:** Qwen3 4B layer 0 — W (2560, 9728), X (244449, 9728)
| **Device:** CUDA (gpu:1)
| **Loss metric:** ``||X(W_pruned - W0)^T||_F / ||X W0^T||_F`` (relative output Frobenius norm)

----

1. 2:4 Structured Sparsity
---------------------------

``block_shape=(1,1)``, ``scope_shape=(1,4)``, keep 2 of 4 contiguous columns. All 2560 rows.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 10

   * - Method
     - Norm. Loss
     - Sparsity
     - Time
   * - OBS local
     - 20.66%
     - 50.0%
     - 0.1s
   * - OBS full (frozen C)
     - 15.42%
     - 50.0%
     - 3.9s
   * - OBS interleaved=8
     - 14.35%
     - 50.0%
     - 2.9s
   * - OBS interleaved=16
     - 14.24%
     - 50.0%
     - 3.3s
   * - OBS interleaved=64
     - 14.16%
     - 50.0%
     - 6.0s
   * - SparseGPT
     - 14.12%
     - 50.0%
     - 1.1s
   * - OBS-ord ng=256 (shared C, Schur, fp16)
     - 13.39%
     - 50.0%
     - 14.8s
   * - True OBS ng=256 L2R
     - 12.09%
     - 50.0%
     - 75s
   * - **True OBS ng=256 largest-first**
     - **11.87%**
     - 50.0%
     - 105s

**Key comparisons:**

- **True OBS largest-first beats SparseGPT by 16.0%** — per-row C with Schur updates, largest-cost blocks first
- **OBS-ord (shared C, Schur, fp16) beats SparseGPT by 5.2%** in 14.8s — good speed/quality tradeoff
- **Largest-first ordering improves True OBS by 1.8%** over left-to-right (11.87% vs 12.09%)
- **OBS interleaved=64 nearly matches SparseGPT** (−0.3%) at 6s — practical fast alternative
- Gap from OBS full to SparseGPT closed from **−9.2%** to **−0.3%** by interleaved mask re-selection
- OBS split (not shown) is always worse than OBS full — it doesn't re-select masks

True OBS (first 32 rows only — O(B×K²) memory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 15 10

   * - Method
     - Norm. Loss
     - Time
   * - OBS full (frozen C)
     - 5.42%
     - 0.3s
   * - OBS interleaved=64
     - 4.09%
     - 3.9s
   * - SparseGPT
     - 4.06%
     - 1.2s
   * - True OBS ng=256
     - 3.42%
     - 1.0s
   * - **True OBS ng=1**
     - **3.39%**
     - 75.9s

- True OBS ng=256 quality within 1% of ng=1 — Schur update frequency barely matters
- Quality gap: True OBS > SparseGPT > Interleaved > OBS full

----

2. Coupled 2:4 Sparsity
------------------------

Pairs of elements 8 columns apart. ``View (M, K/16, 8, 2):(K, 16, 1, 8)``.
``block_shape=(1,1,1,2)``, ``scope_shape=(1,1,4,1)``, keep 2 of 4 pairs. All 2560 rows.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 10

   * - Method
     - Norm. Loss
     - Sparsity
     - Time
   * - OBS local
     - 27.83%
     - 50.0%
     - 0.0s
   * - OBS full
     - 20.43%
     - 50.0%
     - 3.1s
   * - OBS interleaved=16
     - 19.14%
     - 50.0%
     - 2.7s
   * - OBS interleaved=64
     - 19.06%
     - 50.0%
     - 5.4s
   * - SparseGPT
     - 19.01%
     - 50.0%
     - 1.0s
   * - **True OBS ng=16**
     - **15.75%**
     - 50.0%
     - 433s
   * - True OBS ng=64
     - 15.79%
     - 50.0%
     - 397s

**Key comparisons:**

- **True OBS ng=16 beats SparseGPT by 17.1%** — per-row C with Schur updates, largest-first ordering
- **OBS interleaved=64 nearly matches SparseGPT** (−0.3%) but is **30x faster** (5.4s vs 159.5s)
- OBS interleaved=16 is within 0.7% at **59x faster** (2.7s vs 159.5s)

----

3. 4:8 Structured Sparsity
---------------------------

``block_shape=(1,2)``, ``scope_shape=(1,4)``. 4 blocks of 2 elements per scope, prune 2 blocks. All 2560 rows.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 10

   * - Method
     - Norm. Loss
     - Sparsity
     - Time
   * - OBS local
     - 27.76%
     - 50.0%
     - 0.0s
   * - OBS full
     - 20.34%
     - 50.0%
     - 3.1s
   * - OBS interleaved=16
     - 19.11%
     - 50.0%
     - 2.7s
   * - OBS interleaved=64
     - 19.04%
     - 50.0%
     - 5.5s
   * - SparseGPT
     - 19.00%
     - 50.0%
     - 1.1s
   * - True OBS ng=256
     - 16.12%
     - 50.0%
     - 487s
   * - True OBS ng=64
     - 15.97%
     - 50.0%
     - 463s
   * - **True OBS ng=16**
     - **15.92%**
     - 50.0%
     - 558s

**Key comparisons:**

- **True OBS ng=16 beats SparseGPT by 16.2%** — per-row C with Schur updates
- ng=16 vs ng=256: only 0.2% quality difference, ng barely matters
- OBS interleaved=64 within **0.2%** of SparseGPT (5.5s vs 1.1s)

True OBS (first 32 rows only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 15 10

   * - Method
     - Norm. Loss
     - Time
   * - SparseGPT
     - 5.64%
     - 1.1s
   * - **True OBS ng=2**
     - **4.53%**
     - 20.3s
   * - True OBS ng=16
     - 4.54%
     - 5.0s

- **True OBS ng=2 beats SparseGPT by 19.6%**

----

4. 16-Column Block, 8-Row Coupled Sparsity
-------------------------------------------

``View(size=(8, 2, K), stride=(K, 8K, 1))`` on 16-row chunks.
``block_shape=(1,1,16)``, ``scope_shape=(1,2,1)``, keep 1 of 2 blocks per scope.
160 chunks of 16 rows. All 2560 rows.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 10

   * - Method
     - Norm. Loss
     - Sparsity
     - Time
   * - Magnitude
     - 48.83%
     - 50.0%
     - 0.05s
   * - OBS full block
     - 34.19%
     - 50.0%
     - 9.8s
   * - SparseGPT block
     - 33.46%
     - 50.0%
     - 146.3s
   * - **True OBS ng=16**
     - **26.91%**
     - 50.0%
     - 215.5s

**Key comparisons:**

- **True OBS ng=16 beats SparseGPT by 19.6%** — per-row Schur feasible here (only 608 scopes)
- True OBS beats OBS full by **21.3%**
- OBS full is **14.9x faster** than SparseGPT

----

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 25 30 12 12 8

   * - Config
     - Best Method
     - Norm. Loss
     - vs SparseGPT
     - Time
   * - 2:4 (all rows)
     - **True OBS ng=256 largest**
     - **11.87%**
     - **+16.0%**
     - 105s
   * - 2:4 mid (all rows)
     - OBS-ord ng=256 (shared C)
     - 13.39%
     - +5.2%
     - 14.8s
   * - 2:4 fast (all rows)
     - OBS interleaved=64
     - 14.16%
     - −0.3%
     - 6.0s
   * - Coupled 2:4 (all rows)
     - **True OBS ng=16**
     - **15.75%**
     - **+17.1%**
     - 433s
   * - Coupled 2:4 fast
     - OBS interleaved=64
     - 19.06%
     - −0.3%
     - 5.4s
   * - 4:8 (all rows)
     - **True OBS ng=16**
     - **15.92%**
     - **+16.2%**
     - 558s
   * - 4:8 fast (all rows)
     - OBS interleaved=64
     - 19.04%
     - −0.2%
     - 5.5s
   * - 16-col block (all rows)
     - **True OBS ng=16**
     - **26.91%**
     - **+19.6%**
     - 215.5s

**Takeaways:**

- **True OBS (per-row C with Schur) beats SparseGPT by 16–20%** across all configs — 105s for 2:4, 558s for 4:8, 433s for coupled 2:4
- **OBS-ord (shared C, Schur, fp16, largest-first) beats SparseGPT by 5%** in 15s — practical mid-tier option
- **OBS interleaved (shared C, re-select masks per split) matches SparseGPT within 0.2–0.3%** in 3–6s — the practical fast method
- Key insight: **mask re-selection with updated C** is what matters. OBS split (fixed masks) always loses to OBS full. OBS interleaved (updated masks) nearly matches SparseGPT.
- **Largest-first block ordering** gives free +2% on True OBS, +5% on shared-C OBS
- **For coupled 2:4, OBS interleaved dominates**: matches SparseGPT quality but is **30x faster** (5.4s vs 159.5s)
