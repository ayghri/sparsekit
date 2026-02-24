# Structured OBS for Arbitrary Block Views, Block Specs, and Group Specs

## 1. Problem Statement

Given a weight matrix `W` of shape `(M, K)` with row-major stride `(K, 1)`, we want to perform **Optimal Brain Surgeon (OBS)** pruning that respects the block and group structure defined by `BlockSpec` and `GroupSpec`.

The key insight: OBS operates on the **input dimension** (columns of `W`), and the Hessian `H = X^T X / N` is a `(K, K)` matrix that captures second-order information about how input features interact. Blocks within the same **group row** share the same slice of the Hessian, so we maintain one Hessian inverse per group row.

### What is a "group row"?

For a linear index `i` into the `(M, K)` matrix:
- **Row index**: `i // K` (which output neuron)
- **Column index**: `i % K` (which input feature)

A **group row** consists of all groups whose blocks share the same set of row indices in `W`. Since OBS updates are applied per-row of `W` (each output neuron's weights are updated independently using the shared Hessian), groups in the same group row share a single Hessian.

## 2. Notation and Setup

### 2.1 Dimensions

| Symbol | Meaning |
|--------|---------|
| `M` | Output dimension (rows of `W`) |
| `K` | Input dimension (columns of `W`) |
| `(bm, bk)` | Block shape: `block_shape` in `BlockSpec` |
| `(Bm, Bk)` | Block grid: `grid_shape = (M//bm, K//bk)` |
| `(gm, gk)` | Group shape: `group_shape` in `GroupSpec` |
| `(Gm, Gk)` | Group grid: `(Bm//gm, Bk//gk)` |

### 2.2 Concrete Example

```
W: (48, 32)          # M=48, K=32
block_shape: (4, 4)  # bm=4, bk=4
grid_shape: (12, 8)  # Bm=12, Bk=8
group_shape: (1, 4)  # gm=1, gk=4
group_grid: (12, 2)  # Gm=12, Gk=2
```

Each group contains `gm * gk = 1 * 4 = 4` blocks.
Each group spans `gm * bm = 4` rows and `gk * bk = 16` columns of `W`.

### 2.3 Data

- `X`: Input data of shape `(N, K)`, where `N` is the number of samples.
- `W`: Weight matrix, `nn.Parameter` of shape `(M, K)`.
- `Y = X @ W.T`: Output of shape `(N, M)`.

## 3. Hessian Structure

### 3.1 Full Hessian

The OBS loss for pruning a weight `w_p` is:

```
delta_L = w_p^2 / (2 * C_pp)
```

where `C = H^{-1}` and `H = X^T X / N` (the `(K, K)` Hessian for the input features).

**Critical observation**: `H` depends only on the input `X`, not on the output dimension. Every row of `W` shares the same `H`. This means:

1. We compute `H` once for the full `(K, K)` space.
2. Each group row can maintain its own `C = H^{-1}` because pruning decisions in one group row don't affect the Hessian of another (they operate on different output rows).

### 3.2 Per-Group-Row Hessians

A **group row** `r` (for `r = 0, ..., Gm-1`) covers output rows `[r*gm*bm, (r+1)*gm*bm)` of `W`.

Within a group row, blocks are organized along the K dimension. A group column `c` covers input columns `[c*gk*bk, (c+1)*gk*bk)`.

However, **we do NOT use a block-diagonal approximation** of `H`. Each group row uses the full `(K, K)` Hessian because blocks across different group columns within the same group row interact through the off-diagonal entries of `H`. The correct OBS update requires the full inverse.

**Per group row `r`**:
```
H_r = H = X^T X / N                  # (K, K), same for all rows
C_r = (H + lambda * I)^{-1}          # (K, K), initially the same for all rows
```

As blocks are pruned in group row `r`, `C_r` is updated via Schur complement, diverging from other rows' `C`.

### 3.3 Block-Level View of C

We can view `C` (shape `(K, K)`) through the block structure. With `Bk = K // bk` block columns:

```
C reshaped as (Bk, bk, Bk, bk) then permuted to (Bk, Bk, bk, bk)
```

The `(i, j)`-th block `C_{ij}` is a `(bk, bk)` matrix representing the interaction between block-column `i` and block-column `j` of `W`.

Similarly, within a group column `c` that spans `gk` block-columns, the relevant submatrix of `C` is:

```
C_group_c = C[c*gk*bk : (c+1)*gk*bk, c*gk*bk : (c+1)*gk*bk]   # (gk*bk, gk*bk)
```

But for the OBS update, we need the **full** `C[:, block_cols]` (cross-group interactions), not just the diagonal block.

## 4. Algorithm Overview

### 4.1 High-Level Loop

```
for each group_row r in [0, Gm):
    C_r = (H + lambda * I)^{-1}           # (K, K)
    W_r = W[r*gm*bm : (r+1)*gm*bm, :]    # (gm*bm, K) -- the rows of W for this group row

    while blocks_remaining_to_prune(r) > 0:
        1. SCORE:  Compute pruning score for each surviving block in this group row
        2. SELECT: Choose the block with the lowest score
        3. UPDATE W: Apply OBS weight compensation to surviving blocks
        4. UPDATE C: Apply Schur complement to remove the pruned block's columns from C_r
        5. ZERO:   Set the pruned block to zero
```

### 4.2 Step 1: Scoring

For a block `b` at block-column position `p` within group row `r`:

- The block covers columns `[p*bk, (p+1)*bk)` in `W`.
- `W_b` has shape `(gm*bm, bk)` -- all the weights in this block across the output rows of the group row.
- `C_{pp}` = `C_r[p*bk:(p+1)*bk, p*bk:(p+1)*bk]` has shape `(bk, bk)`.

**Score (loss degradation from pruning block `b`)**:

```
score_b = Tr[ W_b^T @ W_b @ C_{pp}^{-1} ]
        = sum_j  w_j^T @ C_{pp}^{-1} @ w_j     (summing over output rows j)
```

where `w_j = W_r[j, p*bk:(p+1)*bk]` is the `(bk,)` weight vector for output row `j`, block column `p`.

**Vectorized**: For all `gm*bm` output rows at once:
```python
W_b = W_r[:, p*bk:(p+1)*bk]                    # (gm*bm, bk)
C_pp_inv = torch.linalg.inv(C_pp)               # (bk, bk)
temp = W_b @ C_pp_inv                            # (gm*bm, bk)
score_b = (temp * W_b).sum()                     # scalar = Tr[W_b^T @ W_b @ C_pp_inv]
```

Alternatively, using the trace identity `Tr[A B] = sum(A * B^T)`:
```python
score_b = (temp * W_b).sum()
```

### 4.3 Step 2: Selection

Among all surviving blocks in the group row (across all group columns), select the block with the minimum score:

```python
b_star = argmin_{b in surviving_blocks} score_b
```

**Design choice**: We can either:
1. **Greedy per group row**: Process each group row independently, pruning one block at a time.
2. **Global greedy**: Across all group rows simultaneously, prune the single globally cheapest block, then recompute. (More expensive but potentially better quality.)
3. **Parallel per group row**: Since group rows are independent (different output rows, each with their own `C_r`), process all group rows in parallel, each pruning one block per step.

**Recommended: Option 3** -- parallel per group row. Each group row independently selects and prunes its cheapest block in each iteration.

### 4.4 Step 3: Weight Update (OBS Compensation)

After selecting block `b_star` at block-column position `p`:

For each output row `j` in the group row:
```
w_j_new = w_j - C_r[:, p*bk:(p+1)*bk] @ C_{pp}^{-1} @ w_j[p*bk:(p+1)*bk]
```

where `w_j` is the full `(K,)` weight vector for output row `j`.

**Vectorized over all output rows** (using `W_r` of shape `(gm*bm, K)`):
```python
p_start, p_end = p * bk, (p + 1) * bk

C_col_p = C_r[:, p_start:p_end]                       # (K, bk)
C_pp = C_r[p_start:p_end, p_start:p_end]              # (bk, bk)
C_pp_inv = torch.linalg.inv(C_pp + eps * I)            # (bk, bk)

W_p = W_r[:, p_start:p_end]                            # (gm*bm, bk)

# Update: each row j gets: w_j -= C_col_p @ C_pp_inv @ w_j[p_start:p_end]
# Vectorized: W_r -= W_p @ C_pp_inv^T @ C_col_p^T = W_p @ (C_col_p @ C_pp_inv)^T
update = W_p @ (C_col_p @ C_pp_inv).T                  # (gm*bm, K)
W_r -= update
W_r[:, p_start:p_end] = 0                              # Zero out pruned block
```

### 4.5 Step 4: Hessian Inverse Update (Schur Complement)

After pruning block at position `p`, update `C_r` to reflect that these columns are gone:

```python
C_col_p = C_r[:, p_start:p_end]                        # (K, bk)
C_pp = C_r[p_start:p_end, p_start:p_end]               # (bk, bk)
C_pp_inv = torch.linalg.inv(C_pp + eps * I)             # (bk, bk)
C_row_p = C_r[p_start:p_end, :]                         # (bk, K)

C_r = C_r - C_col_p @ C_pp_inv @ C_row_p               # (K, K)

# Zero out pruned rows/cols for numerical stability
C_r[p_start:p_end, :] = 0
C_r[:, p_start:p_end] = 0
```

This is the **Woodbury / Schur complement** update. After this, `C_r` is the inverse of the Hessian restricted to the surviving columns, with the pruned positions zeroed out.

**Verification**: See `test_woodbury.py` which confirms that this rank-`bk` update matches the direct inverse of the reduced Hessian.

## 5. Integration with BlockSpec and GroupSpec

### 5.1 Using BlockSpec for Block Views

`BlockSpec` provides the block-structured view of `W`:

```python
block = BlockSpec(W, block_shape=(bm, bk))

# block.grid_shape = (Bm, Bk) = (M//bm, K//bk)
# block.block_view(W) gives shape (Bm, Bk, bm, bk) after reorder
#   - First two dims: block grid position
#   - Last two dims: elements within a block
```

### 5.2 Using GroupSpec for Group Views

`GroupSpec` adds grouping on top of blocks:

```python
group = GroupSpec(block, group_shape=(gm, gk))

# group.grid_shape = (Gm, Gk) = (Bm//gm, Bk//gk)
# group.block_to_group(block_norms) reshapes (Bm, Bk) -> (Gm, Gk, gm, gk)
#   after reorder: (Gm, Gk, gm, gk)
```

### 5.3 Mapping to OBS Structures

**Group row `r`** = all groups with `group_grid_row == r`, i.e., groups `(r, 0), (r, 1), ..., (r, Gk-1)`.

These span output rows `[r*gm*bm, (r+1)*gm*bm)` of `W`.

**Within a group `(r, c)`**, there are `gm * gk` blocks. The blocks are indexed by their position within the group: `(i, j)` for `i in [0, gm), j in [0, gk)`.

Block `(i, j)` in group `(r, c)` maps to:
- Block-grid position: `(r*gm + i, c*gk + j)` in the `(Bm, Bk)` grid
- Weight rows: `[(r*gm + i)*bm, (r*gm + i + 1)*bm)`
- Weight columns: `[(c*gk + j)*bk, (c*gk + j + 1)*bk)`

**Hessian column index** for block `(i, j)` in group `(r, c)`:
- Columns `[(c*gk + j)*bk, (c*gk + j + 1)*bk)` in the `(K, K)` Hessian

## 6. Generalization to Arbitrary Block Views

### 6.1 Non-Square Blocks

The algorithm naturally handles `bm != bk`. The key dimensions are:

- **Scoring**: `C_{pp}` is always `(bk, bk)` (determined by the input/column block size).
- **Weight block**: `W_b` is `(bm_effective, bk)` where `bm_effective = gm * bm` is the total number of output rows in the group row.
- **Hessian**: Always `(K, K)`, independent of `bm`.

### 6.2 Multi-Dimensional Blocks

For `BlockSpec` with `rank > 2` (e.g., conv weight `(C_out, C_in, kH, kW)` with `block_shape=(bc_out, bc_in, 1, 1)`):

The OBS formulation requires flattening the weight to a 2D matrix. For a conv layer with weight shape `(C_out, C_in, kH, kW)`:
- Flatten to `(C_out, C_in * kH * kW)`, i.e., `M = C_out`, `K = C_in * kH * kW`
- The input `X` for the Hessian is the unfolded input patches of shape `(N, K)`
- Blocks and groups are then defined on this 2D view

The `block_view` method of `BlockSpec` handles the reshaping. The OBS algorithm only needs the 2D `(M, K)` interpretation.

### 6.3 BlockCoupling

When multiple parameters are coupled (e.g., `W_Q` and `W_K` in attention), `BlockCoupling` merges their block grids via axis permutations. For structured OBS:

- Each coupled parameter contributes blocks to the same group grid
- The Hessian is constructed for each parameter's input independently
- Pruning decisions are made jointly across coupled parameters within each group
- Weight updates are applied to each parameter using its own Hessian

This requires maintaining separate `C` matrices for each coupled parameter but making joint pruning decisions.

## 7. Implementation Plan

### 7.1 Data Structures

```python
@dataclass
class StructuredOBSConfig:
    """Configuration for structured OBS."""
    damp: float = 1e-4            # Hessian damping factor
    block: BlockSpec              # Block specification
    group: GroupSpec              # Group specification
    num_prune: int                # Number of blocks to prune per group (or sparsity target)

class StructuredOBSSolver:
    """Solver for structured OBS pruning."""

    def __init__(self, X: Tensor, block: BlockSpec, group: GroupSpec, damp: float = 1e-4):
        self.block = block
        self.group = group
        self.damp = damp

        # Shapes
        self.Gm = group.grid_shape[0]  # Number of group rows
        self.Gk = group.grid_shape[1]  # Number of group columns
        self.gm = group.group_shape[0] # Blocks per group (row dim)
        self.gk = group.group_shape[1] # Blocks per group (col dim)
        self.bm = block.block_shape[0] # Block height
        self.bk = block.block_shape[1] # Block width
        self.K = block.shape[1]        # Full input dimension

        # Hessian: one C per group row (initially all the same)
        H = X.T @ X / X.shape[0]      # (K, K)
        C = torch.linalg.inv(H + damp * torch.eye(self.K))
        self.C = C.unsqueeze(0).expand(self.Gm, -1, -1).clone()  # (Gm, K, K)

        # Track which blocks are pruned: (Bm, Bk) grid
        self.pruned = torch.zeros(block.grid_shape, dtype=torch.bool)
```

### 7.2 Core Methods

```python
def compute_scores(self) -> Tensor:
    """
    Compute OBS pruning score for every surviving block.

    Returns: (Bm, Bk) tensor of scores, inf for already-pruned blocks.
    """
    W = self.block.param.data                         # (M, K)
    scores = torch.full(self.block.grid_shape, float('inf'))

    for r in range(self.Gm):                          # per group row
        C_r = self.C[r]                               # (K, K)
        row_start = r * self.gm * self.bm
        row_end = (r + 1) * self.gm * self.bm
        W_r = W[row_start:row_end, :]                 # (gm*bm, K)

        for p in range(self.block.grid_shape[1]):     # per block column
            if self.pruned[r * self.gm, p]:           # skip if any block in this column is pruned
                continue

            col_start = p * self.bk
            col_end = (p + 1) * self.bk

            C_pp = C_r[col_start:col_end, col_start:col_end]   # (bk, bk)
            C_pp_inv = torch.linalg.inv(C_pp)                   # (bk, bk)

            W_p = W_r[:, col_start:col_end]                     # (gm*bm, bk)
            temp = W_p @ C_pp_inv                                # (gm*bm, bk)
            score = (temp * W_p).sum()                           # scalar

            # Assign score to all block-grid rows in this group row
            for i in range(self.gm):
                scores[r * self.gm + i, p] = score


def update_weights_and_hessian(self, r: int, p: int):
    """
    After selecting block column p in group row r:
    1. Update W for all output rows in this group row
    2. Update C_r via Schur complement
    3. Zero out the pruned block
    """
    C_r = self.C[r]
    col_start, col_end = p * self.bk, (p + 1) * self.bk
    row_start = r * self.gm * self.bm
    row_end = (r + 1) * self.gm * self.bm

    W = self.block.param.data
    W_r = W[row_start:row_end, :]

    # Weight update
    C_col_p = C_r[:, col_start:col_end]                   # (K, bk)
    C_pp = C_r[col_start:col_end, col_start:col_end]      # (bk, bk)
    C_pp_inv = torch.linalg.inv(C_pp)                      # (bk, bk)

    W_p = W_r[:, col_start:col_end]                        # (gm*bm, bk)
    update = W_p @ (C_col_p @ C_pp_inv).T                  # (gm*bm, K)
    W[row_start:row_end, :] -= update
    W[row_start:row_end, col_start:col_end] = 0

    # Hessian update (Schur complement)
    C_row_p = C_r[col_start:col_end, :]                    # (bk, K)
    self.C[r] -= C_col_p @ C_pp_inv @ C_row_p
    self.C[r][col_start:col_end, :] = 0
    self.C[r][:, col_start:col_end] = 0

    # Mark pruned
    for i in range(self.gm):
        self.pruned[r * self.gm + i, p] = True


def prune(self, num_prune_per_group: int):
    """
    Main pruning loop.

    For each group row, prune num_prune_per_group blocks greedily.
    All group rows are independent and can be parallelized.
    """
    for step in range(num_prune_per_group):
        scores = self.compute_scores()                     # (Bm, Bk)

        # For each group row, find the block column with min score
        # Reshape scores to (Gm, gm, Bk) and take min over gm and Bk
        # Actually: scores are constant within a group row, so just take one representative
        for r in range(self.Gm):
            # Get scores for this group row (all gm block-rows have the same score)
            row_scores = scores[r * self.gm, :]            # (Bk,)
            p_star = torch.argmin(row_scores).item()

            if row_scores[p_star] == float('inf'):
                continue  # All blocks pruned

            self.update_weights_and_hessian(r, p_star)
```

### 7.3 Vectorized Implementation

The per-group-row loop can be fully vectorized:

```python
def compute_scores_vectorized(self) -> Tensor:
    """
    Batch score computation across all group rows and block columns.

    Returns: (Gm, Bk) tensor of scores.
    """
    W = self.block.param.data                              # (M, K)
    bk = self.bk
    Bk = self.block.grid_shape[1]
    gm_bm = self.gm * self.bm

    # Reshape W to (Gm, gm*bm, K)
    W_rows = W.view(self.Gm, gm_bm, self.K)               # (Gm, gm*bm, K)

    scores = torch.full((self.Gm, Bk), float('inf'), device=W.device)

    for p in range(Bk):
        col_s, col_e = p * bk, (p + 1) * bk

        # C_{pp} for all group rows: (Gm, bk, bk)
        C_pp = self.C[:, col_s:col_e, col_s:col_e]         # (Gm, bk, bk)
        C_pp_inv = torch.linalg.inv(C_pp)                   # (Gm, bk, bk)

        # W_p for all group rows: (Gm, gm*bm, bk)
        W_p = W_rows[:, :, col_s:col_e]                     # (Gm, gm*bm, bk)

        # Batched: temp = W_p @ C_pp_inv -> (Gm, gm*bm, bk)
        temp = torch.bmm(
            W_p.view(self.Gm, gm_bm, bk),
            C_pp_inv
        )                                                    # (Gm, gm*bm, bk)

        # Score = sum over (gm*bm, bk) for each group row
        scores[:, p] = (temp * W_p).sum(dim=(1, 2))         # (Gm,)

    # Mask pruned blocks
    # pruned_per_group_row: (Gm, Bk) - take first block row of each group row
    pruned_mask = self.pruned[::self.gm, :]                  # (Gm, Bk)
    scores[pruned_mask] = float('inf')

    return scores
```

### 7.4 Batched Hessian Update

```python
def batch_update(self, selections: Tensor):
    """
    Apply OBS update for all group rows simultaneously.

    selections: (Gm,) tensor of block column indices to prune.
    """
    bk = self.bk
    W = self.block.param.data
    gm_bm = self.gm * self.bm
    W_rows = W.view(self.Gm, gm_bm, self.K)

    for r in range(self.Gm):
        p = selections[r].item()
        if p < 0:
            continue  # Skip if nothing to prune

        col_s, col_e = p * bk, (p + 1) * bk

        # Weight update
        C_col_p = self.C[r, :, col_s:col_e]                # (K, bk)
        C_pp = self.C[r, col_s:col_e, col_s:col_e]         # (bk, bk)
        C_pp_inv = torch.linalg.inv(C_pp)

        W_p = W_rows[r, :, col_s:col_e]                    # (gm*bm, bk)
        update = W_p @ (C_col_p @ C_pp_inv).T              # (gm*bm, K)
        W_rows[r] -= update
        W_rows[r, :, col_s:col_e] = 0

        # Hessian update
        C_row_p = self.C[r, col_s:col_e, :]                # (bk, K)
        self.C[r] -= C_col_p @ C_pp_inv @ C_row_p
        self.C[r, col_s:col_e, :] = 0
        self.C[r, :, col_s:col_e] = 0
```

## 8. Pruning Strategy Within Groups

### 8.1 Per-Group-Row Independence

Since each group row operates on different output rows of `W` and maintains its own `C_r`, group rows are **fully independent**. This means:

1. Each group row can decide independently which blocks to prune.
2. All group rows can be processed in parallel.
3. The number of blocks pruned per group row can vary.

### 8.2 Sparsity Target

Given a target sparsity (e.g., 50% of blocks pruned):

- **Uniform**: Prune the same number of blocks per group: `num_prune = floor(sparsity * gk)` per group.
- **N:M pattern**: Prune exactly `M-N` out of every `M` blocks within each group (e.g., 2:4 means prune 2 out of every 4 blocks).
- **Global budget**: Allow different group rows to prune different amounts, with a total budget across all groups.

### 8.3 Pruning Order

Within each group row, blocks are pruned **greedily**: one at a time, always selecting the block with the lowest score, then updating `W` and `C` before scoring again. This is because:

1. The OBS score depends on `C`, which changes after each pruning.
2. The optimal second block to prune depends on which block was pruned first.
3. Greedy sequential pruning is a good approximation to the combinatorial optimum.

## 9. Memory and Compute Considerations

### 9.1 Memory

- **Hessian inverses**: `Gm` copies of `(K, K)` matrices. For large `K`, this can be significant.
  - Example: `K = 4096, Gm = 1024` -> `1024 * 4096^2 * 4 bytes = 64 GB` (too much!)
  - **Mitigation**: Process group rows sequentially or in small batches, recomputing `C` from `H` for each batch.

### 9.2 Compute

- **Scoring**: `O(Gm * Bk * gm*bm * bk^2)` per step (dominated by the `bk x bk` inverse and matrix multiply).
- **Weight update**: `O(Gm * gm*bm * K * bk)` per step.
- **Hessian update**: `O(Gm * K^2 * bk)` per step (Schur complement is a rank-`bk` update to a `K x K` matrix).
- **Total steps**: `num_prune` per group row, so total is `num_prune * (scoring + update)`.

### 9.3 Optimization: Process Group Rows Sequentially

To avoid storing all `Gm` copies of `C`:

```python
C_template = torch.linalg.inv(H + damp * I)  # (K, K)

for r in range(Gm):
    C_r = C_template.clone()
    W_r = W[r*gm*bm : (r+1)*gm*bm, :]

    for step in range(num_prune):
        scores = score_all_blocks(W_r, C_r)
        p_star = argmin(scores)
        update_W_and_C(W_r, C_r, p_star)
```

Memory: `O(K^2)` for a single `C_r` + `O(K^2)` for the template = `O(K^2)`.

## 10. Connecting Scores to BlockSpec/GroupSpec APIs

### 10.1 Using block_view for Scoring

```python
# Get block view: (Bm, Bk, bm, bk) after reorder
B = block.block_view(W)                         # (Bm, Bk, bm, bk)

# Group the block view: (Gm, Gk, gm, gk, bm, bk)
G = group.block_to_group(B)

# For group row r, group column c, block (i, j) within group:
# Weight block = G[r, c, i, j]  shape (bm, bk)
# Block column index p = c * gk + j
# Block row index = r * gm + i
```

### 10.2 Using block_norms for Initial Scoring (Magnitude Baseline)

```python
# Block L2 norms: (Bm, Bk)
block_norms = block.block_norms()

# Grouped norms: (Gm, Gk, gm*gk)
grouped_norms = group.grouped_block_norms(None)
```

This gives magnitude-based scores. The OBS score replaces `||block||^2` with `Tr[block^T @ block @ C_{pp}^{-1}]`, which accounts for second-order curvature.

### 10.3 Using apply_mask After Pruning

```python
# After determining which blocks to prune:
pruned_mask = torch.zeros(block.grid_shape, dtype=torch.bool)
pruned_mask[pruned_positions] = True
block.apply_mask(pruned_mask)
```

## 11. Edge Cases and Considerations

### 11.1 Numerical Stability

- Always add damping to `H` before inverting: `C = (H + lambda * I)^{-1}`
- Add small `eps * I` to `C_{pp}` before inverting in score/update formulas
- After Schur complement update, zero out the pruned rows/columns of `C` to prevent numerical drift

### 11.2 When gm > 1 (Multi-Row Groups)

When `gm > 1`, multiple block-rows belong to the same group row. The score for a block at position `(r*gm + i, p)` sums contributions from all `gm * bm` output rows in the group row, not just the `bm` rows of that specific block. This is correct because pruning column `p` zeros out that column for ALL output rows in the group row simultaneously.

### 11.3 Interaction Between Group Columns

Blocks in different group columns within the same group row interact through the off-diagonal entries of `C`. Pruning block column `p1` in one group column affects the scores of blocks in other group columns through the Schur complement update of `C`. This is why we use the full `(K, K)` Hessian per group row rather than a block-diagonal approximation per group column.

### 11.4 Relationship to test_block_obs.py

The existing `test_block_obs.py` uses a block-diagonal Hessian approximation (one `(gk*bk, gk*bk)` Hessian per group column). This is a **simplification** that ignores cross-group-column interactions. The full implementation described here uses the complete `(K, K)` Hessian per group row, which is more accurate but more expensive.

## 12. Summary

| Component | Role in Structured OBS |
|-----------|----------------------|
| `BlockSpec` | Defines the block grid, provides `block_view`, `apply_mask`, `block_norms` |
| `GroupSpec` | Defines group rows/columns, provides `block_to_group`, `grouped_block_norms` |
| `H = X^T X / N` | Hessian of the quadratic loss w.r.t. input features |
| `C = H^{-1}` | Inverse Hessian, one per group row (initially shared, diverges after pruning) |
| `score_b` | `Tr[W_b^T @ W_b @ C_{pp}^{-1}]` -- loss degradation from pruning block `b` |
| Weight update | `W_r -= W_p @ (C_col_p @ C_pp_inv)^T` -- OBS compensation |
| Hessian update | `C_r -= C_col_p @ C_pp_inv @ C_row_p` -- Schur complement |
| Pruning loop | Greedy: score -> select min -> update W -> update C -> repeat |
