They are **not** justified to use (U) as a drop-in replacement for (H^{-1}) *entrywise*. That would be mathematically incorrect (and the papers’ notation encourages exactly that misreading). What *is* justified is narrower:

* you can use a **Cholesky factor (U)** of (H^{-1}) to **carry out the same Gaussian-elimination / Schur-complement sequence** that would otherwise require explicitly updating (H^{-1}) after each eliminated dimension.

That is “correct”, but it is **not** the same statement as “(U) equals (H^{-1})”.

### 1) What is incorrect

If (H^{-1} = U^\top U) (upper-triangular (U)), then in general:

* ((H^{-1})*{ij} \neq U*{ij})
* ((H^{-1})*{ii} \neq U*{ii})
* ((H^{-1})*{ii} \neq U*{ii}^2)

So any step that literally substitutes
$$
(H^{-1})*{ii}\ \leftarrow\ U*{ii}
$$
is wrong.

### 2) What is correct (the actual justification)

The correct justification is: **the elimination update on (H^{-1}) corresponds to taking a trailing submatrix of the Cholesky factor**.

Let (K = H^{-1}) and assume you eliminate index 1 (after permutation). Partition:
$$
K=\begin{bmatrix} a & b^\top\ b & C\end{bmatrix}.
$$
The exact Schur update for the *inverse* after eliminating that index is:
$$
K_{\text{new}} = C - \frac{1}{a} b b^\top.
$$

Now write (K = U^\top U) with
$$
U=\begin{bmatrix} u_{11} & u_{12}\ 0 & U_{22}\end{bmatrix}.
$$
Then (algebraically, always):

* (a = u_{11}^2)
* (b = u_{11}u_{12}^\top)
* (C = u_{12}^\top u_{12} + U_{22}^\top U_{22})

Plugging into the Schur update gives:
$$
K_{\text{new}}
= \left(u_{12}^\top u_{12} + U_{22}^\top U_{22}\right) - \frac{1}{u_{11}^2}(u_{11}u_{12}^\top)(u_{11}u_{12})
= U_{22}^\top U_{22}.
$$

**So the updated inverse is represented exactly by the trailing block (U_{22}).** No recomputation, no explicit Schur operation. This is the mathematically correct “use (U) instead of repeatedly updating (H^{-1})” statement.

### 3) Where GPTQ/SparseGPT notation goes off the rails

They often:

* compute (U = \text{chol}(H^{-1})),
* then overwrite the symbol “(H^{-1})” with (U),
* and keep writing formulas that look like they are indexing (H^{-1}).

Those formulas are only correct if you reinterpret them as:

* “use the corresponding *row/pivot information in the factor* needed for the elimination sweep,” not
* “use the entries of the inverse.”

Example of the only legitimate relationship between the inverse row and the factor row (for the active pivot):
$$
\text{row}*1(K) = u*{11},\text{row}*1(U).
$$
So (U) stores the same “row information” but normalized by (\sqrt{K*{11}}=u_{11}). That matches the papers’ “minor difference is (\sqrt{(\cdot)}) normalization” remark, but they don’t keep the notation clean.

### 4) Practical bottom line

* If you want to be mathematically strict: **do not replace ((H^{-1})*{ii}) by (U*{ii})**.
* What is strict and correct: **using (U) to drive the same elimination sequence**, because removing an index corresponds to dropping the first row/column of (U) (taking the trailing block).
* The algorithmic updates in GPTQ/SparseGPT are correct *as an elimination procedure on the factor*, but the *paper notation* is misleading because it suggests elementwise equivalence between (H^{-1}) and (U).

If you point to a specific equation/line in GPTQ (page 5) you think is “using (U) like (H^{-1})”, I can rewrite that exact line in a mathematically correct form (keeping the same computation but fixing the symbols and scalings).
