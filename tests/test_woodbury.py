import numpy.linalg as LA
import numpy as np

X = np.random.randn(100, 5)
H = X.T @ X / X.shape[0] + 0.1 * np.eye(X.shape[1])
C = LA.inv(H)
print("Original C:\n", C)
col_idx = 2
kept_cols = np.arange(X.shape[1]) != col_idx
pruned_C = (
    C
    - C[:, col_idx : col_idx + 1]
    @ C[col_idx : col_idx + 1, :]
    / C[col_idx, col_idx]
)
print("Pruned C:\n", pruned_C)
pruned_X = X[:, kept_cols]
direct_C = LA.inv(
    pruned_X.T @ pruned_X / pruned_X.shape[0] + 0.1 * np.eye(pruned_X.shape[1])
)
print("Direct C:\n", direct_C)
print(direct_C)
print(LA.norm(pruned_C[kept_cols][:, kept_cols] - direct_C))
