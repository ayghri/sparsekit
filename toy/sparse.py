import torch
import torch.linalg as LA

device = torch.device("cuda")
lamb = 1e-4
num_samples = 64
m = 56
k = 64


X = torch.randn(num_samples, k, device=device)
W = torch.randn(m, k, device=device)
Y = X @ W.T
H = (X.T@X)/num_samples + lamb * torch.eye(k).to(X)
C = LA.inv(H)


# our goal here is to some kind of OBS, but with structure
# first, we take the tensor W, but with custom layout
# the idea here is that we put each 2 elements that are away by 8 step in k in the same block
W_l = torch.as_strided(W, (56, 4, 8, 2), (k, 16, 1, 8))

# We now reshape W_l, so that each 4 rows of the original matrix are in the same group

W_l = W_l.reshape(14, 4, 4, 8, 2)

# Now, we need to apply OBS, so that  W_l[i,:,m,0:4] and  W_l[i,:,m,4:]have exactly 2 x (1,4,1,1,2) non zero
# blocks, that is W_l[i,j,m,0:4].norm(-1) and  W_l[i,j,m,4:].norm(-1) each have exactly 2 non zeros
# We need to use OBS, where we have one matrix C per W_l[i] slice,
# iteratively find the best (1,4,1,1,2) block to remove, and update C for that slice
# we also need the implementation to support any arbitrary shape



