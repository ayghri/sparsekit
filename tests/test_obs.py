import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------
# Model: Linear -> affine RMSNorm -> ReLU -> Linear
# -----------------------------
class RMSNormAffine(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, y):
        s = torch.sqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y / s) * self.gamma + self.beta

class SmallNet(nn.Module):
    def __init__(self, in_dim=784, hidden=16, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.rms = RMSNormAffine(hidden, eps=1e-6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.fc1(x)
        z = self.rms(y)
        h = self.act(z)
        logits = self.fc2(h)
        return logits

# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def eval_acc_loss(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += ce(logits, yb).item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total, loss_sum / total

# -----------------------------
# Build exact Gauss–Newton Hessian for phi=[vec(W1); b1]
# RMG-OBS: uses affine RMSNorm Jacobian metric G(y)=J_y^T J_y
# Vanilla-OBS: uses G=I (linear-output objective)
# -----------------------------
def build_H_rms_fc1(model, X_samples, device):
    """
    Exact H = E[ [J_theta^T J_theta, J_theta^T J_b; J_b^T J_theta, J_b^T J_b] ]
    for z = gamma*(y/s) + beta, y = W1 x + b1, phi=[vec(W1); b1].
    """
    model.eval()
    fc1 = model.fc1
    gamma = model.rms.gamma.detach().to(device)
    eps = model.rms.eps

    W1 = fc1.weight.detach().to(device)  # (d,n)
    b1 = fc1.bias.detach().to(device)    # (d,)
    d, n = W1.shape
    dn = d * n

    Y = X_samples @ W1.t() + b1  # (T,d)

    H_tt = torch.zeros(dn, dn, device=device)
    H_tb = torch.zeros(dn, d, device=device)
    H_bb = torch.zeros(d, d, device=device)

    I = torch.eye(d, device=device)
    Dg = torch.diag(gamma)

    T = X_samples.shape[0]
    for t in range(T):
        x = X_samples[t]  # (n,)
        y = Y[t]          # (d,)

        s = torch.sqrt(y.pow(2).mean() + eps)     # scalar
        u = (y / s).view(d, 1)                    # (d,1)
        P = I - (u @ u.t()) / d                   # (d,d)
        J = (Dg @ P) / s                          # (d,d)
        G = J.t() @ J                             # (d,d)

        xxT = torch.outer(x, x)                   # (n,n)
        H_tt += torch.kron(G, xxT)                # (dn,dn)

        # H_tb = E[(I_d ⊗ x) G]
        for i_out in range(d):
            H_tb[i_out*n:(i_out+1)*n, :] += torch.outer(x, G[i_out, :])

        H_bb += G

    H_tt /= T
    H_tb /= T
    H_bb /= T

    H = torch.zeros(dn + d, dn + d, device=device)
    H[:dn, :dn] = H_tt
    H[:dn, dn:] = H_tb
    H[dn:, :dn] = H_tb.t()
    H[dn:, dn:] = H_bb
    return H

def build_H_vanilla_fc1(model, X_samples, device):
    """
    Vanilla linear OBS objective on y: 1/2 E||delta y||^2
    => G = I_d.
    """
    fc1 = model.fc1
    W1 = fc1.weight.detach().to(device)
    d, n = W1.shape
    dn = d * n

    T = X_samples.shape[0]
    Ex = X_samples.mean(dim=0)                    # (n,)
    ExxT = (X_samples.t() @ X_samples) / T        # (n,n)

    H = torch.zeros(dn + d, dn + d, device=device)
    H[:dn, :dn] = torch.kron(torch.eye(d, device=device), ExxT)
    H[dn:, dn:] = torch.eye(d, device=device)

    # cross term: E[(I_d ⊗ x)] = I_d ⊗ Ex
    for i_out in range(d):
        H[i_out*n:(i_out+1)*n, dn + i_out] = Ex
    H[dn:, :dn] = H[:dn, dn:].t()
    return H

# -----------------------------
# Greedy single-parameter OBS pruning (weights only), with inverse updates
# -----------------------------
def obs_prune_phi(H, phi, d_bias, prune_count, damp=1e-3):
    """
    phi = [vec(W1); b1]  (size m=dn+d)
    Prune weights only (first dn entries) using greedy min score w_p^2 / diag(H^{-1})_pp.
    Uses exact single-parameter OBS update and updates H^{-1} by rank-1 downdate.
    """
    m = H.shape[0]
    dn = m - d_bias

    # damping
    lam = damp * torch.trace(H) / m
    Hinv = torch.inverse(H + lam * torch.eye(m, device=H.device))

    phi = phi.clone()
    pruned = torch.zeros(m, dtype=torch.bool, device=H.device)
    eligible = torch.zeros(m, dtype=torch.bool, device=H.device)
    eligible[:dn] = True  # weights only

    diagHinv = torch.diag(Hinv)
    for _ in range(prune_count):
        idxs = torch.where(eligible & (~pruned))[0]
        diag = diagHinv[idxs].clamp_min(1e-12)
        scores = phi[idxs].pow(2) / diag
        p = idxs[torch.argmin(scores)].item()

        Hinv_pp = Hinv[p, p].clamp_min(1e-12)
        col = Hinv[:, p].clone()

        # OBS update: phi <- phi - (phi_p/Hinv_pp) * Hinv[:,p], then enforce phi_p=0
        phi = phi - (phi[p] / Hinv_pp) * col
        phi[p] = 0.0

        # inverse update (delete-style / infinite penalty)
        Hinv = Hinv - torch.outer(col, Hinv[p, :]) / Hinv_pp

        pruned[p] = True
        Hinv[p, :] = 0.0
        Hinv[:, p] = 0.0
        Hinv[p, p] = 1.0
        diagHinv = torch.diag(Hinv)

    return phi

def set_fc1_from_phi(model, phi):
    d, n = model.fc1.weight.shape
    dn = d * n
    with torch.no_grad():
        model.fc1.weight.copy_(phi[:dn].view(d, n))
        model.fc1.bias.copy_(phi[dn:dn+d])

# -----------------------------
# Main experiment
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # MNIST
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(Subset(train_ds, range(20000)), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    hidden = 16  # keep small-ish so H is manageable: (dn+d)=(16*784+16)=12560
    model = SmallNet(hidden=hidden).to(device)

    # Train
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(2):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()

    base_acc, base_loss = eval_acc_loss(model, test_loader, device)

    # Samples for Hessian estimation (from train)
    X_samp = []
    for xb, _ in train_loader:
        X_samp.append(xb.view(xb.size(0), -1))
        if sum(t.size(0) for t in X_samp) >= 256:
            break
    X_samp = torch.cat(X_samp, dim=0)[:256].to(device)

    # Build Hessians
    H_rms = build_H_rms_fc1(model, X_samp, device)
    H_lin = build_H_vanilla_fc1(model, X_samp, device)

    # Phi
    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    phi0 = torch.cat([W1.reshape(-1), b1])

    # Prune K weights in fc1
    K = 200  # adjust
    phi_rms = obs_prune_phi(H_rms, phi0, d_bias=hidden, prune_count=K, damp=1e-3)
    phi_lin = obs_prune_phi(H_lin, phi0, d_bias=hidden, prune_count=K, damp=1e-3)

    # Evaluate RMG-OBS
    model_rms = SmallNet(hidden=hidden).to(device)
    model_rms.load_state_dict(model.state_dict())
    set_fc1_from_phi(model_rms, phi_rms)
    acc_rms, loss_rms = eval_acc_loss(model_rms, test_loader, device)

    # Evaluate Vanilla-OBS
    model_lin = SmallNet(hidden=hidden).to(device)
    model_lin.load_state_dict(model.state_dict())
    set_fc1_from_phi(model_lin, phi_lin)
    acc_lin, loss_lin = eval_acc_loss(model_lin, test_loader, device)

    print(f"Baseline    acc={base_acc:.4f}  loss={base_loss:.4f}")
    print(f"RMG-OBS     acc={acc_rms:.4f}  loss={loss_rms:.4f}   (pruned {K} fc1 weights)")
    print(f"Vanilla OBS acc={acc_lin:.4f}  loss={loss_lin:.4f}   (pruned {K} fc1 weights)")

if __name__ == "__main__":
    main()
