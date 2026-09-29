"""
The WaveNet model from makemore_rnn.ipynb, written with real PyTorch modules.

Mapping from the hand-rolled version:
    Embedding(27, n_emb)                  -> nn.Embedding
    FlatenConsecutive(2) + Linear(...)    -> nn.Conv1d(in, out, kernel_size=2, stride=2)
    BatchNorm1D                           -> nn.BatchNorm1d   (expects channels in dim 1)
    Tanh                                  -> nn.Tanh
    layer.training = False                -> model.eval()
    manual p.data += -lr * p.grad         -> optim.AdamW + a scheduler
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1337)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DATA = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------- data
BLOCK = 8
words = open(DATA / "names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB = len(stoi)

def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * BLOCK
        for ch in w + ".":
            X.append(ctx); Y.append(stoi[ch])
            ctx = ctx[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

import random; random.seed(32); random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build(words[:n1])
Xdev, Ydev = build(words[n1:n2])
Xte, Yte = build(words[n2:])

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True, drop_last=True)

# ---------------------------------------------------------------- model
class WaveNet(nn.Module):
    """Stacked strided causal convs. Each block halves the time dimension,
    so log2(block_size) blocks collapse the context to a single vector."""

    def __init__(self, vocab=VOCAB, n_emb=24, n_hidden=128, block_size=BLOCK, kernel=2):
        super().__init__()
        n_blocks = round(math.log(block_size, kernel))
        assert kernel ** n_blocks == block_size, "block_size must be kernel ** n_blocks"

        self.emb = nn.Embedding(vocab, n_emb)
        chans = [n_emb] + [n_hidden] * n_blocks
        layers = []
        for i in range(n_blocks):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=kernel, stride=kernel, bias=False),
                nn.BatchNorm1d(chans[i + 1]),
                nn.Tanh(),
            ]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, vocab)
        with torch.no_grad():                     # keep the first logits near uniform
            self.head.weight *= 0.1
            self.head.bias.zero_()

    def forward(self, idx):                       # idx: (B, T) int64
        x = self.emb(idx).transpose(1, 2)         # (B, C, T)  channels first for conv
        x = self.trunk(x)                         # (B, H, 1)
        return self.head(x.squeeze(-1))           # (B, vocab)

# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def split_loss(model, X, Y, bs=8192):
    model.eval()                                  # switches BatchNorm to running stats
    total = 0.0
    for i in range(0, X.shape[0], bs):
        xb, yb = X[i:i + bs].to(DEV), Y[i:i + bs].to(DEV)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
    model.train()
    return total / X.shape[0]

def train(model, epochs=20, lr=3e-3, wd=0.01):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.15)
    hist = []
    for ep in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        d = split_loss(model, Xdev, Ydev)
        hist.append(d)
        print(f"epoch {ep+1:2d}/{epochs}  train(batch) {loss.item():.3f}  dev {d:.4f}  lr {sched.get_last_lr()[0]:.5f}")
    return hist

def save(model, opt, path="wavenet.pt"):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)

def load(path="wavenet.pt"):
    ckpt = torch.load(path, map_location=DEV)
    m = WaveNet().to(DEV); m.load_state_dict(ckpt["model"])
    return m

@torch.no_grad()
def sample(model, n=10):
    model.eval()
    out = []
    for _ in range(n):
        ctx, name = [0] * BLOCK, []
        while True:
            probs = F.softmax(model(torch.tensor([ctx], device=DEV)), dim=1)
            ix = torch.multinomial(probs, 1).item()
            if ix == 0: break
            name.append(itos[ix]); ctx = ctx[1:] + [ix]
        out.append("".join(name))
    model.train()
    return out

if __name__ == "__main__":
    model = WaveNet().to(DEV)
    print(model)
    print(f"\n{sum(p.numel() for p in model.parameters()):,} parameters | device {DEV}\n")
    train(model)
    print(f"\ntrain {split_loss(model, Xtr, Ytr):.4f}   dev {split_loss(model, Xdev, Ydev):.4f}")
    print("samples:", ", ".join(sample(model, 12)))
