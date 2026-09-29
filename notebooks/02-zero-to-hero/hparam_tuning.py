"""
Hyperparameter tuning experiments on the makemore MLP (Karpathy-style).
Every experiment reuses the same train() function; only the knobs change.

Run:  python3 hparam_tuning.py
Out:  plots/*.png
"""
import math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(8)

# paths are resolved from this file, so the script runs from any cwd
DATA = Path(__file__).resolve().parents[2] / "data"
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
words = open(DATA / "names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
itos = {i + 1: s for i, s in enumerate(chars)}
stoi["."] = 0
itos[0] = "."
VOCAB = 27
BLOCK = 3

random.seed(32)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))

def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * BLOCK
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx); Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build(words[:n1])
Xdev, Ydev = build(words[n1:n2])
Xte, Yte = build(words[n2:])
Xsm, Ysm = build(words[:500])          # small-data regime: 500 names only
print(f"train {Xtr.shape[0]}  dev {Xdev.shape[0]}  test {Xte.shape[0]}  small {Xsm.shape[0]}")

# ---------------------------------------------------------------- model
def make_model(n_emb=10, n_hidden=200, seed=234):
    g = torch.Generator().manual_seed(seed)
    fan_in = n_emb * BLOCK
    C  = torch.randn((VOCAB, n_emb), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g) * (5 / 3) / fan_in**0.5  # kaiming for tanh
    b1 = torch.zeros(n_hidden)
    W2 = torch.randn((n_hidden, VOCAB), generator=g) * 0.01
    b2 = torch.zeros(VOCAB)
    ps = [C, W1, b1, W2, b2]
    for p in ps: p.requires_grad = True
    return ps

def forward(ps, X):
    C, W1, b1, W2, b2 = ps
    emb = C[X].view(X.shape[0], -1)
    h = torch.tanh(emb @ W1 + b1)
    return h @ W2 + b2

@torch.no_grad()
def loss_on(ps, X, Y, cap=None):
    if cap and X.shape[0] > cap:
        X, Y = X[:cap], Y[:cap]
    return F.cross_entropy(forward(ps, X), Y).item()

def train(ps, steps, lr, batch=32, lr_schedule=None, eval_every=None, seed=1337, data=None):
    """lr_schedule(i, steps) -> lr overrides the constant lr. Returns history dict."""
    X, Y = data if data else (Xtr, Ytr)
    g = torch.Generator().manual_seed(seed)
    hist = {"step": [], "loss": [], "lr": [], "eval_step": [], "train": [], "dev": []}
    for i in range(steps):
        ix = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = F.cross_entropy(forward(ps, X[ix]), Y[ix])
        for p in ps: p.grad = None
        loss.backward()
        cur = lr_schedule(i, steps) if lr_schedule else lr
        for p in ps: p.data += -cur * p.grad
        hist["step"].append(i); hist["loss"].append(loss.item()); hist["lr"].append(cur)
        if eval_every and (i % eval_every == 0 or i == steps - 1):
            hist["eval_step"].append(i)
            hist["train"].append(loss_on(ps, X, Y, cap=30000))
            hist["dev"].append(loss_on(ps, Xdev, Ydev))
    return hist

def ema(xs, alpha=0.02):
    out, m = [], xs[0]
    for x in xs:
        m = (1 - alpha) * m + alpha * x
        out.append(m)
    return out

# ---------------------------------------------------------------- plot style
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "lines.linewidth": 2, "figure.dpi": 150,
})
def finish(ax, title, sub=None, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", pad=14 if sub else 8)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)

t0 = time.time()

# ================================================================ 1. LR range test
print("[1] lr range test")
ps = make_model()
lo, hi, N = -4, 0.3, 1500
sched = lambda i, n: 10 ** (lo + (hi - lo) * i / n)
h = train(ps, N, lr=None, lr_schedule=sched)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(h["lr"], ema(h["loss"], 0.05), color=S1)
best = min(range(len(h["loss"])), key=lambda i: ema(h["loss"], 0.05)[i])
ax.axvline(h["lr"][best], color=S2, ls="--", lw=1.5)
ax.annotate(f"minimum  lr≈{h['lr'][best]:.2f}", (h["lr"][best], 4.05), color=S2,
            fontsize=9, ha="right", xytext=(-8, 0), textcoords="offset points")
ax.set_xscale("log"); ax.set_ylim(2.2, 4.2)
finish(ax, "1. LR range test", "one run, lr swept 1e-4 → 2 over 1500 steps. Pick ~3-10x below the minimum.",
       "learning rate (log)", "smoothed batch loss")
fig.tight_layout(); fig.savefig(PLOTS / "01_lr_range.png"); plt.close(fig)

# ================================================================ 2. constant LRs
print("[2] lr comparison")
lrs = [0.003, 0.03, 0.1, 0.5, 3.0]
cols = [S1, S2, S3, S4, S5]
fig, ax = plt.subplots(figsize=(7, 4.2))
for lr, c in zip(lrs, cols):
    h = train(make_model(), 6000, lr=lr)
    y = ema(h["loss"], 0.01)
    ax.plot(h["step"], y, color=c, label=f"lr = {lr}")
    print(f"   lr={lr:<6} final smoothed {y[-1]:.3f}")
ax.set_ylim(2.0, 4.5); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.annotate("lr = 3.0 diverges off-chart", (300, 4.35), color=S5, fontsize=9)
finish(ax, "2. What each learning rate does", "same model & seed, 6000 steps, batch 32",
       "step", "smoothed train loss")
fig.tight_layout(); fig.savefig(PLOTS / "02_lr_curves.png"); plt.close(fig)

# ================================================================ 3. capacity vs generalization
print("[3] capacity sweep, two data regimes")
sizes = [10, 30, 100, 300, 1000]
res = {}
for tag, data, cap in [("full (182k examples)", (Xtr, Ytr), 60000),
                       ("small (3.5k examples)", (Xsm, Ysm), None)]:
    tr_l, dv_l = [], []
    for n in sizes:
        ps = make_model(n_hidden=n)
        train(ps, 12000, lr=0.1, lr_schedule=lambda i, s: 0.1 if i < 0.7 * s else 0.01, data=data)
        tr_l.append(loss_on(ps, *data, cap=cap)); dv_l.append(loss_on(ps, Xdev, Ydev))
        print(f"   {tag:22} hidden={n:<5} train {tr_l[-1]:.3f}  dev {dv_l[-1]:.3f}")
    res[tag] = (tr_l, dv_l)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (tag, (tr_l, dv_l)) in zip(axes, res.items()):
    ax.plot(sizes, tr_l, color=S1, marker="o", ms=6, label="train")
    ax.plot(sizes, dv_l, color=S2, marker="o", ms=6, label="dev")
    ax.fill_between(sizes, tr_l, dv_l, color=S2, alpha=0.08)
    ax.set_xscale("log"); ax.legend(frameon=False, fontsize=9)
    finish(ax, f"3. Capacity — {tag}", "12000 steps each, lr 0.1 then 0.01",
           "hidden units (log)", "cross-entropy loss")
axes[0].text(0.03, 0.06, "no gap: data >> parameters", transform=axes[0].transAxes,
             fontsize=9, color=INK2)
axes[1].text(0.03, 0.06, "gap widens with capacity = overfitting", transform=axes[1].transAxes,
             fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig(PLOTS / "03_capacity.png"); plt.close(fig)

# ================================================================ 4. train vs dev over time
print("[4] train/dev curves, two data regimes")
runs = []
for tag, data in [("full (182k examples)", (Xtr, Ytr)), ("small (3.5k examples)", (Xsm, Ysm))]:
    ps = make_model(n_hidden=1000, n_emb=24)
    h = train(ps, 25000, lr=0.1, eval_every=500, data=data)
    runs.append((tag, h))
    k = min(range(len(h["dev"])), key=lambda i: h["dev"][i])
    print(f"   {tag:22} best dev {h['dev'][k]:.3f} @ step {h['eval_step'][k]}  final dev {h['dev'][-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (tag, h) in zip(axes, runs):
    ax.plot(h["eval_step"], h["train"], color=S1, label="train")
    ax.plot(h["eval_step"], h["dev"], color=S2, label="dev")
    k = min(range(len(h["dev"])), key=lambda i: h["dev"][i])
    ax.axvline(h["eval_step"][k], color=INK2, ls="--", lw=1)
    ax.annotate(f"best dev {h['dev'][k]:.3f}\n@ step {h['eval_step'][k]}",
                (h["eval_step"][k], max(h["dev"])), color=INK2, fontsize=9,
                xytext=(-8, -4), textcoords="offset points", ha="right")
    ax.legend(frameon=False, fontsize=9)
    finish(ax, f"4. Train vs dev — {tag}", "1000 hidden units, 24-dim embeddings, lr 0.1",
           "step", "full-split loss")
fig.tight_layout(); fig.savefig(PLOTS / "04_train_dev.png"); plt.close(fig)

# ================================================================ 5. random search
print("[5] random search")
rng = random.Random(0)
trials = []
for t in range(30):
    lr = 10 ** rng.uniform(-2.5, 0.2)
    nh = rng.choice([50, 100, 200, 400])
    ne = rng.choice([4, 8, 12, 20])
    bs = rng.choice([16, 32, 64, 128])
    ps = make_model(n_emb=ne, n_hidden=nh, seed=234)
    train(ps, 4000, lr=lr, batch=bs)
    d = loss_on(ps, Xdev, Ydev)
    d = 5.0 if math.isnan(d) else min(d, 5.0)
    trials.append((lr, nh, ne, bs, d))
    print(f"   {t:2d} lr={lr:.4f} h={nh:<4} emb={ne:<3} bs={bs:<4} dev {d:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
sc = ax.scatter([t[0] for t in trials], [t[4] for t in trials],
                s=[t[1] / 4 for t in trials], c=S1, alpha=0.75, edgecolors=SURF, linewidths=1.5)
b = min(trials, key=lambda t: t[4])
ax.scatter([b[0]], [b[4]], s=b[1] / 4, c=S2, edgecolors=SURF, linewidths=1.5, zorder=3)
ax.annotate(f"best: lr={b[0]:.3f}, h={b[1]}, emb={b[2]}, bs={b[3]}\ndev {b[4]:.3f}",
            (b[0], b[4]), color=S2, fontsize=9, xytext=(-10, 12), textcoords="offset points", ha="right")
ax.set_xscale("log")
finish(ax, "5a. 30 random configs", "dot size = hidden units; 4000 steps each",
       "learning rate (log)", "dev loss")
ax = axes[1]
best_so_far, m = [], 9e9
for t in trials:
    m = min(m, t[4]); best_so_far.append(m)
ax.step(range(1, len(trials) + 1), best_so_far, color=S3, where="post")
finish(ax, "5b. Best dev loss so far", "most of the gain lands in the first ~10 trials",
       "trial", "best dev loss")
fig.tight_layout(); fig.savefig(PLOTS / "05_random_search.png"); plt.close(fig)

print(f"done in {time.time()-t0:.0f}s -> plots/")
