# Progress log

Newest first. One entry per working session — what I built, what finally clicked, what
is still fuzzy. The "still fuzzy" line is the most valuable one: it is what I bring to
the next Claude chat.

---

## Where I am right now (2026-09-29)

- **Track:** Karpathy, *Neural Networks: Zero to Hero* — video 6 (nanoGPT), part-way.
- **Last thing built:** `02-zero-to-hero/nano-gpt/train.ipynb` — token + position
  embeddings feeding a linear head, trained with AdamW on tinyshakespeare.
- **Stopped exactly before:** self-attention. No `Head`, `MultiHeadAttention` or `Block`
  class exists yet. The model currently has no way to look at previous tokens.

## Next up

- [ ] Self-attention: the `Head` class — why query · key, why divide by √head_size,
      why the `tril` mask.
- [ ] Multi-head attention, then a full `Block` (attention + feed-forward + residual).
- [ ] LayerNorm — and being able to say out loud how it differs from the BatchNorm1D I
      hand-rolled in `makemore_rnn.ipynb`.
- [ ] Scale up: bigger `n_embd`, `block_size`, dropout, and generate readable text.

---

## Log

### 2026-09-21 — nanoGPT: data pipeline and the bigram baseline
Built `nano-gpt/train.ipynb`: char encode/decode, 90/10 train/val split, `get_batch`
with random offsets, a `BigramLanguageModel` with token **and** position embeddings, a
`generate` loop that samples one token at a time, training with AdamW.
**Still fuzzy:** _(fill in)_

### 2026-08-30 — the same WaveNet, in real PyTorch
`makemore_torch.py`: rewrote the hand-rolled net with `nn.Embedding`, `nn.Conv1d`
(kernel 2, stride 2) standing in for `FlattenConsecutive` + `Linear`, `nn.BatchNorm1d`,
`AdamW` + scheduler, `DataLoader`, MPS device, grad clipping, checkpoint save/load.
**What it proved:** every PyTorch module maps onto something I had already written by hand.

### 2026-08-22 — hyperparameter experiments
`hparam_tuning.py`: LR range test, LR curve comparison, capacity sweep, train/dev gap,
random search — five figures in `plots/`. Also `makemore_rnn.ipynb`: hierarchical
WaveNet-style network from my own `Linear`, `BatchNorm1D`, `Tanh`, `Embedding`,
`FlattenConsecutive` classes.

### 2026-08-19 — MLP character model
`makemore_mpl.ipynb`: Bengio-style MLP over a 3-character context — embedding table,
hidden tanh layer, softmax, minibatch training, train/dev/test split.

### 2026-08-17 — bigram character model
`makemore.ipynb`: counted bigrams into a 27×27 matrix, sampled from it, then trained the
same model with gradient descent and saw the two losses meet.

### 2026-08-16 — micrograd
`micrograd.ipynb`: `Value` class with `+`, `*`, `**`, `tanh`, `exp`, topological sort,
`.backward()`; then `Neuron` / `Layer` / `MLP` on top. Backprop written by hand, no torch.

### 2026-08-14 — first PyTorch tensors
`scratch/tensor_test.ipynb`: torch / torchvision setup.

### 2026-06-02 — Kaggle House Prices
`01-handson-ml/house-prices-kaggle.ipynb`: dropped columns >50% NaN, imputation,
train/test split, first submission pipeline.

### 2026-03-13 — California housing
`01-handson-ml/housing.ipynb`: EDA, histograms, feature inspection.

### 2026-03-10 — classification
`01-handson-ml/classification.ipynb`: MNIST with KNN + `GridSearchCV` over weights and
`n_neighbors`; Titanic fetch + load.

_(Dates before 2026-09-29 come from file timestamps — the code predates this repo.)_

---

## Entry template

```markdown
### YYYY-MM-DD — <title>
**Built:** what runs now that did not run before.
**Clicked:** the idea I could now explain to someone else.
**Still fuzzy:** the question I want to open the next Claude chat with.
```
