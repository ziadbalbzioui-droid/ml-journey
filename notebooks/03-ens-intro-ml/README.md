# 03-ens-intro-ml/

ENS intro-ML practical labs, as I work through them. Teacher outputs and answers were
stripped; exercises are `### YOUR CODE HERE ###` blocks with hints. The original teacher
notebooks (answer keys) are kept locally and are **not** committed — this repo is public.

## Order

Work the files in numerical order. 1 → 2 is a real dependency: notebook 2's three
exercises are all `GridSearchCV`, and notebook 1 is where that (plus `train_test_split`,
`classification_report`, `confusion_matrix`) is introduced.

| Notebook | Focus | Note |
|---|---|---|
| `1_supervised_classification.ipynb` | k-NN vs logistic regression, confusion matrix, cross-validated model selection | Half is exposition — go straight to the 7 assignments |
| `2_trees_forests_boosting.ipynb` | Trees → random forest → AdaBoost, grid search with CV | Needs notebook 1 first |
| `3_clustering.ipynb` | KMeans, DBSCAN, silhouette, ARI, GMM — evaluating without labels | Nearly independent |
| `4_mlp_pytorch.ipynb` | MLP from scratch | `4_mlp_pytorch_holes.ipynb` is the blank version |
| `5_cnn.ipynb` | CIFAR-10 conv nets, dropout, **transfer learning** | Transfer learning is the payload |
| `6_rnn_lstm.ipynb` | RNN/LSTM seq2seq | Overlaps makemore |
| `7_vision_transformer.ipynb` | Patches as tokens, attention maps, seq2seq Transformer | Hardest; a full day |

## Changes made to the originals

- Device selection prefers Apple MPS: the labs hardcoded
  `cuda:0 if torch.cuda.is_available() else "cpu"`, which silently fell back to CPU on
  an M4. Now `mps → cuda → cpu`.
- `5_cnn.ipynb`: fixed a `NameError` in the transfer-learning section — the cell that
  prints trainable parameters referenced an undefined `inception` instead of `model_ft`.

## Running

Run from this folder. Beyond the repo environment, the labs need
`torchvision`, `torchmetrics` and `seaborn`.

- `3_clustering.ipynb` reads `penguins_data.csv` (Palmer Penguins, committed here — 16 KB).
- `4_mlp_pytorch.ipynb` and `5_cnn.ipynb` download MNIST / CIFAR-10 (~170 MB) into
  `./data/`, which is gitignored.
- `6_rnn_lstm.ipynb` and `7_vision_transformer.ipynb` download the `fra-eng` corpus.
