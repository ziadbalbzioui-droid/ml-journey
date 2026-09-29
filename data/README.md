# data/

Everything in this folder is **gitignored** — datasets are re-downloadable, so they
do not belong in git history.

Restore them with:

```bash
python3 scripts/fetch_data.py
```

Expected layout:

```
data/
├── names.txt                  32k first names        → makemore (02-zero-to-hero)
├── tinyshakespeare.txt        1.1 MB of Shakespeare  → nano-gpt
├── housing/housing.csv        California housing     → 01-handson-ml/housing.ipynb
├── titanic/{train,test}.csv   Titanic                → 01-handson-ml/classification.ipynb
└── house-prices/              Kaggle, manual download → 01-handson-ml/house-prices-kaggle.ipynb
    ├── train.csv  test.csv  sample_submission.csv  data_description.txt
```
