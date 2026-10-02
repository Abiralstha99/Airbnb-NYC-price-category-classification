# Airbnb NYC price category classification

This project compares a tuned decision tree and a small neural network on a binary classification task: given attributes of a New York City Airbnb listing, classify its listed price as **high** or **low**. The supplied dataset defines `high` as a price of at least **$180**, its 75th-percentile cutoff. The listed price itself is excluded from model inputs.

The project demonstrates exploratory analysis, leakage-aware preprocessing, stratified model selection, and a comparison on a held-out test set. It classifies a category in one historical dataset; it does not forecast prices, revenue, or investment returns.

## Repository layout

```text
.
├── data/
│   └── airbnbListingsData.csv
├── notebooks/
│   └── airbnb_price_classification.ipynb
├── requirements.txt
└── README.md
```

The CSV contains 28,022 listings and 51 columns. It was supplied with the original project. Its original publisher, collection date, and redistribution terms are not documented here; verify them before republishing the dataset in a new repository.

## Run the analysis

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook notebooks/airbnb_price_classification.ipynb
```

Select the virtual environment's Python kernel and run all cells from top to bottom. The notebook finds the CSV whether Jupyter starts in the project root or `notebooks/`. The tree search evaluates 36 parameter combinations across five folds, and the neural network trains for at most 20 epochs, so a full run may take several minutes. Notebook outputs are cleared in version control; running it produces the current metrics and figures.

## Modeling approach

1. Encode `price_category` as `high = 1` and `low = 0`; remove `price`, unused free text, amenities, and a duplicate host count.
2. Reserve a stratified 20% test set. Keep it out of preprocessing fits and model selection.
3. For the tree, impute numeric fields and one hot encode categorical fields inside a scikit-learn pipeline. Select depth and node size by five-fold stratified cross-validation using high-class F1.
4. For the neural network, split the training portion again for validation. Fit imputation, encoding, and scaling on its training subset only. Train a 64 → 32 hidden-unit network with early stopping.
5. Evaluate both final models on the same held-out test set using accuracy, high-class precision, recall, and F1.

The dataset has 7,237 high-price listings (25.8%). F1 is the selection metric because accuracy alone can obscure errors on this smaller class. The notebook reports the actual cross-validation score and held-out results after execution; previously saved scores were removed because the preprocessing and model-selection method changed.

## Limitations

- The label is derived from listed price, so the model learns a category defined by this dataset's $180 cutoff. It does not estimate a numeric price.
- Results are from one random stratified holdout. There is no temporal or external validation, and the dataset's collection period is unknown.
- Location and host activity can proxy for neighborhood wealth or host tenure. Group-level error rates have not been measured.
- Decision tree feature importance describes split usage, not causal effects. A deployment decision would require data provenance, a defined use case and error cost, and evaluation on newer data.

The cleaned analysis incorporates the original notebook's two-model comparison and discussion of potential bias.
