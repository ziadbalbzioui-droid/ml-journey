"""Gradient boosting for regression (squared loss), from scratch.

Weak learner: sklearn's DecisionTreeRegressor for now; swap in your own tree later.
Written 2026-10-26 (TODO version filled in). Run:  python gbm_scratch.py
The experiment grid (lr x n_trees) at the bottom is not yet predicted/analysed.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingScratch:
    def __init__(self, n_trees=100, learning_rate=0.1, max_depth=2):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        # F0 = argmin_c sum_i 1/2 (y_i - F0)^2
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)  # current predictions on the training set
        self.trees = []
        for _ in range(self.n_trees):
            # pseudo-residuals r_i = -dL/dF at the current predictions
            r = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)
            tree.fit(X, r)
            # add the new correction (a tree is a function: call .predict)
            F = F + self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        F = self.F0
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(0, 2 * np.pi, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.normal(size=n)

    # Sanity check against sklearn: should print a difference ~1e-15
    ours = GradientBoostingScratch(n_trees=100, learning_rate=0.1, max_depth=2).fit(X, y)
    ref = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(X, y)
    print("max |ours - sklearn|:", np.abs(ours.predict(X) - ref.predict(X)).max())

    # Experiments: write your predictions BEFORE running this part
    X_test = rng.uniform(0, 2 * np.pi, size=(1000, 1))
    y_test = np.sin(X_test[:, 0]) + 0.3 * rng.normal(size=1000)
    for lr in [1.0, 0.1]:
        for m in [1, 10, 200]:
            model = GradientBoostingScratch(n_trees=m, learning_rate=lr).fit(X, y)
            mse_train = np.mean((model.predict(X) - y) ** 2)
            mse_test = np.mean((model.predict(X_test) - y_test) ** 2)
            print(f"lr={lr:<4} trees={m:<4} train MSE={mse_train:.3f}  test MSE={mse_test:.3f}")
