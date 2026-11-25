import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        logp = np.log(y_pred)
        logpComp = np.log(1-y_pred)

        L = np.sum(np.dot(logp,y_true)) + np.sum(np.dot(
            logpComp, 1-y_true))
        L = -L/n

        return np.round(L,4)
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        lgPred = np.log(y_pred)
        L = y_true*lgPred
        
        L = np.sum(L)
        L = np.round(-L/n, 4)
        return L
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pass
