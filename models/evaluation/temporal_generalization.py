"""
Temporal generalization of a stimulus code (King and Dehaene 2014).

A decoder is trained on the population state at step t and tested at step t'.
A stable code gives a square block of high accuracy that lasts as long as the
code does. A transient code gives high accuracy only near the diagonal inside a
short window. Vishne, Gerber, Knight and Deouell (2023, Cell Reports 42, 112752)
used this contrast to separate sustained ventral-stream content from transient
prefrontal content.

Cross-validation is over TRIALS. The same trials never appear in the training
and the test fold at any pair of steps, so a trial-specific quirk cannot carry a
decoder across time.

Time is in environment steps. No Hz.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["classify_dynamics", "label_shuffle_threshold", "temporal_generalization"]

SUSTAINED_FRACTION = 0.8   # share of the window that must decode and generalize
TRANSIENT_EARLY = 3        # steps after onset that a transient code must cover
TRANSIENT_LATE_MAX = 0.5   # max share of the later window a transient code may cover


def _check(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 3 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X must be [trials, T, D] matching y, got {X.shape} and {y.shape}")
    if np.bincount(np.unique(y, return_inverse=True)[1]).min() < 2:
        raise ValueError("every class needs at least 2 trials")
    return X, y


def temporal_generalization(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                            seed: int = 0) -> np.ndarray:
    """[T, T] accuracy, row = training step, column = test step."""
    X, y = _check(X, y)
    _, T, _ = X.shape
    n_splits = min(n_splits, int(np.bincount(np.unique(y, return_inverse=True)[1]).min()))
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X[:, 0], y))
    correct = np.zeros((T, T))
    for train, test in folds:
        for t in range(T):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[train, t], y[train])
            for u in range(T):
                correct[t, u] += np.sum(clf.predict(X[test, u]) == y[test])
    return correct / len(y)


def label_shuffle_threshold(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                            n_perm: int = 20, seed: int = 0) -> float:
    """p95 of the diagonal accuracy under labels shuffled across trials.

    The maximum over steps is taken per permutation, so the threshold controls
    the family of cells, not one cell.
    """
    X, y = _check(X, y)
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        acc = np.diag(temporal_generalization(X, shuffled, n_splits, seed))
        maxima.append(float(acc.max()))
    return float(np.percentile(maxima, 95))


def classify_dynamics(acc: np.ndarray, threshold: float, onset: int, length: int) -> str:
    """SUSTAINED, TRANSIENT or MIXED for the window [onset, onset + length).

    SUSTAINED: the diagonal is above threshold on at least SUSTAINED_FRACTION of
    the window, and a decoder trained at the second window step is above
    threshold on at least SUSTAINED_FRACTION of the window.
    TRANSIENT: the first TRANSIENT_EARLY window steps decode, and at most
    TRANSIENT_LATE_MAX of the remaining steps do.
    """
    window = np.arange(onset, onset + length)
    diag = np.diag(acc)[window] > threshold
    anchor = min(onset + 1, acc.shape[0] - 1)
    carried = acc[anchor, window] > threshold
    if diag.mean() >= SUSTAINED_FRACTION and carried.mean() >= SUSTAINED_FRACTION:
        return "SUSTAINED"
    early, late = diag[:TRANSIENT_EARLY], diag[TRANSIENT_EARLY:]
    if early.all() and (late.size == 0 or late.mean() <= TRANSIENT_LATE_MAX):
        return "TRANSIENT"
    return "MIXED"
