"""Picklable model callables for the spec 12 evaluation tests (process pools need importable
objects).
"""

from __future__ import annotations

import os
import time

import numpy as np


class BatchDot:
    """``X @ a`` with a large random ``a``: rounding depends on batch composition."""

    def __init__(self, m: int, seed: int = 0) -> None:
        self.a = np.random.default_rng(seed).standard_normal(m) * 1e8

    def __call__(self, X):
        return X @ self.a

    def grad(self, X):
        return np.tile(self.a, (X.shape[0], 1)) + X.sum()


class BatchDotGrad:
    def __init__(self, inner: BatchDot) -> None:
        self.inner = inner

    def __call__(self, X):
        return self.inner.grad(X)


class BatchDotBoth:
    def __init__(self, inner: BatchDot) -> None:
        self.inner = inner

    def __call__(self, X):
        return self.inner(X), self.inner.grad(X)


def strict_value(X):
    """Asserts the one contract [C-11]: 2-D, writeable, C-contiguous float64."""
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.flags.writeable
    assert X.flags.c_contiguous
    assert X.dtype == np.float64
    return X.sum(axis=1)


def strict_grad(X):
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.flags.writeable
    assert X.flags.c_contiguous
    return 2.0 * X


def mutating_value(X):
    out = X.sum(axis=1)
    X[...] = np.nan
    return out


class RowModel:
    """Value = first column (the global row index); raises on chosen rows, optionally after a
    delay.
    """

    def __init__(self, fail_rows=(), slow_rows=(), delay: float = 0.0, nan_rows=()) -> None:
        self.fail_rows = set(fail_rows)
        self.slow_rows = set(slow_rows)
        self.nan_rows = set(nan_rows)
        self.delay = delay

    def __call__(self, X):
        rows = {int(r + 0.5) for r in X[:, 0]}
        if rows & self.slow_rows:
            time.sleep(self.delay)
        bad = sorted(rows & self.fail_rows)
        if bad:
            raise ValueError(f"model failed on row {bad[0]}")
        out = X[:, 0].copy()
        for i, r in enumerate(X[:, 0]):
            if int(r + 0.5) in self.nan_rows:
                out[i] = np.nan
        return out


def die(X):
    os._exit(1)


def point_value(x):
    return float(np.sum(x**2))


def point_grad(x):
    return 2.0 * x


def point_both(x):
    return float(np.sum(x**2)), 2.0 * x
