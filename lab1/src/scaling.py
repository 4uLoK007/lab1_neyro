"""Масштабирование: MinMax, Standard + сравнение со sklearn."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MinMaxScalerCustom:
    """Min-Max масштабирование в диапазон [a, b]."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b
        self.min_, self.max_ = None, None

    def fit(self, x: np.ndarray):
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0,
                         1, self.max_ - self.min_)
        return self.a + (x - self.min_) * (self.b - self.a) / denom

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0,
                         1, self.max_ - self.min_)
        return (x_new - self.a) * denom / (self.b - self.a) + self.min_


class StandardScalerCustom:
    """Стандартизация: (x - mean) / std (ddof=0)."""

    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return (x - self.mean_) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return x_new * std + self.mean_


def compare_with_sklearn(x: np.ndarray) -> None:
    custom_mm = MinMaxScalerCustom().fit(x)
    sk_mm = MinMaxScaler().fit(x)
    assert np.allclose(custom_mm.transform(x), sk_mm.transform(x), atol=1e-10)

    custom_std = StandardScalerCustom().fit(x)
    sk_std = StandardScaler().fit(x)
    assert np.allclose(custom_std.transform(x), sk_std.transform(x), atol=1e-10)

    x_back = custom_mm.inverse_transform(custom_mm.transform(x))
    assert np.allclose(x, x_back, atol=1e-10)
    print("Сравнение с sklearn: OK")