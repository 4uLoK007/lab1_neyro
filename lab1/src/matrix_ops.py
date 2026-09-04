"""Задания 23-26, 33: производная, свёртка, произведения, нормы, маски."""

import numpy as np
from scipy.signal import convolve


def derivative_1d(x: np.ndarray) -> np.ndarray:
    """Производная (градиент) одномерного сигнала."""
    return np.gradient(x)


def gradient_2d(data: np.ndarray) -> np.ndarray:
    """Градиент многомерного массива по строкам (axis=0)."""
    return np.gradient(data, axis=0)


def convolution(x: np.ndarray, y: np.ndarray,
                mode: str = "full") -> np.ndarray:
    """Свёртка через NumPy."""
    return np.convolve(x, y, mode=mode)


def convolution_scipy(x: np.ndarray, y: np.ndarray,
                      mode: str = "full") -> np.ndarray:
    """Свёртка через SciPy."""
    return convolve(x, y, mode=mode)


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Скалярное произведение."""
    return float(np.dot(x, y))


def cross_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Векторное произведение (для 3-мерных векторов)."""
    return np.cross(x[:3], y[:3])


def norm_l1(x: np.ndarray) -> float:
    """Норма L1."""
    return float(np.linalg.norm(x, ord=1))


def norm_l2(x: np.ndarray) -> float:
    """Норма L2."""
    return float(np.linalg.norm(x, ord=2))


def binary_masks(x: np.ndarray) -> dict:
    """Возвращает набор бинарных масок."""
    return {
        "positive": x > 0,
        "negative": x < 0,
        "zero": x == 0,
        "interval_[-1,1]": (x >= -1) & (x <= 1),
    }