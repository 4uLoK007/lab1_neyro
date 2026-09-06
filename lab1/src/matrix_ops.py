"""Задания 23-26, 33, 46-49: производная, свёртка, произведения, нормы,
маски, разреженная матрица, обратная матрица, достройка F, PCA-пайплайн."""

import numpy as np
from scipy.signal import convolve


# =====================================================================
# Задание 23. Производная и градиент
# =====================================================================
def derivative_1d(x: np.ndarray) -> np.ndarray:
    return np.gradient(x)


def gradient_2d(data: np.ndarray) -> np.ndarray:
    return np.gradient(data, axis=0)


# =====================================================================
# Задание 24. Свёртка
# =====================================================================
def convolution(x: np.ndarray, y: np.ndarray,
                mode: str = "full") -> np.ndarray:
    return np.convolve(x, y, mode=mode)


def convolution_scipy(x: np.ndarray, y: np.ndarray,
                      mode: str = "full") -> np.ndarray:
    return convolve(x, y, mode=mode)


# =====================================================================
# Задание 25. Скалярное и векторное произведения
# =====================================================================
def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))


def cross_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.cross(x[:3], y[:3])


# =====================================================================
# Задание 26. Нормы L1 и L2
# =====================================================================
def norm_l1(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=1))


def norm_l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=2))


# =====================================================================
# Задание 33. Бинарные маски
# =====================================================================
def binary_masks(x: np.ndarray) -> dict:
    return {
        "positive": x > 0,
        "negative": x < 0,
        "zero": x == 0,
        "interval_[-1,1]": (x >= -1) & (x <= 1),
    }


# =====================================================================
# Задание 46. Разреженная матрица SciPy
# =====================================================================
def sparse_matrix_demo(rows: int = 100_000, cols: int = 100_000,
                       density: float = 1e-5):
    from scipy import sparse
    S = sparse.random(rows, cols, density=density, format="csr")
    return S, S.nnz


# =====================================================================
# Задание 47. Обратная матрица
# =====================================================================
def inverse_matrix_demo(size: int = 100) -> dict:
    M = np.random.rand(size, size)
    det = np.linalg.det(M)

    result = {"det": float(det), "size": size}

    if abs(det) > 1e-10:
        M_inv = np.linalg.inv(M)
        err = float(np.abs(M @ M_inv - np.eye(size)).max())
        result["inv_ok"] = True
        result["max_error"] = err
    else:
        result["inv_ok"] = False
        result["max_error"] = None

    return result


# =====================================================================
# Задание 48. Достроить массив до F (n, m, 3)
# =====================================================================
def build_F(X: np.ndarray) -> np.ndarray:
    """Собирает F = stack([X, X_mm, X_std], axis=-1) формы (n, m, 3)."""
    from src.scaling import MinMaxScalerCustom, StandardScalerCustom

    X_mm = MinMaxScalerCustom().fit_transform(X)
    X_std = StandardScalerCustom().fit_transform(X)
    F = np.stack([X, X_mm, X_std], axis=-1)
    return F


# =====================================================================
# Задание 49. Пайплайн sklearn (PCA)
# =====================================================================
def sklearn_pca_pipeline(F: np.ndarray) -> dict:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA

    n = len(F)
    X_flat = F.reshape(n, -1)

    scalers = {"minmax": MinMaxScaler(), "std": StandardScaler()}
    results = {}

    for name, scaler in scalers.items():
        pipe = Pipeline([("scaler", scaler), ("pca", PCA())])
        pipe.fit(X_flat)
        ll = pipe.named_steps["pca"].score_samples(
            pipe.named_steps["scaler"].transform(X_flat)
        ).sum()
        results[name] = float(ll)
        print(f"[49] {name}: log-likelihood = {ll:.4f}")

    best = max(results, key=results.get)
    print(f"[49] Лучший вариант: {best}")
    results["best"] = best
    return results


def plot_pca_cumulative(F: np.ndarray,
                        name: str = "task49_pca_cumulative"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from src.io_utils import save_figure

    n = len(F)
    X_flat = F.reshape(n, -1)
    pca = PCA().fit(X_flat)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA: кумулятивная объяснённая дисперсия")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig