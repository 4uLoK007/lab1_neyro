"""Задания 14-22, 27-32: статистический анализ."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import spectrogram, periodogram
from scipy.interpolate import interp1d, UnivariateSpline

from src.io_utils import save_figure


def plot_series(data: np.ndarray, title: str = "Исходные данные") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f"col_{i}")
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "task14_series")
    return fig


def plot_histogram(x: np.ndarray, bins: int = 30,
                   name: str = "task15_hist") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_title("Нормализованная гистограмма")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def show_sorted(df: pd.DataFrame) -> None:
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(np.sort(df[col].to_numpy()))


def plot_ecdf(x: np.ndarray, name: str = "task17_ecdf") -> plt.Figure:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.set_title("Эмпирическая функция распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def column_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "std": float(np.std(x, ddof=0)),
        "mode": float(stats.mode(x, keepdims=False).mode),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
    }


def pandas_stats(df: pd.DataFrame) -> None:
    print("=== describe() ===")
    print(df.describe())
    print("\n=== mode() ===")
    print(df.mode())
    print("\n=== median() ===")
    print(df.median())
    print("\n=== var(ddof=0) ===")
    print(df.var(ddof=0))


def ci_mean(x: np.ndarray, alpha: float = 0.05) -> tuple:
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    se = s / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)


def ci_var(x: np.ndarray, alpha: float = 0.05) -> tuple:
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low


def covariance_correlation(data: np.ndarray) -> tuple:
    cov = np.cov(data, rowvar=False)
    corr = np.corrcoef(data, rowvar=False)
    return cov, corr


def pandas_cov_corr(data: np.ndarray) -> tuple:
    df = pd.DataFrame(data)
    return df.cov(), df.corr()


def correlation_significance(x: np.ndarray, y: np.ndarray,
                             alpha: float = 0.05) -> dict:
    r, p_value = stats.pearsonr(x, y)
    significant = p_value < alpha
    print(f"r = {r:.4f}, p = {p_value:.4f}")
    if significant:
        print(f"Корреляция значима (p < {alpha})")
    else:
        print(f"Корреляция незначима (p >= {alpha})")
    return {"r": r, "p_value": p_value, "significant": significant}


def cross_correlation(x: np.ndarray, y: np.ndarray,
                      max_lags: int = 50) -> tuple:
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]


def plot_cross_correlation(x: np.ndarray, y: np.ndarray,
                           max_lags: int = 50,
                           name: str = "task22_crosscorr") -> plt.Figure:
    lags, c = cross_correlation(x, y, max_lags)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(lags, c, basefmt=" ")
    ax.set_title("Взаимная корреляция")
    ax.set_xlabel("Лаг")
    ax.set_ylabel("Корреляция")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def test_distribution(x: np.ndarray, dist: str = "norm",
                      alpha: float = 0.05) -> dict:
    if dist == "uniform":
        lo, hi = x.min(), x.max()
        stat, p = stats.kstest(x, "uniform", args=(lo, hi - lo))
    elif dist == "norm":
        stat, p = stats.kstest(x, "norm", args=(x.mean(), x.std()))
    else:
        raise ValueError(f"Неизвестное распределение: {dist}")
    print(f"[KS] dist={dist}, stat={stat:.4f}, p={p:.4f}")
    if p > alpha:
        print(f"  Нет оснований отвергнуть H0 (p > {alpha})")
    else:
        print(f"  H0 отвергается (p <= {alpha})")
    return {"stat": stat, "p_value": p, "dist": dist}


def shapiro_test(x: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p = stats.shapiro(x)
    print(f"[Shapiro] stat={stat:.4f}, p={p:.4f}")
    return {"stat": stat, "p_value": p}


def plot_spectrogram(x: np.ndarray, fs: float = 1.0,
                     name: str = "task28_spectrogram") -> plt.Figure:
    f, t, Sxx = spectrogram(x, fs=fs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_ylabel("Частота")
    ax.set_xlabel("Время")
    ax.set_title("Спектрограмма")
    save_figure(fig, name)
    return fig


def plot_periodogram(x: np.ndarray, fs: float = 1.0,
                     name: str = "task29_periodogram") -> plt.Figure:
    f, Pxx = periodogram(x, fs=fs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    ax.set_title("Периодограмма")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def plot_fft_amplitude(x: np.ndarray,
                       name: str = "task30_fft") -> plt.Figure:
    n = len(x)
    fft = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq[:half], np.abs(fft[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    ax.set_title("АЧХ через FFT")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def cubic_interpolation(x: np.ndarray, factor: int = 10) -> tuple:
    x_idx = np.arange(len(x))
    f_cubic = interp1d(x_idx, x, kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    y_new = f_cubic(x_new)
    return x_new, y_new


def plot_cubic_interpolation(x: np.ndarray, factor: int = 10,
                             name: str = "task31_cubic") -> plt.Figure:
    x_new, y_new = cubic_interpolation(x, factor)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(x)), x, "o", label="Исходные")
    ax.plot(x_new, y_new, "-", label="Кубическая интерполяция")
    ax.legend()
    ax.set_title("Кубическая интерполяция")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig


def spline_interpolation(x: np.ndarray, factor: int = 10,
                         k: int = 3, s: float = 0) -> tuple:
    x_idx = np.arange(len(x))
    spl = UnivariateSpline(x_idx, x, k=k, s=s)
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    y_new = spl(x_new)
    return x_new, y_new


def plot_spline_interpolation(x: np.ndarray, factor: int = 10,
                              name: str = "task32_spline") -> plt.Figure:
    x_new, y_new = spline_interpolation(x, factor)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(x)), x, "o", label="Исходные")
    ax.plot(x_new, y_new, "-", label="Сплайн")
    ax.legend()
    ax.set_title("Интерполяция сплайнами")
    ax.grid(True, alpha=0.3)
    save_figure(fig, name)
    return fig