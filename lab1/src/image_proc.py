"""Задание 50: обработка изображения средствами SciPy."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from sklearn.decomposition import PCA

from src.io_utils import save_figure


# =====================================================================
# 1. Разделение цветовых каналов
# =====================================================================
def split_channels(img: np.ndarray) -> dict:
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение (H, W, 3)")

    r = img[..., 0].copy()
    g = img[..., 1].copy()
    b = img[..., 2].copy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);       axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds");   axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens"); axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues");  axes[3].set_title("Синий канал")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_channels")
    plt.close(fig)

    return {"r": r, "g": g, "b": b}


# =====================================================================
# 2. Поворот изображения на 40 градусов
# =====================================================================
def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    rotated = rotate(
        img.astype(np.float32),
        angle=angle,
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
    )
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);     axes[0].set_title("Исходное")
    axes[1].imshow(rotated); axes[1].set_title(f"Поворот {angle}°")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_rotated")
    plt.close(fig)

    return rotated


# =====================================================================
# 3. Ч/б изображение
# =====================================================================
def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = (0.299 * img[..., 0] +
                0.587 * img[..., 1] +
                0.114 * img[..., 2]).astype(np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);             axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/б")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_gray")
    plt.close(fig)

    return gray_u8


# =====================================================================
# 4. Гистограмма ч/б
# =====================================================================
def gray_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    save_figure(fig, "task50_gray_hist")
    plt.close(fig)

    return {"hist": hist, "bin_edges": bin_edges}


# =====================================================================
# 5. Разбиение на 3 части по гистограмме
# =====================================================================
def split_by_histogram(gray: np.ndarray, hist_info: dict):
    hist = hist_info["hist"]
    edges = hist_info["bin_edges"]
    cum = np.cumsum(hist)
    total = cum[-1]

    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1:.0f})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1:.0f}–{t2:.0f})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2:.0f})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_split3")
    plt.close(fig)

    return part1, part2, part3


# =====================================================================
# 6. Чёрная круговая рамка
# =====================================================================
def add_circular_frame(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2

    Y, X = np.ogrid[:h, :w]
    mask_outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[mask_outside] = 0
    else:
        framed[mask_outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);    axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("В круговой рамке")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_framed")
    plt.close(fig)

    return framed


# =====================================================================
# 7. Случайный шум
# =====================================================================
def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_noisy")
    plt.close(fig)

    return noisy


# =====================================================================
# 8. Гауссовское размытие
# =====================================================================
def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);     axes[0].set_title("Исходное")
    axes[1].imshow(blurred); axes[1].set_title(f"Размытие (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_blurred")
    plt.close(fig)

    return blurred


# =====================================================================
# 9. Sharpening (unsharp masking)
# =====================================================================
def sharpen_image(img: np.ndarray, sigma: float = 2.0,
                  amount: float = 1.5) -> np.ndarray:
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);       axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title(f"Sharpening (amount={amount})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_sharpened")
    plt.close(fig)

    return sharpened


# =====================================================================
# 10. PCA к изображению
# =====================================================================
def apply_pca_to_image(img, n_components=32, svd_solver='full'):
    """
    Применяет PCA к изображению.

    Параметры
    ----------
    img : np.ndarray
        Входное изображение (H, W) или (H, W, C).
    n_components : int
        Желаемое число главных компонент. Автоматически
        ограничивается значением min(n_samples, n_features).
    svd_solver : str
        'full' или 'randomized' — обрезают n_components сами.
        'covariance_eigh' (sklearn >= 1.5) падает с ValueError,
        если n_components > min(n_samples, n_features).

    Возвращает
    ----------
    reconstructed : np.ndarray
        Восстановленное изображение той же формы, что и img.
    """
    # --- 1. Приводим изображение к 2D-матрице (n_samples, n_features) ---
    original_shape = img.shape
    flat = img.reshape(-1, original_shape[-1]) if img.ndim == 3 else img.reshape(-1, 1)

    n_samples, n_features = flat.shape

    # --- 2. ЯВНО вычисляем допустимый максимум компонент ---
    max_components = min(n_samples, n_features)

    # --- 3. ЯВНО ограничиваем n_components ---
    n_components = int(n_components)
    if n_components > max_components:
        print(f"[PCA] n_components={n_components} > max={max_components}, "
              f"понижаем до {max_components}")
        n_components = max_components

    if n_components < 1:
        raise ValueError(
            f"PCA невозможен: данные имеют форму {flat.shape}, "
            f"max_components={max_components}"
        )

    # --- 4. ЯВНО передаём n_components и solver в PCA ---
    pca_k = PCA(n_components=n_components, svd_solver=svd_solver)
    transformed = pca_k.fit_transform(flat)

    # --- 5. Обратное преобразование ---
    reconstructed = pca_k.inverse_transform(transformed)
    reconstructed = reconstructed.reshape(original_shape)

    return reconstructed

# =====================================================================
# Общий запуск Задания 50
# =====================================================================
def run_task50(image_path: str) -> dict:
    """Последовательно выполняет все 10 подзаданий."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 50: обработка изображения")
    print("=" * 60)

    img = plt.imread(image_path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    results = {}
    results["channels"] = split_channels(img)
    print("[50.1] Каналы — ок")

    results["rotated"] = rotate_image(img, angle=40.0)
    print("[50.2] Поворот на 40° — ок")

    results["gray"] = to_grayscale(img)
    print("[50.3] Ч/б — ок")

    results["hist"] = gray_histogram(results["gray"])
    print("[50.4] Гистограмма — ок")

    results["parts"] = split_by_histogram(results["gray"], results["hist"])
    print("[50.5] Разбиение на 3 части — ок")

    results["framed"] = add_circular_frame(img)
    print("[50.6] Круговая рамка — ок")

    results["noisy"] = add_noise(img, sigma=25.0)
    print("[50.7] Шум — ок")

    results["blurred"] = gaussian_blur(img, sigma=2.0)
    print("[50.8] Размытие — ок")

    results["sharpened"] = sharpen_image(img, sigma=2.0, amount=1.5)
    print("[50.9] Sharpening — ок")

    results["pca"] = apply_pca_to_image(img, n_components=32)
    print("[50.10] PCA — ок")

    print("=" * 60)
    print("ЗАДАНИЕ 50 ВЫПОЛНЕНО. Графики в output/")
    print("=" * 60)
    return results