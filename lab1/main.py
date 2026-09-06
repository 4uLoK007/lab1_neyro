"""main.py — точка входа лабораторной работы №1.

Задания 1-13: чтение/запись, предобработка.
Задания 14-37: статистический анализ, матричные операции,
               скользящее окно.
"""

from src.config import set_seeds
set_seeds(42)

import numpy as np
import pandas as pd

from src.io_utils import read_table, write_table, show_table
from src.data_loader import check_data_quality, extract_columns, cast_types
from src.preprocess import handle_missing, to_numpy, split_train_val_test
from src.scaling import (MinMaxScalerCustom, StandardScalerCustom,
                         compare_with_sklearn)

from src import stats_analysis as sa
from src import matrix_ops as mo
from src import windows as win


# ============================================================
# ВАЖНО: столбцы по твоему варианту.
# Со скрина видно: x1, x5, x2, y. Если по варианту другие — замени.
# ============================================================
VARIANT_COLS = ["x1", "x2", "x5", "y"]


def main():
    # ============================================================
    # РАЗДЕЛ 1. Чтение и запись данных
    # ============================================================
    print("=" * 60)
    print("РАЗДЕЛ 1. Чтение и запись")
    print("=" * 60)

    # --- Задание 1: чтение ---
    df = read_table("data/dataset_var15.csv")

    # --- Задание 5: вывод таблицы ---
    print("\n[5] Первые 5 строк:")
    show_table(df, n=5)

    # --- Задание 4: проверка качества ---
    print("\n[4] Отчёт о качестве данных:")
    report = check_data_quality(df)
    for k, v in report.items():
        print(f"  {k}: {v}")

    # --- Задание 2: запись во все форматы ---
    print("\n[2] Сохранение во все форматы:")
    paths = write_table(df, "dataset_processed")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")

    # ============================================================
    # РАЗДЕЛ 2. Предобработка
    # ============================================================
    print("\n" + "=" * 60)
    print("РАЗДЕЛ 2. Предобработка")
    print("=" * 60)

    # --- Задание 6: извлечение столбцов по варианту ---
    print(f"\n[6] Извлечение столбцов: {VARIANT_COLS}")
    df_subset = extract_columns(df, VARIANT_COLS)
    print(f"    shape = {df_subset.shape}")

    # --- Задание 8: обработка "ошибка" и пропусков ---
    print("\n[8] Обработка некорректных значений и пропусков:")
    for c in VARIANT_COLS:
        df_subset = handle_missing(df_subset, c, "cast")   # "ошибка" → NaN
    print("    NaN после cast:")
    print(df_subset.isna().sum())

    for c in VARIANT_COLS:
        df_subset = handle_missing(df_subset, c, "mean")   # NaN → среднее
    print("    NaN после fill mean:")
    print(df_subset.isna().sum())

    # --- Задание 7: явные типы ---
    type_map = {c: "float64" for c in VARIANT_COLS}
    df_typed = cast_types(df_subset, type_map)
    print("\n[7] Типы после cast_types:")
    print(df_typed.dtypes)

    for c in VARIANT_COLS:
        df_subset = handle_missing(df_subset, c, "mean")   # NaN → среднее
    print("    NaN после fill mean:")
    print(df_subset.isna().sum())

    # --- Задание 7: явные типы ---
    type_map = {c: "float64" for c in VARIANT_COLS}
    df_typed = cast_types(df_subset, type_map)
    print("\n[7] Типы после cast_types:")
    print(df_typed.dtypes)

    # --- Задание 9: DataFrame → NumPy ---
    data = to_numpy(df_typed)
    print(f"\n[9] data.shape = {data.shape}")

    # --- Задание 12: разбиение на 3 части ---
    xtr, xval, xte = split_train_val_test(data, (70, 15, 15), percent=True)
    print(f"\n[12] train: {xtr.shape}, val: {xval.shape}, test: {xte.shape}")

    # ============================================================
    # РАЗДЕЛ 3. Задания 14-37
    # ============================================================
    print("\n" + "=" * 60)
    print("ЗАДАНИЯ 14-37")
    print("=" * 60)

    x = data[:, 0]    # первый столбец варианта — основной
    y = data[:, -1]   # последний столбец варианта (обычно целевой y)

    # --- 14: график исходных данных ---
    print("\n[14] График исходных данных...")
    sa.plot_series(data, "Исходные данные (вариант)")

    # --- 15: гистограмма ---
    print("[15] Гистограмма...")
    sa.plot_histogram(x, bins=30, name="task15_hist_x")

    # --- 16: сортированные столбцы ---
    print("[16] Отсортированные столбцы:")
    sa.show_sorted(df_typed)

    # --- 17: ECDF ---
    print("[17] Эмпирическая функция распределения...")
    sa.plot_ecdf(x, name="task17_ecdf_x")

    # --- 18: статистики ---
    print("\n[18] Статистики по первому столбцу:")
    for k, v in sa.column_stats(x).items():
        print(f"  {k}: {v:.4f}")
    print("\n[18] Пандас-статистики:")
    sa.pandas_stats(df_typed)

    # --- 19: доверительные интервалы ---
    print("\n[19] Доверительные интервалы (alpha=0.05):")
    print("  CI mean:", sa.ci_mean(x))
    print("  CI var :", sa.ci_var(x))

    # --- 20: ковариация и корреляция ---
    print("\n[20] Ковариация и корреляция:")
    cov, corr = sa.covariance_correlation(data)
    print("  Cov:\n", cov)
    print("  Corr:\n", corr)

    # --- 21: значимость корреляции ---
    print("\n[21] Значимость корреляции x vs y:")
    sa.correlation_significance(x, y)

    # --- 22: взаимная корреляция ---
    print("[22] Взаимная корреляция...")
    sa.plot_cross_correlation(x, y, max_lags=50)

    # --- 23: производная и градиент ---
    print("\n[23] Производная и градиент:")
    print("  производная x (первые 5):", mo.derivative_1d(x)[:5])
    print("  градиент data shape:", mo.gradient_2d(data).shape)

    # --- 24: свёртка ---
    print("\n[24] Свёртка:")
    conv_np = mo.convolution(x, y)
    conv_sp = mo.convolution_scipy(x, y)
    print("  len:", len(conv_np),
          "| совпадает с scipy:", np.allclose(conv_np, conv_sp))

    # --- 25: скалярное и векторное произведения ---
    print("\n[25] Скалярное и векторное произведения:")
    print("  dot(x, y)            =", mo.dot_product(x, y))
    print("  cross(x[:3], y[:3])  =", mo.cross_product(x, y))

    # --- 26: нормы L1, L2 ---
    print("\n[26] Нормы:")
    print("  ||x||_1 =", mo.norm_l1(x))
    print("  ||x||_2 =", mo.norm_l2(x))

    # --- 27: тесты распределения ---
    print("\n[27] Проверка гипотез о распределении:")
    sa.test_distribution(x, dist="norm")
    sa.test_distribution(x, dist="uniform")
    sa.shapiro_test(x)

    # --- 28: спектрограмма ---
    print("\n[28] Спектрограмма...")
    sa.plot_spectrogram(x, fs=1.0)

    # --- 29: периодограмма ---
    print("[29] Периодограмма...")
    sa.plot_periodogram(x, fs=1.0)

    # --- 30: АЧХ через FFT ---
    print("[30] АЧХ через FFT...")
    sa.plot_fft_amplitude(x)

    # --- 31: кубическая интерполяция ---
    print("[31] Кубическая интерполяция...")
    sa.plot_cubic_interpolation(x, factor=10)

    # --- 32: интерполяция сплайнами ---
    print("[32] Интерполяция сплайнами...")
    sa.plot_spline_interpolation(x, factor=10)

    # --- 33: бинарные маски ---
    print("\n[33] Бинарные маски:")
    for name, m in mo.binary_masks(x).items():
        print(f"  {name}: {m.sum()} элементов")

    # --- 34: сравнение со sklearn ---
    print("\n[34] Сравнение собственного и sklearn-масштабирования:")
    compare_with_sklearn(data[:, [0]])

    # --- 36-37: скользящее окно и среднее ---
    print("\n[36-37] Скользящее окно и среднее:")
    w = win.sliding_window(x, width=5)
    ma = win.moving_average(x, width=5)
    print("  sliding_window shape:", w.shape)
    print("  moving_average (первые 5):", ma[:5])

    print("\n" + "=" * 60)
    print("ЗАДАНИЯ 14-37 ВЫПОЛНЕНЫ. Графики сохранены в output/")
    print("=" * 60)

        # ============================================================
    # ЗАДАНИЯ 38-49
    # ============================================================
    from src import tensors as tn
    from src import matrix_ops as mo2

    # --- 38-45: тензоры ---
    tn.run_all(data)

    # --- 46: разреженная матрица ---
    print("\n[46] Разреженная матрица SciPy:")
    S, nnz = mo2.sparse_matrix_demo(10_000, 10_000, density=1e-5)
    print(f"  shape = {S.shape}, nnz = {nnz}, "
          f"плотность = {nnz / (S.shape[0] * S.shape[1]):.2e}")

    # --- 47: обратная матрица ---
    print("\n[47] Обратная матрица:")
    inv_info = mo2.inverse_matrix_demo(size=100)
    print(f"  det = {inv_info['det']:.4e}")
    print(f"  inv_ok = {inv_info['inv_ok']}, max_error = {inv_info['max_error']}")

    # --- 48: достройка массива до F ---
    print("\n[48] Достройка массива до F:")
    F = mo2.build_F(data)
    print(f"  F.shape = {F.shape}")

    # --- 49: пайплайн sklearn (PCA) ---
    print("\n[49] Пайплайн sklearn (PCA):")
    mo2.sklearn_pca_pipeline(F)
    mo2.plot_pca_cumulative(F)

    # ============================================================
    # ЗАДАНИЕ 50
    # ============================================================
    import os
    from src import image_proc as ip

    image_path = "data/image.png"
    if os.path.exists(image_path):
        ip.run_task50(image_path)
    else:
        print(f"\n[50] Файл {image_path} не найден — пропускаю Задание 50.")
        print("     Положи любое цветное изображение в data/image.png и перезапусти.")


if __name__ == "__main__":
    main()