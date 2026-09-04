"""Предобработка данных: пропуски, NumPy, разбиение."""

import pandas as pd
import numpy as np

from src.data_loader import check_data_quality


# =====================================================================
# Задание 8. Меню обработки проблем
# =====================================================================
def handle_missing(df: pd.DataFrame, col: str, action: str,
                   value=None) -> pd.DataFrame:
    """Обрабатывает пропуски в столбце col.

    action: 'drop', 'cast', 'mean', 'ffill', 'bfill', 'zero', 'value'
    """
    if col not in df.columns:
        raise KeyError(f"Столбец не найден: {col}")

    df = df.copy()

    if action == "drop":
        df = df.dropna(subset=[col])
    elif action == "cast":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif action == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif action == "ffill":
        df[col] = df[col].ffill()
    elif action == "bfill":
        df[col] = df[col].bfill()
    elif action == "zero":
        df[col] = df[col].fillna(0)
    elif action == "value":
        if value is None:
            raise ValueError("Для action='value' нужно указать value")
        df[col] = df[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")

    return df


def interactive_menu(df: pd.DataFrame) -> pd.DataFrame:
    """Интерактивное меню обработки пропусков."""
    while True:
        print("\n=== Меню обработки ===")
        print("1. Показать проблемы")
        print("2. Обработать столбец")
        print("3. Выход")
        choice = input("Выбор: ").strip()

        if choice == "1":
            print(check_data_quality(df))
        elif choice == "2":
            col = input("Столбец: ").strip()
            action = input(
                "Действие (drop/cast/mean/ffill/bfill/zero/value): "
            ).strip()
            val = input("Значение: ").strip() if action == "value" else None
            df = handle_missing(df, col, action, val)
            print("Готово.")
        elif choice == "3":
            break
        else:
            print("Неверный выбор.")

    return df


# =====================================================================
# Задание 9. DataFrame → NumPy
# =====================================================================
def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64)


# =====================================================================
# Задание 12. Разбиение на 3 части
# =====================================================================
def split_train_val_test(x: np.ndarray, ratios, percent: bool = False):
    """Разбивает массив на train/val/test.

    ratios: кортеж (r1, r2, r3). percent=True — в процентах.
    """
    r = np.array(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_test = n - n_train - n_val

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test