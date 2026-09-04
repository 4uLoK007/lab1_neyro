"""Загрузка и первичная обработка табличных данных."""

import pandas as pd
import numpy as np


def check_data_quality(df: pd.DataFrame) -> dict:
    """Проверяет пустые ячейки, некорректные данные, типы."""
    report = {
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {},
    }
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad_mask = converted.isna() & df[col].notna()
            if bad_mask.any():
                report["non_numeric"][col] = df.loc[bad_mask, col].head(5).tolist()
    return report


def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Возвращает новый DataFrame с нужными столбцами."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()


def cast_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """type_map: {'col': 'float64', ...}"""
    df = df.copy()
    for col, dtype in type_map.items():
        if col not in df.columns:
            raise KeyError(f"Столбец не найден: {col}")
        df[col] = df[col].astype(dtype)
    return df