import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import set_seeds
from io_utils import read_table, write_table, save_figure


def main():
    # 1. Фиксируем seed
    set_seeds()
    print("Seed установлен.")

    # 2. Путь к датасету
    data_path = os.path.join("data", "dataset_var15.csv")
    print(f"Читаем файл: {data_path}")

    # 3. Читаем таблицу
    df = read_table(data_path)
    print("Таблица успешно прочитана.")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print("\nПервые 5 строк:")
    print(df.head())

    # 4. Сохраняем во все форматы
    paths = write_table(df, "dataset_processed")
    print("\nФайлы сохранены:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()