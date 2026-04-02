"""
Модуль для загрузки и первичного осмотра данных.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Класс для загрузки данных из CSV файла."""

    def __init__(self, file_path: str):
        """
        Инициализация загрузчика данных.

        Args:
            file_path: Путь к CSV файлу
        """
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.

        Args:
            encoding: Кодировка файла (по умолчанию 'utf-8')

        Returns:
            DataFrame с загруженными данными
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.file_path}")

        self.df = pd.read_csv(self.file_path, encoding=encoding)
        return self.df

    def get_summary(self) -> dict:
        """
        Получение сводной информации о датасете.

        Returns:
            Словарь с основной информацией о данных
        """
        if self.df is None:
            raise ValueError("Сначала загрузите данные методом load_data()")

        summary = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()
        }
        return summary

    def print_info(self) -> None:
        """Вывод основной информации о датасете в консоль."""
        if self.df is None:
            raise ValueError("Сначала загрузите данные методом load_data()")

        print("=" * 60)
        print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
        print("=" * 60)
        print(f"Количество записей: {self.df.shape[0]:,}")
        print(f"Количество признаков: {self.df.shape[1]}")
        print("\nТипы данных:")
        print(self.df.dtypes)
        print("\nПропущенные значения:")
        df_temp = self.df.replace(r'^\s*$', np.nan, regex=True) # Замена пустых строк на Nan для корректного подсчета
        missing = df_temp.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Пропущено': missing,
            '%': missing_pct
        })
        print(missing_df[missing > 0] if any(missing > 0) else "Нет пропущенных значений")
        print("\nЦелевая переменная (Churn):")
        if 'Churn' in self.df.columns:
            print(self.df['Churn'].value_counts())
            print(f"\nДоля оттока: {(self.df['Churn'] == 'Yes').mean():.2%}")
        print("=" * 60)


def load_telecom_data(file_path: str = 'data/telco_churn.csv') -> Tuple[pd.DataFrame, dict]:
    """
    Основная функция для загрузки данных телеком датасета.

    Args:
        file_path: Путь к файлу с данными

    Returns:
        Кортеж (DataFrame, summary_dict)
    """
    loader = DataLoader(file_path)
    df = loader.load_data()
    summary = loader.get_summary()
    loader.print_info()

    return df, summary


if __name__ == '__main__':
    # Пример использования
    df, summary = load_telecom_data()