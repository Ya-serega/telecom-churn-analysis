"""
Модуль для очистки и предобработки данных.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

# Отключаем предупреждения для чистого вывода
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DataCleaner:
    """Класс для очистки и предобработки данных телеком датасета."""

    def __init__(self, df: pd.DataFrame):
        """
        Инициализация очистителя данных.

        Args:
            df: Исходный DataFrame
        """
        self.df = df.copy()
        self.cleaning_log: List[str] = []

    def log_action(self, action: str) -> None:
        """Логирование выполненного действия по очистке."""
        self.cleaning_log.append(action)

    def handle_missing_values(self) -> 'DataCleaner':
        """
        Обработка пропущенных значений.

        Returns:
            Self для цепочки вызовов
        """
        # TotalCharges может иметь пустые строки вместо NaN
        if 'TotalCharges' in self.df.columns:
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
            self.df['TotalCharges'] = pd.to_numeric(
                self.df['TotalCharges'],
                errors='coerce'
            )
            missing_count = self.df['TotalCharges'].isnull().sum()
            if missing_count > 0:
                # Заполняем медианой по tenure
                self.df['TotalCharges'] = self.df.groupby('tenure', observed=False)['TotalCharges'] \
                    .transform(lambda x: x.fillna(x.median()))

                # Если всё ещё есть NaN - заполняем общей медианой
                median_value = self.df['TotalCharges'].median()
                self.df['TotalCharges'] = self.df['TotalCharges'].fillna(median_value)

                self.log_action(f"Обработано {missing_count} пропусков в TotalCharges")
            else:
                self.log_action("Пропуски в TotalCharges не найдены")

        return self

    def convert_data_types(self) -> 'DataCleaner':
        """
        Преобразование типов данных к корректным.

        Returns:
            Self для цепочки вызовов
        """
        # Числовые признаки
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for col in numeric_cols:
            if col in self.df.columns:
                if col == 'SeniorCitizen':
                    # SeniorCitizen оставляем как int (0/1)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int8')
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.log_action(f"Преобразован тип {col} к numeric")

        # Категориальные признаки (оптимизация памяти)
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

        # Целевая переменная
        if 'Churn' in self.df.columns:
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            self.log_action("Преобразована целевая переменная Churn: Yes=1, No=0")

        return self

    def remove_duplicates(self) -> 'DataCleaner':
        """
        Удаление дубликатов записей.

        Returns:
            Self для цепочки вызовов
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_count = initial_count - len(self.df)

        if removed_count > 0:
            self.log_action(f"Удалено {removed_count} дубликатов записей")
        else:
            self.log_action("Дубликаты записей не найдены")

        return self

    def create_derived_features(self) -> 'DataCleaner':
        """Создание производных признаков для анализа."""

        # Сегмент по длительности обслуживания
        self.df['tenure_group'] = pd.cut(
            self.df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12 мес', '13-24 мес', '25-48 мес', '49+ мес']
        )
        self.log_action("Создан признак tenure_group (сегмент по длительности)")

        # Средний чек за месяц службы (защита от деления на 0)
        self.df['avg_monthly_charge'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)
        self.df['avg_monthly_charge'] = self.df['avg_monthly_charge'].fillna(
            self.df['avg_monthly_charge'].median()
        )
        self.log_action("Создан признак avg_monthly_charge (средний чек за месяц)")

        # Количество дополнительных услуг
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        available_services = [col for col in service_cols if col in self.df.columns]

        if available_services:
            self.df['num_services'] = sum(
                self.df[col] == 'Yes' for col in available_services
            )
            self.df['num_services'] = self.df['num_services'].fillna(0)
            self.log_action("Создан признак num_services (количество доп. услуг)")

        # ФИНАЛЬНАЯ ПРОВЕРКА: заполняем все оставшиеся пропуски
        final_missing = self.df.isnull().sum().sum()
        if final_missing > 0:
            # Заполняем числовые колонки медианой
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())

            # Заполняем категориальные колонки модой
            category_cols = self.df.select_dtypes(include=['category', 'object']).columns
            for col in category_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            self.log_action(f"Заполнено {final_missing} остаточных пропусков")
        else:
            self.log_action("Остаточные пропуски не найдены")

        return self

    def get_cleaning_report(self) -> Dict:
        """
        Получение отчёта о выполненной очистке.

        Returns:
            Словарь с отчётом
        """
        return {
            'initial_shape': self.df.shape,
            'actions_performed': len(self.cleaning_log),
            'log': self.cleaning_log,
            'final_missing': self.df.isnull().sum().sum()
        }

    def get_cleaned_data(self) -> pd.DataFrame:
        """Возвращает очищенный DataFrame."""
        return self.df


def clean_prepare_telecom_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Основная функция для очистки телеком данных.

    Args:
        df: Исходный DataFrame
        verbose: Печатать ли отчёт

    Returns:
        Очищенный DataFrame
    """
    cleaner = DataCleaner(df)

    cleaner.handle_missing_values()
    cleaner.convert_data_types()
    cleaner.remove_duplicates()
    cleaner.create_derived_features()

    # ФИНАЛЬНАЯ ВАЛИДАЦИЯ
    final_missing = cleaner.df.isnull().sum().sum()

    if verbose:
        print("=" * 60)
        print("ОТЧЁТ О ОЧИСТКЕ")
        print("=" * 60)
        report = cleaner.get_cleaning_report()
        for action in report['log']:
            print(f"{action}")
        print(f"\nИтоговый размер: {report['initial_shape'][0]} записей × {report['initial_shape'][1]} признаков")
        print(f"Пропусков осталось: {final_missing}")
        print(f"Статус: {'ГОТОВО' if final_missing == 0 else 'ТРЕБУЕТ ВНИМАНИЯ'}")
        print("=" * 60)

    return cleaner.get_cleaned_data()


if __name__ == '__main__':
    # Пример использования
    from data_loader import load_telecom_data

    df, _ = load_telecom_data()
    df_clean = clean_prepare_telecom_data(df)
    print(df_clean.head())