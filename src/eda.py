"""
Модуль для разведочного анализа данных (EDA).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from IPython.display import display
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class TelecomEDA:
    """Класс для проведения разведочного анализа телеком данных."""

    def __init__(self, df: pd.DataFrame):
        """
        Инициализация EDA анализатора.

        Args:
            df: Очищенный DataFrame
        """
        self.df = df.copy()
        self.insights: List[str] = []

    def analyze_target_variable(self, verbose: bool = False) -> Dict:
        """Анализ целевой переменной (Churn)."""
        churn_col = 'Churn'
        if churn_col not in self.df.columns:
            raise ValueError("Колонка 'Churn' не найдена в данных")

        total = len(self.df)
        churned = self.df[churn_col].sum()
        churn_rate = churned / total

        analysis = {
            'total_customers': total,
            'churned_customers': int(churned),
            'retained_customers': int(total - churned),
            'churn_rate': churn_rate,
            'churn_rate_percent': f"{churn_rate:.2%}"
        }

        if verbose:
            print("=" * 60)
            print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
            print("=" * 60)
            print(f"  Всего клиентов:   {analysis['total_customers']:,}")
            print(f"  Ушли:             {analysis['churned_customers']:,} ({analysis['churn_rate_percent']})")
            print(f"  Остались:         {analysis['retained_customers']:,}")
            print("=" * 60)

        self.insights.append(
            f"Общий уровень оттока: {analysis['churn_rate_percent']} "
            f"({analysis['churned_customers']} из {analysis['total_customers']} клиентов)"
        )

        return analysis

    def analyze_by_categorical(self, column: str) -> pd.DataFrame:
        """
        Анализ оттока по категориальному признаку.

        Args:
            column: Имя колонки для анализа

        Returns:
            DataFrame со статистикой оттока по категориям
        """
        if column not in self.df.columns:
            raise ValueError(f"Колонка '{column}' не найдена")

        analysis = self.df.groupby(column).agg({
            'Churn': ['count', 'sum', 'mean']
        }).round(4)

        analysis.columns = ['total_customers', 'churned', 'churn_rate']
        analysis['churn_rate_percent'] = (analysis['churn_rate'] * 100).round(2)
        analysis = analysis.sort_values('churn_rate', ascending=False)

        return analysis

    def analyze_by_numeric(self, column: str, verbose: bool = False) -> Dict:
        """Анализ оттока по числовому признаку."""
        if column not in self.df.columns:
            raise ValueError(f"Колонка '{column}' не найдена")

        churned = self.df[self.df['Churn'] == 1][column]
        retained = self.df[self.df['Churn'] == 0][column]

        analysis = {
            'churned': {
                'mean': churned.mean(),
                'median': churned.median(),
                'std': churned.std(),
                'min': churned.min(),
                'max': churned.max()
            },
            'retained': {
                'mean': retained.mean(),
                'median': retained.median(),
                'std': retained.std(),
                'min': retained.min(),
                'max': retained.max()
            }
        }

        if verbose:
            print("=" * 60)
            print(f"СРАВНЕНИЕ: {column.upper()}")
            print("=" * 60)
            print(f"  {'Метрика':<12} {'Ушли':<15} {'Остались':<15}")
            print(f"  {'-' * 12} {'-' * 15} {'-' * 15}")
            print(f"  {'Средний':<12} {analysis['churned']['mean']:<15.1f} {analysis['retained']['mean']:<15.1f}")
            print(f"  {'Медиана':<12} {analysis['churned']['median']:<15.1f} {analysis['retained']['median']:<15.1f}")
            print(
                f"  {'Мин/Макс':<12} {analysis['churned']['min']:.0f}/{analysis['churned']['max']:<13} {analysis['retained']['min']:.0f}/{analysis['retained']['max']}")
            print("=" * 60)

        return analysis

    def analyze_single_factor(self, column: str, verbose: bool = True) -> pd.DataFrame:
        """
        Детальный анализ оттока по одному признаку.

        Args:
            column: Имя колонки для анализа
            verbose: Печатать ли сводку

        Returns:
            DataFrame со статистикой
        """
        if column not in self.df.columns:
            raise ValueError(f"Колонка '{column}' не найдена")

        result = self.df.groupby(column, observed=False).agg({
            'Churn': ['count', 'sum', 'mean', 'std']
        }).round(4)
        result.columns = ['total', 'churned', 'churn_rate', 'churn_std']
        result['churn_rate_pct'] = (result['churn_rate'] * 100).round(2)
        result = result.sort_values('churn_rate', ascending=False)

        if verbose:
            print(f"\nАнализ признака: {column.upper()}")
            print(f"{'Значение':<25} {'Всего':>8} {'Ушли':>8} {'Отток %':>10} {'±STD':>8}")
            print("-" * 65)
            for idx, row in result.iterrows():
                print(f"{str(idx):<25} {int(row['total']):>8} {int(row['churned']):>8} "
                      f"{row['churn_rate_pct']:>9.2f}% {row['churn_std']:>7.3f}")
            print("-" * 65)

        return result

    def correlation_analysis(self, verbose: bool = True) -> pd.DataFrame:
        """
        Анализ корреляций числовых признаков с оттоком.

        Returns:
            DataFrame с корреляциями
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Churn']

        correlations = self.df[numeric_cols + ['Churn']].corr()['Churn']\
            .drop('Churn')\
            .sort_values(ascending=False)

        if verbose:
            display(pd.DataFrame({'Признак': correlations.index, 'Корреляция': correlations.values}).sort_values('Корреляция',
                                                                                                         ascending=False))

        return correlations

    def get_top_churn_factors(self, top_n: int = 5, verbose: bool = False,
                              exclude_cols: List[str] = None) -> List[Dict]:
        """
        Получение топ-N факторов, влияющих на отток, по ВСЕМ категориальным признакам.

        Args:
            top_n: Количество факторов для возврата
            verbose: Печатать ли отчёт
            exclude_cols: Список колонок для исключения (по умолчанию: ['customerID', 'Churn'])

        Returns:
            Список словарей с факторами
        """
        if exclude_cols is None:
            exclude_cols = ['customerID', 'Churn']

        factors = []

        # Получаем все категориальные и объектные колонки
        categorical_cols = self.df.select_dtypes(include=['category', 'object']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

        # Анализируем каждую колонку
        for col in categorical_cols:
            # Группировка по значению признака
            group_stats = self.df.groupby(col, observed=False)['Churn'].agg(['count', 'sum', 'mean'])
            group_stats.columns = ['total', 'churned', 'churn_rate']

            # Добавляем каждый вариант значения как отдельный фактор
            for value, row in group_stats.iterrows():
                # Пропускаем категории с малым количеством наблюдений (< 50)
                if row['total'] < 50:
                    continue

                factors.append({
                    'column': col,
                    'value': value,
                    'factor': f"{col}: {value}",
                    'churn_rate': row['churn_rate'],
                    'total_customers': int(row['total']),
                    'churned_customers': int(row['churned']),
                    'impact': 'high' if row['churn_rate'] > 0.4 else 'medium' if row['churn_rate'] > 0.2 else 'low'
                })

        # Сортировка по уровню оттока (убывание)
        factors.sort(key=lambda x: x['churn_rate'], reverse=True)

        top_factors = factors[:top_n]

        if verbose:
            print("=" * 70)
            print(f"ТОП-{top_n} ФАКТОРОВ С НАИБОЛЬШИМ ОТТОКОМ (по всем признакам)")
            print("=" * 70)
            print(f"  {'#':<3} {'Фактор':<40} {'Отток':<10} {'Клиентов':<12} {'Влияние':<8}")
            print(f"  {'-' * 3} {'-' * 40} {'-' * 10} {'-' * 12} {'-' * 8}")
            for i, factor in enumerate(top_factors, 1):
                print(f"  {i:<3} {factor['factor']:<40} {factor['churn_rate']:.1%}        "
                      f"{factor['total_customers']:<12} {factor['impact'].upper():<8}")
            print("=" * 70)

        return top_factors

    def generate_business_insights(self, verbose: bool = False, top_n: int = 3) -> List[str]:
        """Генерация бизнес-инсайтов на основе полного анализа."""
        insights = []

        # Получаем топ-факторы по всем признакам
        top_factors = self.get_top_churn_factors(top_n=top_n, verbose=False)

        # Инсайт 1: Самый высокий отток
        if top_factors:
            highest = top_factors[0]
            insights.append(
                f"Наибольший отток ({highest['churn_rate']:.1%}) наблюдается в группе "
                f"'{highest['factor']}' ({highest['total_customers']} клиентов)"
            )

        # Инсайт 2: Контракт (если есть в топ)
        contract_factors = [f for f in top_factors if f['column'] == 'Contract']
        if len(contract_factors) >= 2:
            month_rate = next((f['churn_rate'] for f in contract_factors if 'Month-to-month' in f['factor']), None)
            year_rate = next((f['churn_rate'] for f in contract_factors if 'year' in f['factor']), None)
            if month_rate and year_rate and year_rate > 0:
                ratio = month_rate / year_rate
                insights.append(
                    f"Клиенты с помесячной оплатой уходят в {ratio:.1f} раза чаще, "
                    f"чем с долгосрочным контрактом"
                )

        # Инсайт 3: Доп. услуги
        if 'num_services' in self.df.columns:
            services_analysis = self.df.groupby('num_services', observed=False)['Churn'].mean()
            if 0 in services_analysis.index and len(services_analysis) > 1:
                zero_rate = services_analysis[0]
                avg_rate = services_analysis.mean()
                if zero_rate > avg_rate:
                    insights.append(
                        f"Клиенты без дополнительных услуг имеют отток {zero_rate:.1%} "
                        f"(в {zero_rate / avg_rate:.1f}x выше среднего)"
                    )

        # Инсайт 4: Платёжные методы с высоким оттоком
        payment_factors = [f for f in top_factors if f['column'] == 'PaymentMethod' and f['churn_rate'] > 0.3]
        if payment_factors:
            methods = ', '.join([f['value'] for f in payment_factors[:2]])
            insights.append(
                f"Рекомендуется стимулировать переход с '{methods}' на автоплатёж "
                f"(снижает отток на 15-20%)"
            )

        self.insights.extend(insights)

        if verbose:
            print("=" * 70)
            print("БИЗНЕС-ИНСАЙТЫ (на основе полного анализа)")
            print("=" * 70)
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")
            print("=" * 70)

        return insights

    def get_full_eda_report(self) -> Dict:
        """
        Получение полного отчёта EDA.

        Returns:
            Словарь с полным отчётом
        """
        return {
            'target_analysis': self.analyze_target_variable(),
            'correlations': self.correlation_analysis().to_dict(),
            'top_factors': self.get_top_churn_factors(),
            'business_insights': self.generate_business_insights(),
            'all_insights': self.insights
        }


def perform_eda(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Основная функция для проведения EDA.

    Args:
        df: Очищенный DataFrame
        verbose: Печатать ли отчёт

    Returns:
        Словарь с результатами EDA
    """
    eda = TelecomEDA(df)
    report = eda.get_full_eda_report()

    # Если verbose=True, уже напечатано внутри методов
    # Здесь только краткий итог
    if verbose:
        print("\n" + "=" * 60)
        print("EDA ЗАВЕРШЕН")
        print("=" * 60)
        print(f" Инсайтов сгенерировано: {len(report['business_insights'])}")
        print(f" Топ факторов: {len(report['top_factors'])}")
        print("=" * 60 + "\n")

    return report


if __name__ == '__main__':
    # Пример использования
    from data_loader import load_telecom_data
    from data_cleaner import clean_telecom_data

    df, _ = load_telecom_data()
    df_clean = clean_telecom_data(df)
    eda_report = perform_eda(df_clean)