"""
Модуль для визуализации результатов анализа.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class TelecomVisualizer:
    """Класс для создания визуализаций телеком анализа."""

    def __init__(self, df: pd.DataFrame, output_dir: str = 'images'):
        """
        Инициализация визуализатора.

        Args:
            df: DataFrame с данными
            output_dir: Директория для сохранения графиков
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка стиля
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_churn_distribution(self, save: bool = True) -> plt.Figure:
        """
        График распределения целевой переменной.

        Args:
            save: Сохранять ли график в файл

        Returns:
            Figure объект
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        churn_counts = self.df['Churn'].value_counts()
        labels = ['Остался', 'Ушёл']
        colors = ['#2ecc71', '#e74c3c']

        axes[0].pie(
            churn_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=(0.05, 0.05),
            shadow=True
        )
        axes[0].set_title('Доля оттока клиентов', fontsize=14, fontweight='bold')

        # Count plot
        sns.countplot(
            data=self.df,
            x='Churn',
            ax=axes[1],
            palette=colors
        )
        axes[1].set_title('Количество клиентов', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('')
        axes[1].set_xticklabels(['Остался', 'Ушёл'])

        # Добавление значений на столбцы
        for p in axes[1].patches:
            axes[1].annotate(
                f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11
            )

        plt.tight_layout()

        if save:
            file_path = self.output_dir / '01_churn_distribution.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено: {file_path}")

        return fig

    def plot_churn_by_contract(self, save: bool = True) -> plt.Figure:
        """
        График оттока по типу контракта.

        Args:
            save: Сохранять ли график в файл

        Returns:
            Figure объект
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Группировка данных
        contract_churn = self.df.groupby('Contract')['Churn'].agg(['count', 'sum'])
        contract_churn['churn_rate'] = contract_churn['sum'] / contract_churn['count']

        # Bar plot
        colors = ['#3498db', '#f39c12', '#e74c3c']
        bars = ax.bar(
            contract_churn.index,
            contract_churn['churn_rate'],
            color=colors,
            edgecolor='black',
            linewidth=1.2
        )

        # Добавление значений
        for bar, rate in zip(bars, contract_churn['churn_rate']):
            ax.annotate(
                f'{rate:.1%}',
                (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )

        ax.set_title('Уровень оттока по типу контракта', fontsize=14, fontweight='bold')
        ax.set_xlabel('Тип контракта')
        ax.set_ylabel('Доля оттока')
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save:
            file_path = self.output_dir / '02_churn_by_contract.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено: {file_path}")

        return fig

    def plot_churn_by_tenure(self, save: bool = True) -> plt.Figure:
        """
        График оттока по длительности обслуживания.

        Args:
            save: Сохранять ли график в файл

        Returns:
            Figure объект
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Box plot
        sns.boxplot(
            data=self.df,
            x='Churn',
            y='tenure',
            ax=ax,
            palette=['#2ecc71', '#e74c3c']
        )

        ax.set_title('Распределение продолжительности обслуживания по оттоку', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(['Остался', 'Ушёл'])
        ax.set_ylabel('Продолжительность обслуживания (месяцы)')

        plt.tight_layout()

        if save:
            file_path = self.output_dir / '03_churn_by_tenure.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено: {file_path}")

        return fig

    def plot_correlation_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Тепловая карта корреляций.

        Args:
            save: Сохранять ли график в файл

        Returns:
            Figure объект
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Выбор числовых колонок
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )

        ax.set_title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            file_path = self.output_dir / '04_correlation_heatmap.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено: {file_path}")

        return fig

    def plot_top_churn_factors(self, top_n: int = 10, save: bool = True,
                               exclude_cols: List[str] = None) -> plt.Figure:
        """
        График топ факторов оттока по ВСЕМ категориальным признакам.

        Args:
            top_n: Количество факторов для отображения
            save: Сохранять ли график
            exclude_cols: Колонки для исключения из анализа

        Returns:
            Figure объект
        """
        if exclude_cols is None:
            exclude_cols = ['customerID', 'Churn']

        fig, ax = plt.subplots(figsize=(14, 8))

        # Собираем данные по всем категориальным признакам
        categorical_cols = self.df.select_dtypes(include=['category', 'object']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

        factors_data = []

        for col in categorical_cols:
            group_stats = self.df.groupby(col, observed=False)['Churn'].agg(['count', 'sum', 'mean'])
            group_stats.columns = ['total', 'churned', 'churn_rate']

            for value, row in group_stats.iterrows():
                if row['total'] < 50:  # Фильтр малых групп
                    continue
                factors_data.append({
                    'label': f"{col}: {value}",
                    'churn_rate': row['churn_rate'],
                    'total': int(row['total'])
                })

        # Создаем DataFrame и сортируем
        factors_df = pd.DataFrame(factors_data).sort_values('churn_rate', ascending=False).head(top_n)

        # Горизонтальный bar plot
        y_labels = factors_df['label'].str.wrap(width=30)  # Перенос длинных названий
        colors = plt.cm.Reds(np.linspace(0.9, 0.4, len(factors_df)))

        bars = ax.barh(y_labels, factors_df['churn_rate'], color=colors, edgecolor='black', linewidth=0.5)

        # Добавляем значения на столбцы
        for bar, rate, total in zip(bars, factors_df['churn_rate'], factors_df['total']):
            ax.annotate(
                f'{rate:.1%} (n={total})',
                (bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2),
                va='center', fontsize=9, fontweight='normal'
            )

        # Оформление
        ax.set_xlabel('Доля оттока', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_title(f'ТОП-{top_n} категорий с наибольшим оттоком', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Поворот подписей для читаемости
        plt.xticks(rotation=0)
        plt.yticks(fontsize=9)

        # Линия среднего оттока для сравнения
        avg_churn = self.df['Churn'].mean()
        ax.axvline(x=avg_churn, color='blue', linestyle=':', linewidth=1.5,
                   label=f'Средний отток: {avg_churn:.1%}')
        ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()

        if save:
            file_path = self.output_dir / '05_top_churn_factors.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

        return fig

    def create_dashboard(self, save: bool = True) -> plt.Figure:
        """
        Создание сводного дашборда.

        Args:
            save: Сохранять ли график в файл

        Returns:
            Figure объект
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Pie chart оттока
        churn_counts = self.df['Churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0, 0].pie(
            churn_counts.values,
            labels=['Остался', 'Ушёл'],
            autopct='%1.1f%%',
            colors=colors,
            explode=(0.05, 0.05)
        )
        axes[0, 0].set_title('Доля оттока', fontsize=12, fontweight='bold')

        # 2. Отток по контракту
        contract_churn = self.df.groupby('Contract')['Churn'].mean()
        axes[0, 1].bar(contract_churn.index, contract_churn.values, color=['#3498db', '#f39c12', '#e74c3c'])
        axes[0, 1].set_title('Отток по продолжительности обслуживания', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=15)

        # 3. Box plot tenure
        sns.boxplot(data=self.df, x='Churn', y='tenure', ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Продолжительность обслуживания', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticklabels(['Остался', 'Ушёл'])

        # 4. MonthlyCharges distribution
        sns.kdeplot(data=self.df, x='MonthlyCharges', hue='Churn', ax=axes[1, 1], fill=True, alpha=0.5)
        axes[1, 1].set_title('Распределение месячного платежа', fontsize=12, fontweight='bold')
        axes[1, 1].legend(['Остался', 'Ушёл'])

        plt.suptitle('TELECOM CHURN ANALYSIS DASHBOARD', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            file_path = self.output_dir / '06_dashboard.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Сохранено: {file_path}")

        return fig

    def save_all_plots(self) -> List[str]:
        """
        Сохранение всех графиков.

        Returns:
            Список путей к сохранённым файлам
        """
        saved_files = []

        print("\n" + "=" * 60)
        print("СОХРАНЕНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 60 + "\n")

        self.plot_churn_distribution()
        saved_files.append(str(self.output_dir / '01_churn_distribution.png'))

        self.plot_churn_by_contract()
        saved_files.append(str(self.output_dir / '02_churn_by_contract.png'))

        self.plot_churn_by_tenure()
        saved_files.append(str(self.output_dir / '03_churn_by_tenure.png'))

        self.plot_correlation_heatmap()
        saved_files.append(str(self.output_dir / '04_correlation_heatmap.png'))

        self.plot_top_churn_factors()
        saved_files.append(str(self.output_dir / '05_top_churn_factors.png'))

        self.create_dashboard()
        saved_files.append(str(self.output_dir / '06_dashboard.png'))

        print(f"\n Всего сохранено графиков: {len(saved_files)}")
        print("=" * 60 + "\n")

        return saved_files


def create_visualizations(df: pd.DataFrame, output_dir: str = 'images') -> List[str]:
    """
    Основная функция для создания визуализаций.

    Args:
        df: DataFrame с данными
        output_dir: Директория для сохранения

    Returns:
        Список путей к файлам
    """
    visualizer = TelecomVisualizer(df, output_dir)
    return visualizer.save_all_plots()


if __name__ == '__main__':
    # Пример использования
    from data_loader import load_telecom_data
    from data_cleaner import clean_telecom_data

    df, _ = load_telecom_data()
    df_clean = clean_telecom_data(df)
    saved_files = create_visualizations(df_clean)

    print("\nСохранённые файлы:")
    for file in saved_files:
        print(f"  • {file}")