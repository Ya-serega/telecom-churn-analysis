"""
Главный скрипт для запуска полного анализа оттока клиентов.
"""

from src.data_loader import load_telecom_data
from src.data_cleaner import clean_telecom_data
from src.eda import perform_eda
from src.visualizer import create_visualizations
import json
from pathlib import Path


def main():
    """Основная функция запуска анализа."""

    print("\n" + "=" * 60)
    print("🚀 TELECOM CHURN ANALYSIS - ЗАПУСК ПРОЕКТА")
    print("=" * 60 + "\n")

    # Шаг 1: Загрузка данных
    print("📥 ШАГ 1: Загрузка данных...")
    df, summary = load_telecom_data('data/telco_churn.csv')

    # Шаг 2: Очистка данных
    print("\n🧹 ШАГ 2: Очистка данных...")
    df_clean = clean_telecom_data(df)

    # Шаг 3: EDA
    print("\n🔍 ШАГ 3: Разведочный анализ...")
    eda_report = perform_eda(df_clean)

    # Шаг 4: Визуализация
    print("\n📊 ШАГ 4: Создание визуализаций...")
    saved_files = create_visualizations(df_clean, output_dir='images')

    # Шаг 5: Сохранение отчёта
    print("\n💾 ШАГ 5: Сохранение отчёта...")

    report = {
        'data_summary': summary,
        'eda_report': {
            'target_analysis': eda_report['target_analysis'],
            'top_factors': eda_report['top_factors'],
            'business_insights': eda_report['business_insights']
        },
        'visualizations': saved_files
    }

    report_path = Path('reports/analysis_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"✓ Отчёт сохранён: {report_path}")

    # Финальный вывод
    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН УСПЕШНО")
    print("=" * 60)
    print("\n📁 Созданные файлы:")
    print(f"  • Данные: data/telco_churn.csv")
    print(f"  • Визуализации: {len(saved_files)} файлов в images/")
    print(f"  • Отчёт: {report_path}")
    print(f"  • SQL запросы: sql/sql_queries.sql")
    print("\n🎯 Ключевые инсайты:")
    for insight in eda_report['business_insights'][:3]:
        print(f"  {insight}")
    print("=" * 60 + "\n")

    return df_clean, eda_report


if __name__ == '__main__':
    df, report = main()