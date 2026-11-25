from cluster_analyzer import ClusterAnalyzer


def run_pipeline():
    print("Старт пайплайна кластеризации...")

    analyzer = ClusterAnalyzer(k_range=range(2, 16), random_state=42, scale_features=True)

    # Шаг 1: загрузка таблицы и числовых признаков
    df, features_matrix = analyzer.load_data("dataset.csv")

    # Шаг 2: подбор числа кластеров (для справки и графиков метрик)
    print("\n[Шаг 2] Подбор оптимального k...")
    analyzer.find_optimal_k(features_matrix)

    # Шаг 3: запуск финального разбиения на 5 кластеров
    print("\n[Шаг 3] Запуск финального разбиения на кластеры...")
    cluster_ids = analyzer.fit_predict(features_matrix, n_clusters=5)

    # Шаг 4: сохранение разметки и модели
    df["cluster"] = cluster_ids
    df.to_csv("dataset_clustered.csv", index=False, encoding="utf-8")
    print("[INFO] Размеченные данные сохранены в dataset_clustered.csv")

    analyzer.save_model("kmeans_model.joblib")

    # Шаг 5: построение графиков
    print("\n[Шаг 5] Отрисовка графиков...")
    analyzer.plot_results(features_matrix, save_prefix="cluster_report")

    print("Пайплайн успешно завершён.")


if __name__ == "__main__":
    run_pipeline()