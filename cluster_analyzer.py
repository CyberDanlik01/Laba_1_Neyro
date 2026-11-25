import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ClusterAnalyzer:
    """Класс для подбора числа кластеров, обучения KMeans и построения отчётных графиков."""

    def __init__(self, k_range=range(2, 16), random_state=42, scale_features=True):
        self.candidate_ks = list(k_range)
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.best_k = None
        self.model = None
        self.labels = None
        self.metrics_df = None

    def _resolve_input_path(self, filepath):
        """Находит файл в текущей папке или рядом со скриптом (в т.ч. без расширения или в Excel)."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dirs = [os.getcwd(), script_dir]

        base, ext = os.path.splitext(filepath)
        rel_names = [filepath, base, f"{base}.csv", f"{base}.xlsx", f"{base}.xls"]
        seen = []

        for folder in base_dirs:
            for name in rel_names:
                if not name:
                    continue
                candidate = name if os.path.isabs(name) else os.path.join(folder, name)
                if candidate not in seen:
                    seen.append(candidate)
                if os.path.exists(candidate):
                    return candidate

        raise FileNotFoundError(
            f"Файл данных не найден: {os.path.abspath(filepath)} "
            f"(проверялись варианты: {', '.join(seen)})"
        )

    def _read_table(self, actual_path):
        """Читает CSV (в т.ч. с разделителем ';') или Excel-файл."""
        ext = os.path.splitext(actual_path)[1].lower()

        if ext in (".xlsx", ".xls"):
            return pd.read_excel(actual_path)

        try:
            df = pd.read_csv(actual_path, sep=None, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

        try:
            return pd.read_csv(actual_path)
        except Exception:
            return pd.read_excel(actual_path)

    def load_data(self, filepath="dataset.csv"):
        """Загружает таблицу данных и извлекает (при необходимости масштабирует) матрицу признаков."""
        actual_path = self._resolve_input_path(filepath)
        df = self._read_table(actual_path)

        feature_cols = [c for c in df.columns if "feature_" in str(c)]
        if not feature_cols:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not feature_cols:
            raise ValueError(
                f"В файле '{actual_path}' не найдено числовых признаков. "
                f"Столбцы: {df.columns.tolist()}"
            )

        X = df[feature_cols].to_numpy(dtype=float)
        if self.scale_features and self.scaler is not None:
            X = self.scaler.fit_transform(X)

        print(
            f"[INFO] Файл '{os.path.basename(actual_path)}' загружен. "
            f"Строк: {X.shape[0]}, признаков: {X.shape[1]}, "
            f"масштабирование: {'вкл' if self.scale_features else 'выкл'}"
        )
        return df, X

    def find_optimal_k(self, X):
        """Оценивает кандидатов k по метрикам Silhouette, WCSS и Calinski-Harabasz."""
        records = []
        sample_limit = 10000 if len(X) > 10000 else len(X)

        for k in self.candidate_ks:
            estimator = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=5,
                random_state=self.random_state,
            )
            assigned = estimator.fit_predict(X)

            sil_score = silhouette_score(
                X, assigned, sample_size=sample_limit, random_state=self.random_state
            )
            ch_score = calinski_harabasz_score(X, assigned)

            records.append(
                {
                    "clusters": k,
                    "wcss": estimator.inertia_,
                    "silhouette": sil_score,
                    "calinski": ch_score,
                }
            )

        self.metrics_df = pd.DataFrame(records)

        best_row_idx = self.metrics_df["silhouette"].idxmax()
        best_row = self.metrics_df.loc[best_row_idx]
        self.best_k = int(best_row["clusters"])

        print("\n=== Сводка метрик по числу кластеров ===")
        print(
            self.metrics_df.to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )
        print(
            f"\n[RESULT] Оптимальное k (по Silhouette): {self.best_k} "
            f"(Silhouette = {best_row['silhouette']:.4f})"
        )
        return self.best_k

    def fit_predict(self, X, n_clusters=None):
        """Обучает финальную модель KMeans и возвращает метки для каждого объекта."""
        chosen_k = int(n_clusters) if n_clusters is not None else self.best_k
        if chosen_k is None:
            chosen_k = self.find_optimal_k(X)

        self.best_k = chosen_k

        self.model = KMeans(
            n_clusters=chosen_k,
            init="k-means++",
            n_init=30,
            random_state=self.random_state,
        )
        self.labels = self.model.fit_predict(X)
        print(f"[INFO] Финальная модель обучена на k={self.best_k} кластерах.")
        return self.labels

    def save_model(self, filename="kmeans_model.joblib"):
        """Сохраняет обученную модель на диск с помощью joblib."""
        if self.model is not None:
            joblib.dump(self.model, filename)
            print(f"[INFO] Модель записана в {filename}")

    def _plot_metrics_panel(self, save_prefix):
        """Строит графики метода локтя и коэффициента силуэта."""
        fig, (ax_wcss, ax_sil) = plt.subplots(1, 2, figsize=(12, 5))

        ax_wcss.plot(
            self.metrics_df["clusters"],
            self.metrics_df["wcss"],
            marker="s",
            linestyle="-",
            color="#6a3d9a",
            linewidth=2,
        )
        ax_wcss.axvline(
            self.best_k,
            color="#e6550d",
            linestyle="--",
            linewidth=1.8,
            label=f"Selected k={self.best_k}",
        )
        ax_wcss.set_title("Elbow Curve (WCSS)")
        ax_wcss.set_xlabel("Clusters (k)")
        ax_wcss.set_ylabel("Within-Cluster Sum of Squares")
        ax_wcss.set_xticks(self.candidate_ks)
        ax_wcss.legend()

        ax_sil.plot(
            self.metrics_df["clusters"],
            self.metrics_df["silhouette"],
            marker="D",
            linestyle="-",
            color="#008080",
            linewidth=2,
        )
        ax_sil.axvline(
            self.best_k,
            color="#e6550d",
            linestyle="--",
            linewidth=1.8,
            label=f"Selected k={self.best_k}",
        )
        ax_sil.set_title("Silhouette Score by k")
        ax_sil.set_xlabel("Clusters (k)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_xticks(self.candidate_ks)
        ax_sil.legend()

        fig.suptitle("Selection of Cluster Count", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_metrics.png", dpi=250)

    def _plot_pca_panel(self, X, save_prefix):
        """Строит 2D- и (при наличии >=3 признаков) 3D-проекции PCA с центроидами."""
        n_dims = X.shape[1]

        pca_2d = PCA(n_components=2, random_state=self.random_state)
        coords_2d = pca_2d.fit_transform(X)
        centers_2d = pca_2d.transform(self.model.cluster_centers_)
        exp_var_2d = pca_2d.explained_variance_ratio_ * 100

        fig = plt.figure(figsize=(15, 6))
        palette = plt.get_cmap("Dark2" if self.best_k <= 8 else "tab20")

        ax_2d = fig.add_subplot(1, 2, 1)
        for cluster_id in range(self.best_k):
            mask = self.labels == cluster_id
            ax_2d.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[palette(cluster_id)],
                label=f"Cluster {cluster_id} (n={mask.sum()})",
                alpha=0.65,
                s=20,
            )

        ax_2d.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c="#ff00ff",
            marker="*",
            s=320,
            edgecolors="#111111",
            label="Centroids",
            zorder=5,
        )
        ax_2d.set_xlabel(f"PC 1 ({exp_var_2d[0]:.2f}%)")
        ax_2d.set_ylabel(f"PC 2 ({exp_var_2d[1]:.2f}%)")
        ax_2d.set_title("PCA 2D Scatter")
        ax_2d.legend(loc="best", fontsize=9, framealpha=0.9)

        if n_dims >= 3:
            pca_3d = PCA(n_components=3, random_state=self.random_state)
            coords_3d = pca_3d.fit_transform(X)
            centers_3d = pca_3d.transform(self.model.cluster_centers_)
            exp_var_3d = pca_3d.explained_variance_ratio_ * 100

            ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
            for cluster_id in range(self.best_k):
                mask = self.labels == cluster_id
                ax_3d.scatter(
                    coords_3d[mask, 0],
                    coords_3d[mask, 1],
                    coords_3d[mask, 2],
                    c=[palette(cluster_id)],
                    alpha=0.55,
                    s=15,
                )

            ax_3d.scatter(
                centers_3d[:, 0],
                centers_3d[:, 1],
                centers_3d[:, 2],
                c="#ff00ff",
                marker="*",
                s=320,
                edgecolors="#111111",
                zorder=5,
            )
            ax_3d.set_xlabel(f"PC 1 ({exp_var_3d[0]:.1f}%)")
            ax_3d.set_ylabel(f"PC 2 ({exp_var_3d[1]:.1f}%)")
            ax_3d.set_zlabel(f"PC 3 ({exp_var_3d[2]:.1f}%)")
            ax_3d.set_title("PCA 3D Scatter")

        fig.suptitle(f"K-Means Clustering Summary (k={self.best_k})", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_clusters.png", dpi=250)

    def plot_results(self, X, save_prefix="cluster_report"):
        """Формирует и сохраняет графики метрик и PCA-визуализации."""
        plt.style.use("bmh")
        self._plot_metrics_panel(save_prefix)
        self._plot_pca_panel(X, save_prefix)
        print(f"[INFO] Графики сохранены: {save_prefix}_metrics.png и {save_prefix}_clusters.png")
        plt.show()

