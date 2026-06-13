import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ---------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------------
def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def wcss(cities, labels, centroids):
    return sum(distance(cities[i], centroids[labels[i]]) ** 2 for i in range(len(cities)))


# ---------------------- АВТОМАТИЧЕСКИЙ ВЫБОР K (метод локтя) ----------------------
def find_optimal_k(cities, max_K=10):
    cities_arr = np.array(cities)
    max_K = min(max_K, len(set(map(tuple, cities_arr))) - 1)
    if max_K < 2:
        return 1, [0]

    wcss_values = []
    for k in range(1, max_K + 1):
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        km.fit(cities_arr)
        wcss_values.append(km.inertia_)

    first = (1, wcss_values[0])
    last = (max_K, wcss_values[-1])
    max_dist, opt_k = -1, 1
    for k in range(2, max_K + 1):
        x0, y0 = k, wcss_values[k - 1]
        area = abs((last[0] - first[0]) * (first[1] - y0) - (first[0] - x0) * (last[1] - first[1]))
        base = np.hypot(last[0] - first[0], last[1] - first[1])
        dist = area / base if base != 0 else 0
        if dist > max_dist:
            max_dist, opt_k = dist, k

    return opt_k, wcss_values


# ---------------------- 1. АЛГОРИТМИЧЕСКАЯ ГРУППИРОВКА (жадный алгоритм) ----------------------
def algo_clustering(cities):
    # Радиус выбирается автоматически: среднее расстояние между всеми парами * 0.5
    n = len(cities)
    all_dists = [distance(cities[i], cities[j]) for i in range(n) for j in range(i + 1, n)]
    radius = np.mean(all_dists) * 0.5

    assigned = [False] * n
    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i]:
            continue
        # Берём город i как центр нового кластера, добавляем всех соседей в радиусе
        for j in range(n):
            if not assigned[j] and distance(cities[i], cities[j]) <= radius:
                labels[j] = cluster_id
                assigned[j] = True
        cluster_id += 1

    opt_k = cluster_id
    centroids = np.array([
        np.mean([cities[j] for j in range(n) if labels[j] == k], axis=0)
        for k in range(opt_k)
    ])

    print(f"  algo_clustering:  автоматически выбрано K={opt_k} (радиус={radius:.2f})")
    return np.array(labels), centroids, opt_k, None


# ---------------------- 2. РУЧНОЙ K-MEANS ----------------------
def my_kmeans(cities, max_iter=100):
    cities = np.array(cities)
    K, wcss_elbow = find_optimal_k(cities)
    print(f"  my_kmeans:        автоматически выбрано K={K}")
    idx = random.sample(range(len(cities)), K)
    centroids = cities[idx].astype(float)
    for _ in range(max_iter):
        labels = np.array([np.argmin([distance(c, cen) for cen in centroids]) for c in cities])
        new = np.zeros_like(centroids)
        for k in range(K):
            cluster = cities[labels == k]
            if len(cluster) > 0:
                new[k] = np.mean(cluster, axis=0)
            else:
                new[k] = centroids[k]
        if np.allclose(centroids, new):
            break
        centroids = new
    return labels, centroids, K, wcss_elbow


# ---------------------- 3. SKLEARN K-MEANS ----------------------
def sklearn_kmeans(cities):
    K, wcss_elbow = find_optimal_k(cities)
    print(f"  sklearn_kmeans:   автоматически выбрано K={K}")
    kmeans = KMeans(n_clusters=K, n_init='auto', random_state=42)
    kmeans.fit(cities)
    return kmeans.labels_, kmeans.cluster_centers_, K, wcss_elbow


# ---------------------- ТЕСТИРОВАНИЕ И ВИЗУАЛИЗАЦИЯ ----------------------
def test_methods(cities, name):
    print(f"\n{name}: N={len(cities)}")
    methods = [algo_clustering, my_kmeans, sklearn_kmeans]
    results = []

    for method in methods:
        try:
            start = time.time()
            labels, cents, opt_k, wcss_elbow = method(cities)
            elapsed = time.time() - start
            w = wcss(cities, labels, cents)
            results.append((method.__name__, labels, cents, elapsed, w, opt_k, wcss_elbow))
            print(f"  {method.__name__:15} время={elapsed:.4f}с  WCSS={w:.1f}")
        except Exception as e:
            print(f"  {method.__name__:15} ОШИБКА: {str(e)}")
            return

    # --- Одна фигура: верхний ряд — кластеры, нижний ряд — метод локтя ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(name, fontsize=16)

    # Верхний ряд — кластеры
    for ax, (mname, labels, cents, _, _, opt_k, _) in zip(axes[0], results):
        K = len(cents)
        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
        ax.set_title(f"{mname} (K={K})")
        for k in range(K):
            pts = [cities[i] for i in range(len(cities)) if labels[i] == k]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=colors[k], alpha=0.6, s=50)
        ax.scatter(cents[:, 0], cents[:, 1], c='black', marker='X', s=200, label='Центроиды')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    # Нижний ряд — метод локтя для my_kmeans и sklearn_kmeans, первая ячейка пуста
    axes[1][0].axis('off')
    elbow_results = [(mname, opt_k, wcss_elbow)
                     for mname, _, _, _, _, opt_k, wcss_elbow in results
                     if wcss_elbow is not None]

    for ax, (mname, opt_k, wcss_elbow) in zip(axes[1][1:], elbow_results):
        K_range = range(1, len(wcss_elbow) + 1)
        ax.plot(list(K_range), wcss_elbow, 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=opt_k, color='r', linestyle='--', label=f'Оптимальное K={opt_k}')
        ax.scatter(opt_k, wcss_elbow[opt_k - 1], color='red', s=150, zorder=5)
        ax.set_title(f"{mname} — метод локтя")
        ax.set_xlabel("Количество кластеров (K)")
        ax.set_ylabel("WCSS")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


# ---------------------- НАБОР ТЕСТОВ ----------------------
def run_all_tests():
    random.seed(42)
    np.random.seed(42)

    # Тест 1: Случайные 50 точек
    cities1 = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)]

    # Тест 2: Три разнесённых кластера
    cities2 = [(0, 0), (1, 0), (0, 1), (1, 1),
               (10, 10), (11, 10), (10, 11), (11, 11),
               (20, 20), (21, 20), (20, 21), (21, 21)]

    # Тест 3: Одна точка повторена
    cities3 = [(5, 5)] * 10

    # Тест 4: Два кластера разного размера
    cluster1 = [(random.gauss(20, 5), random.gauss(20, 5)) for _ in range(20)]
    cluster2 = [(random.gauss(60, 5), random.gauss(60, 5)) for _ in range(10)]
    cities4 = cluster1 + cluster2

    tests = [
        (cities1, "Случайные 50 точек"),
        (cities2, "Три разнесённых кластера"),
        (cities3, "Одна точка повторена"),
        (cities4, "Два кластера разного размера"),
    ]

    for data, desc in tests:
        test_methods(data, desc)


if __name__ == "__main__":
    run_all_tests()