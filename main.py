from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA
import numpy as np
import time
import csv
import os
from datetime import datetime
from download_data import load_mnist_full, get_subset


# PARAMETERS
pca_dims = [2, 5, 10, 20, 50, 100, 200]
K_values = range(5, 16)
n_runs = 10
csv_file = "clustering_results.csv"
pca_to_samples = {
    2:   15000,
    5:   15000,
    10:  12000,
    20:  10000,
    50:   8000,
    100:  6000,
    200:  5000
}

pca_to_bandwidths = {
    2:   [0.1, 0.3, 0.5, 0.8],
    5:   [0.5, 0.8, 1.0, 1.5],
    10:  [0.8, 1.5, 2.0, 3.0],
    20:  [2.0, 2.5, 3.0, 4.0],
    50:  [3.0, 4.0, 4.5, 5.0],
    100: [4.0, 4.5, 5.0, 5.5],
    200: [4.0, 5.0, 5.5, 6.0]
}


# DATA
X, y = load_mnist_full()


def apply_pca(X, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    start = time.time()
    X_red = pca.fit_transform(X)
    elapsed = time.time() - start
    return X_red, elapsed

# Non-optimized rand index O(n^2)
def rand_index(labels_true, labels_pred):
    n = len(labels_true)
    a = 0
    b = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_class = labels_true[i] == labels_true[j]
            same_cluster = labels_pred[i] == labels_pred[j]
            if same_class and same_cluster:
                a += 1
            elif not same_class and not same_cluster:
                b += 1
    return 2 * (a + b) / (n * (n - 1))

# Optimized rand index using contingency matrix O(n)
def rand_index_fast(labels_true, labels_pred):
    # contingency[i][j] = number of elements of class i assigned to the predicted cluster j
    contingency = contingency_matrix(labels_true, labels_pred)
    n = np.sum(contingency)
    a = np.sum(contingency * (contingency - 1)) / 2 # number of couples of elements same cluster and class
    sum_c = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1)) / 2 # number of same class couples
    sum_k = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1)) / 2 # number of same cluster couples
    total_pairs = n * (n - 1) / 2 # number of total couples
    b = (total_pairs - sum_c - sum_k + a) # b = total couples - same class couples - same cluster couples + a (because sum_c + sum_k includes a)
    return (a + b) / total_pairs


def GaussianMixture(X, K, iterations):
    n, d = X.shape
    mu = X[np.random.choice(n, K, replace=False)]
    variances = np.ones((K, d))
    pi = np.ones(K) / K

    for _ in range(iterations):
        log_gamma = np.zeros((n, K))
        for k in range(K):
            log_coeff = -0.5 * np.sum(np.log(2 * np.pi * variances[k]))
            log_exponent = -0.5 * np.sum((X - mu[k]) ** 2 / variances[k], axis=1)
            log_gamma[:, k] = np.log(pi[k]) + log_coeff + log_exponent

        log_gamma_max = log_gamma.max(axis=1, keepdims=True)
        gamma = np.exp(log_gamma - log_gamma_max)

        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-12
        gamma /= gamma_sum

        Nk = gamma.sum(axis=0)
        pi = Nk / n
        mu = (gamma.T @ X) / Nk[:, None]

        for k in range(K):
            diff = X - mu[k]
            variances[k] = (gamma[:, k][:, None] * diff**2).sum(axis=0) / Nk[k]

        variances += 1e-3

    labels = np.argmax(gamma, axis=1)
    return labels, mu, variances, pi


def mean_shift_clustering(X, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    return ms.fit_predict(X)


def normalized_cut_clustering(X, K, seed):
    sc = SpectralClustering(
        n_clusters=K,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans',
        random_state=seed
    )
    return sc.fit_predict(X)


def main():
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "model", "pca_dim", "K",
                "run", "rand_index_mean", "rand_index_std", "time", "n_clusters"
            ])


    timestamp = datetime.now().isoformat()

    for d in pca_dims:
        print(f"\n=== PCA dim = {d} ===")
        X_sub, y_sub = get_subset(pca_to_samples[d], X, y)
        X_pca, pca_time = apply_pca(X_sub, d)

        # GMM
        for K in K_values:
            RI_list = []
            times = []
            for run in range(n_runs):
                start = time.time()
                labels = GaussianMixture(X_pca, K, iterations=30)
                elapsed = time.time() - start
                R = rand_index_fast(y_sub, labels)
                RI_list.append(R)
                times.append(elapsed)
                print(f"GMM | PCA={d} | k={K} | run={run} | RI={R:.3f}")

            RI_mean = np.mean(RI_list)
            RI_std = np.std(RI_list)
            time_mean = np.mean(times)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, "GMM", d, K, "mean", RI_mean, RI_std, time_mean, K
                ])


        # NORMALIZED CUT
        for K in K_values:
            start = time.time()
            labels = normalized_cut_clustering(X_pca, K, 1)
            elapsed = time.time() - start
            R = rand_index_fast(y_sub, labels)
            print(f"NCut | PCA={d} | k={K} | run={run} | RI={R:.3f}")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, "NormalizedCut", d, K, "mean", R, 0, elapsed, K
                ])


        # MEAN SHIFT
        for bw in pca_to_bandwidths[d]:
            start = time.time()
            labels = mean_shift_clustering(X_pca, bw)
            elapsed = time.time() - start
            RI = rand_index_fast(y_sub, labels)
            n_clusters = len(np.unique(labels))

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, "MeanShift", d, bw, "single", RI, 0, elapsed, n_clusters
                ])

            print(f"MeanShift | PCA={d} | bw={bw} | clusters={n_clusters} | RI={RI:.3f}")


if __name__ == '__main__':
    main()