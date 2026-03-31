import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

from download_data import load_mnist_full, get_subset
from main import GaussianMixture

sns.set(style="whitegrid", context="talk")

# Load data
df = pd.read_csv("clustering_results.csv")

def best_result(df, model):
    if model in ['GMM', 'NormalizedCut']:
        # per GMM e NC prendiamo il Rand Index massimo tra i valori disponibili
        subset = df[df['model'] == model]
        idx_max = subset['rand_index_mean'].idxmax()
        return subset.loc[idx_max]
    elif model == 'MeanShift':
        # per Mean Shift, la colonna K contiene la bandwidth
        subset = df[df['model'] == model]
        idx_max = subset['rand_index_mean'].idxmax()
        return subset.loc[idx_max]

# Ottieni i migliori risultati
best_gmm = best_result(df, 'GMM')
best_ncut = best_result(df, 'NormalizedCut')
best_ms = best_result(df, 'MeanShift')

summary_df = pd.DataFrame([best_gmm, best_ncut, best_ms])

# Grafico a barre con error bar
plt.figure(figsize=(8,5))
plt.bar(summary_df['model'], summary_df['rand_index_mean'], #yerr=summary_df['rand_index_std'],
        capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel("Rand Index (best)")
plt.title("Comparison of Best Clustering Results")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("figures/best_clustering_comparison.png", dpi=300)


plt.figure(figsize=(8,5))
plt.bar(summary_df['model'], summary_df['time'], capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel("Time (best)")
plt.yscale('log')
plt.title("Comparison of Best Clustering Results (log scale)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("figures/best_clustering_time_comparison.png", dpi=300)


PCA_DIM = 200
K = 10
N_SAMPLES = 8000
ITERATIONS = 30

# ======================
# LOAD DATA
# ======================
X, y = load_mnist_full()
X_sub, _ = get_subset(N_SAMPLES, X, y)

# ======================
# PCA
# ======================
pca = PCA(n_components=PCA_DIM, random_state=42)
X_pca = pca.fit_transform(X_sub)

# ======================
# GMM
# ======================
labels, mu, _, _ = GaussianMixture(X_pca, K, ITERATIONS)

# ======================
# PCA RECONSTRUCTION
# ======================
mu_reconstructed = pca.inverse_transform(mu)

# ======================
# VISUALIZATION
# ======================
fig, axes = plt.subplots(1, K, figsize=(1.5*K, 2))

for k in range(K):
    axes[k].imshow(mu_reconstructed[k].reshape(28, 28), cmap='gray')
    axes[k].axis('off')
    axes[k].set_title(f'C{k}')

plt.tight_layout()
plt.savefig("figures/gmm_means.png", dpi=300)
plt.show()