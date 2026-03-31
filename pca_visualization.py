# ======================
# 2D PCA Visualization of Best Clusters (Qualitative Analysis)
# ======================
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from download_data import get_subset
from figures import best_gmm, best_ms
from main import (
    apply_pca, GaussianMixture, mean_shift_clustering,
    pca_to_samples, X, y
)

# ======================
# Compute best clustering results
# ======================
X_best = {}
labels_best = {}

# ----- GMM -----
X_gmm_high, _ = apply_pca(
    get_subset(pca_to_samples[best_gmm['pca_dim']], X, y)[0],
    int(best_gmm['pca_dim'])
)
labels_best['GMM'] = GaussianMixture(
    X_gmm_high, int(best_gmm['K']), iterations=30
)

# ----- Mean Shift -----
X_ms_high, _ = apply_pca(
    get_subset(pca_to_samples[best_ms['pca_dim']], X, y)[0],
    int(best_ms['pca_dim'])
)
labels_best['MeanShift'] = mean_shift_clustering(
    X_ms_high, best_ms['K']  # K contiene la bandwidth
)

# ======================
# PCA to 2D for visualization only
# ======================
pca_2d = PCA(n_components=2, random_state=42)
X_best['GMM'] = pca_2d.fit_transform(X_gmm_high)
X_best['MeanShift'] = pca_2d.fit_transform(X_ms_high)

# ======================
# Plot
# ======================
plt.figure(figsize=(12, 5))

# ---------- GMM ----------
plt.subplot(1, 2, 1)
plt.scatter(
    X_best['GMM'][:, 0], X_best['GMM'][:, 1],
    c=labels_best['GMM'], cmap='tab10',
    s=12, alpha=0.7
)
plt.title(
    f"GMM – 2D PCA Projection"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.4)

# ---------- Mean Shift ----------
plt.subplot(1, 2, 2)
plt.scatter(
    X_best['MeanShift'][:, 0], X_best['MeanShift'][:, 1],
    c=labels_best['MeanShift'], cmap='tab10',
    s=12, alpha=0.7
)
plt.title(
    f"Mean Shift – 2D PCA Projection"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.4)

plt.tight_layout()
plt.savefig("figures/clustering_2d.png", dpi=300)
plt.show()
